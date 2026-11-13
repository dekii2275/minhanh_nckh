"""Compare top-confidence and GT-bbox-assisted YOLO pose on seed-0 validation.

GT is used at inference time only for this diagnostic. No official test data.
The same detections, preprocessing and GT-trained classifier are paired.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import scipy
import scipy.io as sio
import torch
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[2] / 'outputs' / 'ultralytics_config'))
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from data.action_dataset import ACTION_CLASSES, normalize_skeleton_sequence
from models.action_bilstm import ActionBiLSTMAttentionClassifier

MANIFEST = ROOT / 'report/02112026_nckh/penn_audit_artifacts/video_manifest.csv'
MODEL = ROOT / 'models/yolo11n-pose.pt'
LABELS = ROOT / 'data/raw/Penn_Action/labels'
FRAMES = ROOT / 'data/raw/Penn_Action/frames'
CKPTS = ROOT / 'outputs/gt_baselines_20261102_v1/bilstm_attention'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        return [r for r in csv.DictReader(f) if r['seed0_split'] == 'validation']

def ious(boxes, gt):
    if len(boxes) == 0: return np.empty(0, np.float32)
    if not np.isfinite(gt).all() or gt[2] <= gt[0] or gt[3] <= gt[1]:
        return np.full(len(boxes), np.nan, np.float32)
    wh = np.maximum(0., np.minimum(boxes[:, 2:], gt[2:]) - np.maximum(boxes[:, :2], gt[:2]))
    inter = wh[:, 0]*wh[:, 1]
    area = np.maximum(0., boxes[:, 2]-boxes[:, 0])*np.maximum(0., boxes[:, 3]-boxes[:, 1])
    union = area + (gt[2]-gt[0])*(gt[3]-gt[1]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def infer(args, samples, out):
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = out/'detections'; pred_dir.mkdir(exist_ok=True)
    info_path = out/'provenance.json'
    info = {'model_sha256':sha(MODEL), 'manifest_sha256':sha(MANIFEST),
            'script_sha256':sha(Path(__file__)), 'imgsz':640, 'confidence':0.25,
            'batch':args.batch, 'device':args.device, 'torch':torch.__version__,
            'ultralytics':ultralytics.__version__, 'opencv':cv2.__version__,
            'numpy':np.__version__, 'scipy':scipy.__version__,
            'scope':'seed0 validation only', 'selection':'argmax IoU with GT bbox; >= 0.30 else missing',
            'classifier':'GT-trained BiLSTM+attention, seed 0/1/2',
            'checkpoints':{str(s):sha(CKPTS/f'seed{s}'/'best.pt') for s in (0,1,2)}}
    if info_path.exists():
        if json.loads(info_path.read_text(encoding='utf-8')) != info:
            raise ValueError('Provenance mismatch; choose new output directory')
    else:
        info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    model = YOLO(str(MODEL)); start=time.time()
    for k, row in enumerate(samples, 1):
        sid=row['id']; target=pred_dir/f'{sid}.npz'
        if target.exists():
            print(f'{k}/{len(samples)} {sid} cached', flush=True); continue
        label=LABELS/f'{sid}.mat'
        if sha(label)!=row['label_sha256']: raise ValueError('GT hash mismatch '+sid)
        mat=sio.loadmat(label)
        if int(mat['train'][0,0])!=1 or str(mat['action'][0])!=row['action']:
            raise ValueError('label/split mismatch '+sid)
        T=int(row['nframes']); gt=np.asarray(mat['bbox'], dtype=np.float32)
        if gt.shape != (T,4): raise ValueError(f'Bad bbox shape {sid}: {gt.shape}')
        boxes=[]; xy=[]; conf=[]; offsets=[0]; frame_hash=hashlib.sha256()
        for start_idx in range(0,T,args.batch):
            ims=[]
            for idx in range(start_idx,min(T,start_idx+args.batch)):
                path=FRAMES/sid/f'{idx+1:06d}.jpg'; raw=path.read_bytes()
                frame_hash.update(len(raw).to_bytes(8,'big')); frame_hash.update(raw)
                im=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                if im is None: raise ValueError('decode failed '+str(path))
                ims.append(im)
            results=model.predict(source=ims,imgsz=640,conf=.25,device=args.device,verbose=False)
            if len(results)!=len(ims): raise RuntimeError('YOLO output count mismatch')
            for result in results:
                n=len(result.boxes)
                if n:
                    boxes.append(np.column_stack((result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())).astype(np.float32))
                    xy.append(result.keypoints.xy.cpu().numpy().astype(np.float32))
                    conf.append(result.keypoints.conf.cpu().numpy().astype(np.float32))
                offsets.append(offsets[-1]+n)
        all_boxes=np.concatenate(boxes) if boxes else np.empty((0,5),np.float32)
        all_xy=np.concatenate(xy) if xy else np.empty((0,17,2),np.float32)
        all_conf=np.concatenate(conf) if conf else np.empty((0,17),np.float32)
        tmp=pred_dir/f'{sid}.tmp.npz'
        np.savez_compressed(tmp, offsets=np.asarray(offsets),boxes=all_boxes,xy=all_xy,conf=all_conf,gt_bbox=gt,
                            image_bytes_sha256=frame_hash.hexdigest(),annotation_sha256=row['label_sha256'])
        os.replace(tmp,target)
        print(f'{k}/{len(samples)} {sid} frames={T} detections={len(all_boxes)} elapsed={time.time()-start:.1f}s',flush=True)

def resample(coords):
    feats=normalize_skeleton_sequence(coords)
    t=np.linspace(0,len(feats)-1,64)
    return np.stack([np.interp(t,np.arange(len(feats)),feats[:,j]) for j in range(24)],axis=-1).astype(np.float32)

def metric(target, predicted):
    y=np.asarray(target); p=np.asarray(predicted)
    f=[]
    for i in range(15):
        tp=np.sum((y==i)&(p==i)); fp=np.sum((y!=i)&(p==i)); fn=np.sum((y==i)&(p!=i))
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'accuracy_pct':100*float(np.mean(y==p)), 'macro_f1_pct':100*float(np.mean(f))}

def evaluate(samples, out, threshold):
    pred_dir=out/'detections'; results=[]; videos=[]; frame_rows=[]
    for row in samples:
        sid=row['id']; p=pred_dir/f'{sid}.npz'
        if not p.exists(): raise FileNotFoundError(str(p))
        d=np.load(p); T=int(row['nframes']); offsets=d['offsets']; boxes=d['boxes']; xy=d['xy']; gt=d['gt_bbox']
        if len(offsets)!=T+1 or str(d['annotation_sha256'])!=row['label_sha256']: raise ValueError('cache mismatch '+sid)
        max_pose=np.full((T,12,2),np.nan,np.float32); gt_pose=max_pose.copy()
        changed=0; covered=0; max_matched=0; nd=0; invalid_gt=0
        for t in range(T):
            lo,hi=offsets[t:t+2]; b=boxes[lo:hi]; scores=ious(b[:,:4],gt[t]); n=len(b)
            if not np.isfinite(gt[t]).all() or gt[t,2]<=gt[t,0] or gt[t,3]<=gt[t,1]: invalid_gt+=1
            best=int(np.argmax(b[:,4])) if n else -1
            target=int(np.argmax(scores)) if n and np.isfinite(scores).all() and np.max(scores)>=threshold else -1
            if best>=0: max_pose[t]=xy[lo+best,5:17]; nd+=1
            if target>=0: gt_pose[t]=xy[lo+target,5:17]; covered+=1
            best_iou=float(scores[best]) if best>=0 and np.isfinite(scores[best]) else None
            if best_iou is not None and best_iou>=threshold: max_matched+=1
            if target>=0 and target!=best: changed+=1
            frame_rows.append({'id':sid,'frame':t+1,'detections':n,'max_index':best,'gt_index':target,
                               'max_iou':best_iou,'gt_iou':float(scores[target]) if target>=0 else None})
        videos.append({'id':sid,'action':row['action'],'target':ACTION_CLASSES.index(row['action']),
                       'frames':T,'detected_frames':nd,'gt_matched_frames':covered,'max_matched_frames':max_matched,
                       'changed_frames':changed,'invalid_gt_bbox_frames':invalid_gt,
                       'confidence_features':resample(max_pose), 'gt_features':resample(gt_pose)})
    device='cuda' if torch.cuda.is_available() else 'cpu'
    labels=[v['target'] for v in videos]
    x1=torch.tensor(np.stack([v['confidence_features'] for v in videos]),device=device)
    x2=torch.tensor(np.stack([v['gt_features'] for v in videos]),device=device)
    by_seed=[]; rng=np.random.default_rng(0)
    for seed in (0,1,2):
        net=ActionBiLSTMAttentionClassifier().to(device)
        net.load_state_dict(torch.load(CKPTS/f'seed{seed}'/'best.pt',map_location=device,weights_only=True)); net.eval()
        predictions=[]
        with torch.inference_mode():
            for x in (x1,x2):
                y=[]
                for start in range(0,len(x),32): y.extend(net(x[start:start+32]).argmax(-1).cpu().tolist())
                predictions.append(y)
        baseline, assisted=predictions
        diff=(np.asarray(assisted)==labels).astype(int)-(np.asarray(baseline)==labels).astype(int)
        boot=np.mean(diff[rng.integers(0,len(labels),size=(10000,len(labels)))],axis=1)*100
        item={'seed':seed,'confidence':metric(labels,baseline),'gt_assisted':metric(labels,assisted),
              'accuracy_difference_pp':100*float(np.mean(diff)),
              'paired_video_bootstrap_95_pct_pp':np.quantile(boot,[.025,.975]).tolist(),
              'improved_videos':int((diff>0).sum()),'worsened_videos':int((diff<0).sum()),
              'both_right':int(((np.asarray(assisted)==labels)&(np.asarray(baseline)==labels)).sum())}
        by_seed.append(item)
        for v,a,b in zip(videos,baseline,assisted): results.append({'seed':seed,'id':v['id'],'action':v['action'],'target':v['target'],'confidence_pred':a,'gt_assisted_pred':b})
    summary={'videos':len(videos),'frames':sum(v['frames'] for v in videos),'iou_threshold':threshold,
             'detected_frames':sum(v['detected_frames'] for v in videos),'gt_matched_frames':sum(v['gt_matched_frames'] for v in videos),
             'max_matched_frames':sum(v['max_matched_frames'] for v in videos),'changed_frames':sum(v['changed_frames'] for v in videos),
             'videos_with_changed_frames':sum(v['changed_frames']>0 for v in videos),
             'invalid_gt_bbox_frames':sum(v['invalid_gt_bbox_frames'] for v in videos),
             'seeds':by_seed,'accuracy_difference_mean_pp':float(np.mean([s['accuracy_difference_pp'] for s in by_seed])),
             'warning':'GT matching is an imperfect diagnostic; no independent test; NaN if no match; no pose confidence threshold; no LR swap.'}
    suffix=str(threshold).replace('.','_')
    (out/f'summary_iou_{suffix}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for filename,data in [(f'frame_matches_iou_{suffix}.csv',frame_rows),(f'predictions_iou_{suffix}.csv',results),
                          (f'videos_iou_{suffix}.csv',[{k:v for k,v in row.items() if not k.endswith('_features')} for row in videos])]:
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--batch',type=int,default=16); p.add_argument('--device',default='0'); p.add_argument('--limit',type=int); p.add_argument('--evaluate-only',action='store_true'); p.add_argument('--iou',type=float,default=.3); args=p.parse_args()
    if args.batch<1 or not 0<=args.iou<=1: p.error('Invalid batch or IoU')
    samples=rows(); samples=samples[:args.limit] if args.limit else samples
    out=Path(args.output)
    if not args.evaluate_only: infer(args,samples,out)
    evaluate(samples,out,args.iou)

if __name__=='__main__': main()
