"""GT-assisted diagnostic: missing pose vs interpolation vs optical-flow ROI recovery.

Prior GT-selected detections define anchors AND eligible gaps, so none of these
results is a standalone, autonomous multi-person tracking evaluation.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import compare_person_selection as base
from ultralytics import YOLO

SOURCE = base.ROOT/'outputs/person_selection_validation_20261102'
KP = slice(5,17)

def gt_selected(row):
    d=np.load(SOURCE/'detections'/f"{row['id']}.npz")
    T=int(row['nframes']); pose=np.full((T,12,2),np.nan,np.float32)
    conf=np.zeros((T,12),np.float32); boxes=np.full((T,4),np.nan,np.float32)
    for t in range(T):
        lo,hi=d['offsets'][t:t+2]; b=d['boxes'][lo:hi]
        iou=base.ious(b[:,:4],d['gt_bbox'][t])
        if len(b) and np.isfinite(iou).all() and iou.max()>=.3:
            idx=lo+int(iou.argmax()); pose[t]=d['xy'][idx,KP]; conf[t]=d['conf'][idx,KP]; boxes[t]=d['boxes'][idx,:4]
    return pose,conf,boxes,d

def eligible_gaps(boxes, max_gap=3):
    missing=~np.isfinite(boxes).all(axis=1); gaps=[]; i=0
    while i<len(missing):
        if not missing[i]: i+=1; continue
        j=i
        while j<len(missing) and missing[j]: j+=1
        if i>0 and j<len(missing) and j-i<=max_gap: gaps.append((i,j))
        i=j
    return gaps

def interpolate(pose,conf,gaps):
    result=pose.copy(); rc=conf.copy()
    for start,end in gaps:
        good=(conf[start-1]>0)&(conf[end]>0)&np.all(pose[start-1]>0,axis=-1)&np.all(pose[end]>0,axis=-1)
        for t in range(start,end):
            weight=(t-start+1)/(end-start+1)
            result[t,good]=(1-weight)*pose[start-1,good]+weight*pose[end,good]
            rc[t,good]=np.minimum(conf[start-1,good],conf[end,good])
    return result,rc

def image(sid,t):
    p=base.FRAMES/sid/f'{t+1:06d}.jpg'
    im=cv2.imread(str(p))
    if im is None: raise ValueError(f'Cannot read {p}')
    return im

def verify_images(row, expected):
    digest=hashlib.sha256()
    for t in range(int(row['nframes'])):
        raw=(base.FRAMES/row['id']/f'{t+1:06d}.jpg').read_bytes()
        digest.update(len(raw).to_bytes(8,'big')); digest.update(raw)
    if digest.hexdigest()!=expected: raise ValueError('Frame hash mismatch '+row['id'])

def flow_box(prev, curr, box):
    gray=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY); nextgray=cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape; x1,y1,x2,y2=box
    a=max(0,int(x1)); b=max(0,int(y1)); c=min(w,int(np.ceil(x2))); e=min(h,int(np.ceil(y2)))
    if c-a<2 or e-b<2: return None,0
    mask=np.zeros((h,w),np.uint8); mask[b:e,a:c]=255
    pts=cv2.goodFeaturesToTrack(gray,maxCorners=100,qualityLevel=.01,minDistance=3,mask=mask)
    if pts is None or len(pts)<4: return None,0
    nxt,status,_=cv2.calcOpticalFlowPyrLK(gray,nextgray,pts,None)
    if nxt is None or status is None: return None,0
    ok=status.reshape(-1)>0; delta=(nxt-pts).reshape(-1,2)[ok]
    if len(delta)<4: return None,len(delta)
    dx,dy=np.median(delta,axis=0); return box+np.array([dx,dy,dx,dy],np.float32),len(delta)

def recover_roi(model, sid, gaps, boxes, pose, conf):
    result=pose.copy(); rc=conf.copy(); records=[]
    for start,end in gaps:
        box=boxes[start-1].copy(); prev=image(sid,start-1)
        for t in range(start,end):
            curr=image(sid,t); moved,count=flow_box(prev,curr,box)
            rec={'id':sid,'frame':t+1,'flow_points':count,'status':'flow_failed','crop_detections':0,'selected_iou':None,'box_confidence':None,'box_x1':None,'box_y1':None,'box_x2':None,'box_y2':None}
            if moved is None:
                records.append(rec); break
            box=moved; h,w=curr.shape[:2]; center=(box[:2]+box[2:])/2; size=(box[2:]-box[:2])*1.5
            x1=max(0,int(np.floor(center[0]-size[0]/2))); y1=max(0,int(np.floor(center[1]-size[1]/2)))
            x2=min(w,int(np.ceil(center[0]+size[0]/2))); y2=min(h,int(np.ceil(center[1]+size[1]/2)))
            if x2-x1<4 or y2-y1<4:
                rec['status']='empty_crop'; records.append(rec); break
            r=model.predict(source=curr[y1:y2,x1:x2],imgsz=640,conf=.1,device='0',verbose=False)[0]
            n=len(r.boxes); rec['crop_detections']=n; rec['status']='no_matching_detection'
            if n:
                bb=r.boxes.xyxy.cpu().numpy(); bb[:,[0,2]]+=x1; bb[:,[1,3]]+=y1
                iou=base.ious(bb,box)
                if len(iou) and np.isfinite(iou).all() and iou.max()>=.1:
                    idx=int(iou.argmax()); pts=r.keypoints.xy[idx,KP].cpu().numpy().copy(); cf=r.keypoints.conf[idx,KP].cpu().numpy()
                    # Ultralytics uses (0,0) for missing keypoints; do not translate those.
                    valid=np.all(pts>0,axis=-1)&(cf>0)
                    pts[valid]+=np.array([x1,y1]); pts[~valid]=np.nan
                    result[t]=pts; rc[t]=cf; box=bb[idx]
                    rec.update(status='recovered',selected_iou=float(iou[idx]),box_confidence=float(r.boxes.conf[idx]),box_x1=float(box[0]),box_y1=float(box[1]),box_x2=float(box[2]),box_y2=float(box[3]))
            records.append(rec); prev=curr
    return result,rc,records

def pose_stats(gt, visibility, gtbox, pred, conf, eligible):
    wh=gtbox[:,2:]-gtbox[:,:2]; diag=np.linalg.norm(wh,axis=1)
    valid_box=np.isfinite(gtbox).all(axis=1)&(wh[:,0]>0)&(wh[:,1]>0)
    gt_ok=(visibility>0)&np.isfinite(gt).all(axis=-1)&(gt[:,:,0]>0)&(gt[:,:,1]>0)&valid_box[:,None]
    pred_ok=np.isfinite(pred).all(axis=-1)&(pred[:,:,0]>0)&(pred[:,:,1]>0)&(conf>0)
    dist=np.linalg.norm(pred-gt,axis=-1)/np.maximum(diag[:,None],1e-9)
    return gt_ok,pred_ok,dist

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--limit',type=int); args=ap.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True); (out/'per_video').mkdir()
    provenance=json.loads((SOURCE/'provenance.json').read_text(encoding='utf-8'))
    if base.sha(base.MODEL)!=provenance['model_sha256'] or base.sha(base.MANIFEST)!=provenance['manifest_sha256']:
        raise ValueError('Source model or manifest hash mismatch')
    for s in (0,1,2):
        if base.sha(base.CKPTS/f'seed{s}'/'best.pt')!=provenance['checkpoints'][str(s)]: raise ValueError('Checkpoint hash mismatch')
    samples=base.rows(); samples=samples[:args.limit] if args.limit else samples
    (out/'provenance.json').write_text(json.dumps({'script_sha256':base.sha(Path(__file__)),'source_provenance_sha256':base.sha(SOURCE/'provenance.json'),'max_gap':3,'source_iou':.3,'crop_expansion':1.5,'crop_yolo_conf':.1,'crop_match_iou':.1,'flow_min_points':4,'track':'median Lucas-Kanade optical flow of features in last bbox; YOLO on crop','scope':'GT-assisted, offline short internal gaps only; bbox anchors and missing gaps use GT'},indent=2),encoding='utf-8')
    model=YOLO(str(base.MODEL)); started=time.time(); feats={k:[] for k in ('missing','interpolation','flow_crop')}; labels=[]; ids=[]
    sums={k:{'gt_joints':0,'pred_joints':0,'pck_correct':0,'error_sum':0.,'gap_gt_joints':0,'gap_pred_joints':0,'gap_pck_correct':0,'gap_error_sum':0.} for k in feats}
    recovered=[]; n_gaps=0; n_eligible=0
    for ix,row in enumerate(samples,1):
        sid=row['id']; pose,conf,boxes,d=gt_selected(row); gaps=eligible_gaps(boxes)
        if gaps:
            verify_images(row,str(d['image_bytes_sha256']))
        int_pose,int_conf=interpolate(pose,conf,gaps)
        track_pose,track_conf,records=recover_roi(model,sid,gaps,boxes,pose,conf) if gaps else (pose.copy(),conf.copy(),[])
        for rec in records:
            rec['gt_bbox_iou']=None
            if rec['status']=='recovered':
                bb=np.array([[rec['box_x1'],rec['box_y1'],rec['box_x2'],rec['box_y2']]],dtype=np.float32)
                gt_score=base.ious(bb,d['gt_bbox'][rec['frame']-1])
                if np.isfinite(gt_score).all(): rec['gt_bbox_iou']=float(gt_score[0])
        recovered.extend(records); n_gaps+=len(gaps); gap_mask=np.zeros(len(pose),bool)
        for a,b in gaps: gap_mask[a:b]=True
        n_eligible+=int(gap_mask.sum())
        label_path=base.LABELS/f'{sid}.mat'
        if base.sha(label_path)!=row['label_sha256']: raise ValueError('Annotation changed '+sid)
        mat=sio.loadmat(label_path); gt=np.stack([mat['x'][:,1:13],mat['y'][:,1:13]],axis=-1); vis=mat['visibility'][:,1:13]
        variants={'missing':(pose,conf),'interpolation':(int_pose,int_conf),'flow_crop':(track_pose,track_conf)}
        for name,(p,c) in variants.items():
            feats[name].append(base.resample(p)); good,valid,dist=pose_stats(gt,vis,d['gt_bbox'],p,c,gap_mask)
            for mask,prefix in [(np.ones(len(pose),bool),''),(gap_mask,'gap_')]:
                g=good&mask[:,None]; ok=g&valid; correct=ok&(dist<=.2); s=sums[name]
                s[prefix+'gt_joints']+=int(g.sum()); s[prefix+'pred_joints']+=int(ok.sum()); s[prefix+'pck_correct']+=int(correct.sum()); s[prefix+'error_sum']+=float(np.sum(dist[ok]))
        np.savez_compressed(out/'per_video'/f'{sid}.npz',baseline=pose,baseline_conf=conf,interpolation=int_pose,interpolation_conf=int_conf,flow_crop=track_pose,flow_crop_conf=track_conf,eligible_gap=gap_mask)
        labels.append(base.ACTION_CLASSES.index(row['action'])); ids.append(sid)
        if gaps: print(f'{ix}/{len(samples)} {sid} gaps={len(gaps)} frames={gap_mask.sum()} recovered={sum(r["status"]=="recovered" for r in records)} elapsed={time.time()-started:.1f}s',flush=True)
    results={}; predictions=[]; device='cuda' if torch.cuda.is_available() else 'cpu'
    for k,features in feats.items():
        x=torch.tensor(np.stack(features),device=device); seeds=[]
        for seed in (0,1,2):
            net=base.ActionBiLSTMAttentionClassifier().to(device); net.load_state_dict(torch.load(base.CKPTS/f'seed{seed}'/'best.pt',map_location=device,weights_only=True)); net.eval(); pred=[]
            with torch.inference_mode():
                for i in range(0,len(x),32): pred.extend(net(x[i:i+32]).argmax(-1).cpu().tolist())
            metrics=base.metric(labels,pred); seeds.append({'seed':seed,**metrics})
            predictions.extend({'method':k,'seed':seed,'id':sid,'target':y,'predicted':z} for sid,y,z in zip(ids,labels,pred))
        s=sums[k]; pose_metrics={}
        for prefix,tag in [('', 'all_visible'),('gap_','eligible_missing_frames')]:
            gt_n=s[prefix+'gt_joints']; pred_n=s[prefix+'pred_joints']
            pose_metrics[tag]={'gt_visible_joints':gt_n,'predicted_joints':pred_n,'coverage_pct':100*pred_n/gt_n if gt_n else None,
                               'pck20_missing_incorrect_pct':100*s[prefix+'pck_correct']/gt_n if gt_n else None,
                               'pck20_among_predicted_pct':100*s[prefix+'pck_correct']/pred_n if pred_n else None,
                               'mean_normalized_error_among_predicted':s[prefix+'error_sum']/pred_n if pred_n else None}
        results[k]={'seeds':seeds,'mean_accuracy_pct':float(np.mean([v['accuracy_pct'] for v in seeds])),'mean_macro_f1_pct':float(np.mean([v['macro_f1_pct'] for v in seeds])),'pose':pose_metrics}
    summary={'videos':len(samples),'eligible_gaps':n_gaps,'eligible_frames':n_eligible,'flow_attempt_records':len(recovered),'flow_recovered_frames':sum(r['status']=='recovered' for r in recovered),'results':results,'note':'GT-assisted gap selection; IoU matching not guaranteed correct; pose IoU bbox diagnostic; PCK20 uses 0.2*GT bbox diagonal and 12 joints with visibility, identity mapping; not independent test.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,data in [('flow_attempts.csv',recovered),('predictions.csv',predictions)]:
        if data:
            with (out/name).open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
