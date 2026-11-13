"""Visualize the saved GT-assisted YOLO correction experiment frame by frame."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tool/person_selection_viewer'))
from person_selection_viewer import OUT, load_video, read_csv, draw_box, text
sys.path.insert(0,str(ROOT/'tool/swap_error_viewer'))
from swap_error_viewer import frame_metrics, draw_pose
sys.path.insert(0,str(ROOT/'tool/dataset_viewer'))
from dataset_viewer import choose
sys.path.insert(0,str(ROOT/'src/gt_baselines'))
from evaluate_pose_corrections import make_variants
sys.path.insert(0,str(ROOT/'src'))
from data.action_dataset import ACTION_CLASSES

RESULT=ROOT/'outputs/pose_corrections_validation_20261103_v2'
POSE_SOURCE=ROOT/'outputs/missing_pose_validation_20261102/per_video'
METHODS=('interpolate3','mean5','median5','interpolate3_mean5')
LABELS={'interpolate3':'NOI SUY <=3 FRAME','mean5':'TRUNG BINH 5 FRAME',
        'median5':'TRUNG VI 5 FRAME','interpolate3_mean5':'NOI SUY + TRUNG BINH 5'}
JOINTS=('vai trai','vai phai','khuyu trai','khuyu phai','co tay trai','co tay phai',
        'hong trai','hong phai','goi trai','goi phai','co chan trai','co chan phai')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inputs():
    provenance=json.loads((RESULT/'provenance.json').read_text(encoding='utf-8'))
    if sha(OUT/'provenance.json')!=provenance['source_provenance_sha256'] or sha(OUT/'frame_matches_iou_0_3.csv')!=provenance['selection_sha256']:
        raise ValueError('Nguon phat hien khac provenance cua thi nghiem.')
    if sha(ROOT/'src/gt_baselines/evaluate_pose_corrections.py')!=provenance['script_sha256']:
        raise ValueError('Ma hieu chinh khac voi ma cua lan chay da luu.')
    rows={r['id']:r for r in read_csv(RESULT/'per_video_metrics.csv')}
    videos={r['id']:r for r in read_csv(OUT/'videos_iou_0_3.csv')}
    manifests={r['id']:r for r in read_csv(RESULT/'input_manifest.csv')}
    frames=defaultdict(list)
    for r in read_csv(OUT/'frame_matches_iou_0_3.csv'):frames[r['id']].append(r)
    predictions=defaultdict(dict)
    for r in read_csv(RESULT/'predictions.csv'):predictions[r['id']][(r['method'],int(r['seed']))]=r
    prior_predictions=defaultdict(list)
    for r in read_csv(OUT/'predictions_iou_0_3.csv'):prior_predictions[r['id']].append(r)
    if len(rows)!=253 or set(rows)!=set(videos) or set(rows)!=set(manifests):
        raise ValueError('Danh sach validation khong khop.')
    return rows,videos,manifests,frames,predictions,prior_predictions

def open_case(sid,rows,videos,manifests,frames,prior_predictions):
    source=OUT/'detections'/f'{sid}.npz';source_pose=POSE_SOURCE/f'{sid}.npz'; rec=manifests[sid]
    if sha(source)!=rec['source_selection_npz_sha256'] or sha(source_pose)!=rec['source_pose_npz_sha256']:
        raise ValueError('Hash pose dau vao cua video khong khop thi nghiem.')
    seq,data,statuses=load_video(sid,videos[sid],frames[sid],prior_predictions[sid])
    try:
        with np.load(RESULT/'per_video'/f'{sid}.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in z.files}
        with np.load(source_pose,allow_pickle=False) as z:
            baseline=z['baseline'];confidence=z['baseline_conf']
        calculated,filled=make_variants(baseline,confidence)
        if not np.array_equal(filled,saved['filled']):raise ValueError('Mat na noi suy khong khop.')
        for name,(pose,conf) in calculated.items():
            if not np.array_equal(pose,saved[name],equal_nan=True) or not np.array_equal(conf,saved[name+'_conf'],equal_nan=True):
                raise ValueError(f'Pose {name} khong khop du lieu hieu chinh da luu.')
        metrics=frame_metrics(seq,data,frames[sid]);video=rows[sid]
        if sum(m['common_count'] for m in metrics)!=int(video['paired_joints']):raise ValueError('So khop chung khong khop.')
        for name in ('raw_identity','fixed_swap')+METHODS:
            total=0.;count=0
            for t,m in enumerate(metrics):
                mask=m['common'];count+=int(mask.sum())
                total+=float(np.linalg.norm(saved[name][t]-m['gt'],axis=1)[mask].sum())
            expected=video[name+'_mean_error_px']
            if count and (not expected or not np.isclose(total/count,float(expected),atol=1e-4,rtol=0)):
                raise ValueError(f'Sai so {name} khong khop thi nghiem.')
        return seq,data,statuses,saved,metrics
    except Exception:
        seq.close();raise

def good_pose(points,conf):
    return np.isfinite(points).all(1)&(points>0).all(1)&np.isfinite(conf)&(conf>0)

def frame_details(saved,metrics,t,method,threshold=1.):
    pre=saved['fixed_swap'][t];pc=saved['fixed_swap_conf'][t]
    post=saved[method][t];qc=saved[method+'_conf'][t];m=metrics[t]
    pre_valid=good_pose(pre,pc);post_valid=good_pose(post,qc)
    displacement=np.linalg.norm(post-pre,axis=1)
    moved=pre_valid&post_valid&(displacement>=threshold)
    filled=saved['filled'][t] if method in ('interpolate3','interpolate3_mean5') else np.zeros(12,bool)
    mask=m['common'];n=int(mask.sum())
    before=float(np.linalg.norm(pre-m['gt'],axis=1)[mask].mean()) if n else None
    after=float(np.linalg.norm(post-m['gt'],axis=1)[mask].mean()) if n else None
    gt=m['gt'];visible=m['gt_visible']&np.isfinite(gt).all(1)&(gt>0).all(1)
    individually_valid=visible&post_valid
    single=float(np.linalg.norm(post-gt,axis=1)[individually_valid].mean()) if individually_valid.any() else None
    return {'moved':moved,'filled':filled,'pre':pre,'post':post,'pre_conf':pc,'post_conf':qc,
            'before':before,'after':after,'common_count':n,'after_visible_count':int(individually_valid.sum()),
            'after_single_error':single}

def matches(mode,item,metric):
    if mode=='all':return True
    if mode=='changed':return bool(item['moved'].any() or item['filled'].any())
    if mode=='filled':return bool(item['filled'].any())
    if mode=='no_gt_match':return metric['missing']
    if item['before'] is None:return False
    if mode=='improved':return item['after']<item['before']-.1
    if mode=='worsened':return item['after']>item['before']+.1
    return False

def draw_changes(img,details):
    h,w=img.shape[:2]
    for j in np.flatnonzero(details['moved']):
        a=np.rint(details['pre'][j]).astype(int);b=np.rint(details['post'][j]).astype(int)
        if 0<=a[0]<w and 0<=a[1]<h and 0<=b[0]<w and 0<=b[1]<h:
            cv2.arrowedLine(img,tuple(a),tuple(b),(0,255,255),2,cv2.LINE_AA,tipLength=.3)
    for j in np.flatnonzero(details['filled']):
        p=details['post'][j]
        if np.isfinite(p).all() and 0<p[0]<w and 0<p[1]<h:
            cv2.circle(img,tuple(np.rint(p).astype(int)),7,(0,165,255),2,cv2.LINE_AA)
    return img

def print_predictions(predictions,sid,method):
    print(f'\n{sid} | du doan HANH DONG CAP VIDEO | truoc: swap co dinh | sau: {LABELS[method]}')
    for seed in (0,1,2):
        a=predictions[sid][('fixed_swap',seed)];b=predictions[sid][(method,seed)]
        print(f"  seed {seed}: GT={ACTION_CLASSES[int(a['target'])]} | truoc={ACTION_CLASSES[int(a['predicted'])]} | sau={ACTION_CLASSES[int(b['predicted'])]}")

def print_frame_details(item,metric,frame):
    print(f'\nFrame {frame}: toa do pixel (truoc / sau); GT visible va khop chung dung de do sai so')
    for j,name in enumerate(JOINTS):
        a=np.array2string(item['pre'][j],precision=2,suppress_small=True)
        b=np.array2string(item['post'][j],precision=2,suppress_small=True)
        gt=np.array2string(metric['gt'][j],precision=2,suppress_small=True)
        print(f"  {name:12s} GT={gt} visible={int(metric['gt_visible'][j])} | {a} -> {b} | noi_suy={int(item['filled'][j])} khop_chung={int(metric['common'][j])}")

def render(seq,data,statuses,saved,metrics,details,video,t,method,paused,overlay,arrows,mode):
    img=seq.frame(t,False);h,w=img.shape[:2];m=metrics[t];detail=details[method][t]
    gt_img,raw_img,before_img,after_img=(img.copy() for _ in range(4))
    gt=m['gt'].copy();gt[~m['gt_visible']]=np.nan
    draw_pose(gt_img,gt)
    rings=(m['gt'],m['gt_visible']) if overlay else None
    draw_pose(raw_img,saved['raw_identity'][t],saved['raw_identity_conf'][t],rings)
    draw_pose(before_img,detail['pre'],detail['pre_conf'],rings)
    draw_pose(after_img,detail['post'],detail['post_conf'],rings)
    if arrows:draw_changes(after_img,detail)
    gtbox=data['gt_bbox'][t]
    if np.isfinite(gtbox).all() and gtbox[2]>gtbox[0] and gtbox[3]>gtbox[1]:
        for pane in (gt_img,raw_img,before_img,after_img):draw_box(pane,gtbox,(50,210,50),'GT bbox',1)
    top=144;bottom=48;canvas=np.zeros((h+top+bottom,w*4,3),np.uint8)
    for i,pane in enumerate((gt_img,raw_img,before_img,after_img)):canvas[top:top+h,i*w:(i+1)*w]=pane
    text(canvas,f"{seq.item['id']} | {seq.item['action']} | frame {t+1}/{seq.count} | {'PAUSE' if paused else 'PLAY'}",6,17)
    text(canvas,f"Trang thai: {statuses[t]} | tim: {mode} | GT ho tro chon nguoi | M: doi phuong phap",6,36,(0,215,255),.40)
    text(canvas,f"Di chuyen >=1px: {int(detail['moved'].sum())}/12 | duoc noi suy: {int(detail['filled'].sum())}/12 | khop GT chung: {detail['common_count']}/12",6,54,scale=.38)
    pre=float(video['fixed_swap_mean_error_px']) if video['fixed_swap_mean_error_px'] else None
    post=float(video[method+'_mean_error_px']) if video[method+'_mean_error_px'] else None
    text(canvas,f"Video (khop chung): truoc {pre:.2f}px | sau {post:.2f}px" if pre is not None and post is not None else 'Video: khong du khop chung',6,72,scale=.38)
    for i,title in enumerate(('GT','YOLO GOC','TRUOC: SWAP CO DINH','SAU: '+LABELS[method])):
        text(canvas,title,i*w+6,96,(50,210,50) if i==0 else (0,215,255),.40)
    text(canvas,'Khong ghep GT: pose thieu' if m['missing'] else 'Cung nguoi GT chon',w+6,114,scale=.34)
    text(canvas,f"Khop chung: {detail['before']:.2f}px" if detail['before'] is not None else 'Khop chung: N/A (thieu)',2*w+6,114,scale=.36)
    text(canvas,f"Khop chung: {detail['after']:.2f}px" if detail['after'] is not None else 'Khop chung: N/A (thieu)',3*w+6,114,scale=.36)
    if detail['before'] is None and detail['after_single_error'] is not None:
        text(canvas,f"GT rieng: {detail['after_visible_count']}/12, {detail['after_single_error']:.2f}px",3*w+6,132,scale=.34)
    text(canvas,'Vong rong: GT | Cham: YOLO | Mui ten vang: di chuyen >=1px | Vong cam: noi suy',6,132,scale=.34)
    # Timeline indicates whether the selected method changed a visible point or filled a joint.
    left,right=6,canvas.shape[1]-7;y=top+h+5
    for x in range(left,right):
        frame=min(seq.count-1,int((x-left)*seq.count/max(1,right-left)))
        item=details[method][frame]
        color=(0,165,255) if item['filled'].any() else ((255,255,0) if item['moved'].any() else (90,90,90))
        cv2.line(canvas,(x,y),(x,y+12),color,1)
    mark=left+int((t+.5)*(right-left)/seq.count);cv2.line(canvas,(mark,y-3),(mark,y+15),(255,255,255),2)
    text(canvas,'CAM: CO NOI SUY | XANH: KHOP DOI >=1px | XAM: KHONG CO THAY DOI LON',left,y+31,scale=.34)
    return canvas

def play(sid,rows,videos,manifests,frames,predictions,prior,start,initial,save):
    seq,data,statuses,saved,metrics=open_case(sid,rows,videos,manifests,frames,prior)
    try:
        if not 1<=start<=seq.count:raise ValueError('Frame ngoai pham vi video.')
        index=start-1;method_index=METHODS.index(initial);paused=True;overlay=True;arrows=True;mode_index=0
        details={name:[frame_details(saved,metrics,t,name) for t in range(seq.count)] for name in METHODS}
        modes=('changed','filled','improved','worsened','no_gt_match','all')
        print_predictions(predictions,sid,METHODS[method_index])
        print('Space: phat/dung | A/D: frame | M: phuong phap | E: bo loc | N/P: tim frame | O: GT | U: mui ten | I: toa do | Q: thoat')
        if save:
            picture=render(seq,data,statuses,saved,metrics,details,rows[sid],index,METHODS[method_index],True,overlay,arrows,'all')
            if not cv2.imwrite(str(save),picture):raise OSError('Khong ghi duoc anh: '+str(save))
            print('Da luu',save);return
        window='Xem pose YOLO truoc va sau hieu chinh';cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1800,700)
        try:
            while True:
                method=METHODS[method_index];mode=modes[mode_index]
                cv2.imshow(window,render(seq,data,statuses,saved,metrics,details,rows[sid],index,method,paused,overlay,arrows,mode))
                key=cv2.waitKey(40 if paused else max(1,int(1000/seq.fps)))&0xff
                if key in (ord('q'),27):break
                if key==ord(' '):paused=not paused
                elif key==ord('a'):index,paused=max(0,index-1),True
                elif key==ord('d'):index,paused=min(seq.count-1,index+1),True
                elif key==ord('r'):index,paused=0,True
                elif key==ord('m'):
                    method_index=(method_index+1)%len(METHODS);print_predictions(predictions,sid,METHODS[method_index])
                elif key==ord('o'):overlay=not overlay
                elif key==ord('u'):arrows=not arrows
                elif key==ord('i'):print_frame_details(details[method][index],metrics[index],index+1)
                elif key==ord('e'):mode_index=(mode_index+1)%len(modes);print('Bo loc:',modes[mode_index],flush=True)
                elif key in (ord('n'),ord('p')):
                    step=1 if key==ord('n') else -1
                    for candidate in range(index+step,seq.count if step==1 else -1,step):
                        item=details[method][candidate]
                        if matches(mode,item,metrics[candidate]):index=candidate;break
                    else:print('Khong con frame thoa bo loc theo huong nay.',flush=True)
                    paused=True
                elif not paused:index=(index+1)%seq.count
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
        finally:cv2.destroyWindow(window)
    finally:seq.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',help='ID validation, vi du 0018');p.add_argument('--frame',type=int,default=1)
    p.add_argument('--method',choices=METHODS,default='mean5');p.add_argument('--list',action='store_true');p.add_argument('--save-frame',type=Path,help='Luu anh thay vi mo cua so; can --sequence')
    args=p.parse_args()
    if args.save_frame and not args.sequence:p.error('--save-frame can --sequence')
    rows,videos,manifests,frames,predictions,prior=inputs()
    if args.list:
        for r in rows.values():print(f"{r['id']} {r['action']} | filled={r['filled_joint_frames']} | before={r['fixed_swap_mean_error_px']} | after={r[args.method+'_mean_error_px']}")
        return
    if args.sequence:
        sid=args.sequence.zfill(4)
        if sid not in rows:p.error('Video khong co trong 253 video validation.')
        play(sid,rows,videos,manifests,frames,predictions,prior,args.frame,args.method,args.save_frame);return
    try:
        while True:
            def fmt(value):return f'{float(value):.2f}' if value else 'N/A'
            r=choose('Chon video validation',list(rows.values()),lambda r:f"{r['id']} {r['action']} | khop duoc noi suy={r['filled_joint_frames']} | truoc={fmt(r['fixed_swap_mean_error_px'])}px | {LABELS[args.method]}={fmt(r[args.method+'_mean_error_px'])}px")
            play(r['id'],rows,videos,manifests,frames,predictions,prior,args.frame,args.method,None)
    except (KeyboardInterrupt,EOFError):print('\nDa thoat.')

if __name__=='__main__':main()
