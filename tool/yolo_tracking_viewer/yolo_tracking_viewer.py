"""Interactively inspect YOLO pose detections and multi-person tracks."""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/ultralytics_config'))
from assignment_fallback import configure
from motion_selector import movement_scores, choose_most_moving
ASSIGNMENT=configure()
import ultralytics
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace

sys.path.insert(0,str(ROOT/'tool/dataset_viewer'))
from dataset_viewer import Sequence, catalog, choose
sys.path.insert(0,str(ROOT/'tool/yolo_pose_viewer'))
from yolo_pose_viewer import VideoSource

EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
       (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
PALETTE=[(255,200,35),(60,200,255),(200,90,255),(100,230,80),
         (255,110,110),(60,160,255),(220,100,170),(170,230,80)]

def make_tracker(name):
    config=Path(ultralytics.__file__).resolve().parent/'cfg/trackers'/f'{name}.yaml'
    if not config.is_file():raise FileNotFoundError(f'Khong tim thay config tracker {config}')
    args=IterableSimpleNamespace(**YAML.load(config))
    if name=='botsort' and args.with_reid:
        raise ValueError('BoT-SORT viewer yeu cau with_reid=False; khong tu dong tai model.')
    return (BYTETracker if name=='bytetrack' else BOTSORT)(args=args)

class PennSource:
    def __init__(self,item):
        self.seq=Sequence('penn',item);self.count=self.seq.count;self.fps=self.seq.fps;self.label=item['action']
        self.bbox=sio.loadmat(item['label'],variable_names=['bbox']).get('bbox')
    def read(self,index):return self.seq.frame(index,False)
    def ground_truth(self,index):
        img=self.seq.frame(index,True)
        if self.bbox is not None and index<len(self.bbox):
            b=np.asarray(self.bbox[index],dtype=float)
            if b.shape==(4,) and np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]:
                cv2.rectangle(img,tuple(np.rint(b[:2]).astype(int)),tuple(np.rint(b[2:]).astype(int)),(40,220,40),2)
        return img
    def close(self):self.seq.close()

class ExternalSource:
    def __init__(self,path):
        self.video=VideoSource(path);self.count=self.video.count;self.fps=self.video.fps;self.label=None
    def read(self,index):return self.video.read(index)
    def ground_truth(self,index):return self.read(index)
    def close(self):self.video.close()

def track_frame(model,tracker,image,confidence,device):
    """Keep all detections; tracker output maps assigned IDs to detection indices."""
    result=model.predict(source=image,imgsz=640,conf=confidence,device=device,verbose=False)[0]
    if result.boxes is None:raise ValueError('YOLO khong tra ve boxes.')
    boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
    n=len(boxes)
    if n and (result.keypoints is None or result.keypoints.conf is None):
        raise ValueError('Model phai tra ve YOLO pose keypoints va confidence.')
    xy=result.keypoints.xy.cpu().numpy() if n else np.empty((0,17,2),np.float32)
    conf=result.keypoints.conf.cpu().numpy() if n else np.empty((0,17),np.float32)
    if xy.shape!=(n,17,2) or conf.shape!=(n,17):raise ValueError('Model can dung 17 khop COCO.')
    tracks=tracker.update(result.boxes.cpu().numpy(),image)
    ids=np.full(n,-1,dtype=np.int64)
    for row in tracks:
        idx=int(row[-1]);track_id=int(row[4])
        if idx<0 or idx>=n or ids[idx]>=0:raise ValueError('Tracker tra index detection khong hop le.')
        ids[idx]=track_id
    return {'boxes':boxes.copy(),'scores':scores.copy(),'xy':xy.copy(),'conf':conf.copy(),'ids':ids}

class CachedPlayback:
    """Tracker is updated once in chronological order; backward seeks use cache."""
    def __init__(self,source,model,tracker,confidence,device):
        self.source=source;self.model=model;self.tracker=tracker;self.confidence=confidence;self.device=device
        self.cache=[]
    def get(self,index):
        if not 0<=index<self.source.count:raise IndexError('Frame ngoai pham vi.')
        while len(self.cache)<=index:
            t=len(self.cache);image=self.source.read(t)
            record=track_frame(self.model,self.tracker,image,self.confidence,self.device)
            self.cache.append(record)
            if t and t%50==0 and t<index:print(f'Dang theo doi tu dau: {t+1}/{index+1}',flush=True)
        return self.cache[index]

def color_for(track_id):return (155,155,155) if track_id<0 else PALETTE[track_id%len(PALETTE)]

def line_text(image,value,x,y,color=(255,255,255),scale=.45):
    cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def draw_history(image,cache,index,length):
    histories={}
    for t in range(max(0,index-length+1),index+1):
        item=cache[t]
        for box,tid in zip(item['boxes'],item['ids']):
            if tid<0:continue
            point=tuple(np.rint((box[:2]+box[2:])/2).astype(int))
            if tid in histories and histories[tid][0]==t-1:
                cv2.line(image,histories[tid][1],point,color_for(int(tid)),2,cv2.LINE_AA)
            histories[tid]=(t,point)

def draw_tracks(image,item,selected,pose_threshold,show_pose,motion=None):
    h,w=image.shape[:2]
    order=sorted(range(len(item['ids'])),key=lambda i:item['ids'][i]==selected if selected is not None else False)
    for i in order:
        box=item['boxes'][i];score=item['scores'][i];tid=int(item['ids'][i]);chosen=selected is not None and tid==selected
        color=(0,255,255) if chosen else color_for(tid);width=3 if chosen else 2
        x1,y1,x2,y2=np.rint(box).astype(int);cv2.rectangle(image,(x1,y1),(x2,y2),color,width)
        label=('ID ?' if tid<0 else f'ID {tid}')+f' {score:.2f}'
        if motion is not None and tid>=0:
            stat=motion.get(tid,{});value=stat.get('score')
            label+=f" M={value:.3f}" if value is not None else ' M=--'
        label+=' [CHON]' if chosen else ''
        line_text(image,label,max(1,min(w-1,x1)),max(13,min(h-3,y1-4)),color,.45)
        if not show_pose:continue
        xy,conf=item['xy'][i],item['conf'][i]
        good=np.isfinite(xy).all(axis=1)&np.isfinite(conf)&(conf>=pose_threshold)
        good&=(xy[:,0]>0)&(xy[:,0]<w)&(xy[:,1]>0)&(xy[:,1]<h)
        for a,b in EDGES:
            if good[a] and good[b]:cv2.line(image,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,width,cv2.LINE_AA)
        for point in xy[good]:cv2.circle(image,tuple(np.rint(point).astype(int)),3,color,-1,cv2.LINE_AA)
    return image

def target_at(items,x,y):
    """Pick smallest tracked box containing click; untracked detections are ignored."""
    candidates=[]
    for box,tid in zip(items['boxes'],items['ids']):
        if tid>=0 and box[0]<=x<=box[2] and box[1]<=y<=box[3]:
            candidates.append(((box[2]-box[0])*(box[3]-box[1]),int(tid)))
    return min(candidates)[1] if candidates else None

def render(source,playback,index,tracker_name,selected,show_gt,show_pose,show_history,threshold,paused,
           motion=None,auto_motion=False,min_pairs=5):
    item=playback.get(index);raw=source.read(index);left=source.ground_truth(index) if show_gt else raw.copy();right=raw.copy()
    if show_history:draw_history(right,playback.cache,index,40)
    draw_tracks(right,item,selected,threshold,show_pose,motion)
    h,w=raw.shape[:2];top=96;bottom=40;canvas=np.zeros((top+h+bottom,w*2,3),np.uint8)
    canvas[top:top+h,:w]=left;canvas[top:top+h,w:]=right
    assigned=[int(x) for x in item['ids'] if x>=0];untracked=int((item['ids']<0).sum())
    line_text(canvas,f"{source.label or 'VIDEO'} | frame {index+1}/{source.count} | {'PAUSE' if paused else 'PLAY'}",6,18)
    line_text(canvas,f"YOLO + {tracker_name.upper()} | assignment: {ASSIGNMENT} | detection={len(item['ids'])} | ID={len(assigned)} | chua ID={untracked}",6,38,scale=.40)
    line_text(canvas,'GT annotation (chi de xem)' if source.label and show_gt else 'Anh goc / GT khong hien',6,70,(50,220,50),.43)
    line_text(canvas,'YOLO pose + tracking: bam vao box de chon ID',w+6,70,(0,215,255),.40)
    selected_visible=selected is not None and selected in assigned
    if auto_motion:
        if selected is None:
            message=f'AUTO MOVE: CHUA DU DU LIEU (can {min_pairs} cap frame)'
        else:
            stat=motion[selected]
            message=f"AUTO MOVE: ID {selected} | M={stat['score']:.3f} | {stat['pairs']} cap | chi la gia thuyet"
    else:
        message=('Chua chon ID (bam vao box ben phai)' if selected is None else
                 (f'ID {selected} dang co mat' if selected_visible else f'ID {selected} KHONG CO MAT - khong tu chon nguoi khac'))
    line_text(canvas,message,6,top+h+18,(0,255,255) if selected_visible else (255,255,255),.40)
    line_text(canvas,'Space phat/dung | A/D frame | M auto | I diem | G GT | H vet | C bo chon | Q thoat',6,top+h+35,scale=.35)
    return canvas

def play(source,model,tracker_name,conf,device,kpt_conf,start,save,
         auto_motion=False,motion_window=20,motion_min_pairs=5,motion_conf=.3):
    tracker=make_tracker(tracker_name);playback=CachedPlayback(source,model,tracker,conf,device)
    try:
        if not 1<=start<=source.count:raise ValueError('Frame bat dau ngoai video.')
        print(f'{tracker_name.upper()} | assignment={ASSIGNMENT} | {source.count} frames | conf={conf} | GT chi hien thi',flush=True)
        start_time=time.perf_counter();playback.get(start-1)
        if start>1:print(f'Da xu ly {start} frame lien tiep trong {time.perf_counter()-start_time:.1f}s.',flush=True)
        print('Space: phat/dung | A/D: lui/tien | M: tu chon theo chuyen dong | I: diem | bam box: chon tay | C: bo chon | G: GT | P: pose | H: vet | R: dau | Q: thoat')
        index=start-1;selected=None;show_gt=True;show_pose=True;show_history=True;paused=True
        if save:
            motion=movement_scores(playback.cache,index,motion_window,motion_conf,motion_min_pairs) if auto_motion else None
            selected=choose_most_moving(motion) if auto_motion else None
            picture=render(source,playback,index,tracker_name,selected,True,True,True,kpt_conf,True,
                           motion,auto_motion,motion_min_pairs)
            if not cv2.imwrite(str(save),picture):raise OSError('Khong ghi duoc anh: '+str(save))
            print('Da luu',save);return
        window='YOLO theo doi nhieu nguoi';cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1450,800)
        state={'index':index,'selected':None,'auto':auto_motion,'width':source.read(index).shape[1],'top':96}
        def mouse(event,x,y,flags,param):
            if event!=cv2.EVENT_LBUTTONDOWN:return
            local_x=x-state['width'];local_y=y-state['top']
            if local_x<0 or local_y<0:return
            chosen=target_at(playback.cache[state['index']],local_x,local_y)
            if chosen is not None:
                state['selected']=chosen;state['auto']=False
                print(f'Da chon tay track ID {chosen}; theo doi nay co the doi danh tinh.',flush=True)
        cv2.setMouseCallback(window,mouse)
        try:
            while True:
                state['index']=index;playback.get(index)
                motion=movement_scores(playback.cache,index,motion_window,motion_conf,motion_min_pairs) if state['auto'] else None
                selected=choose_most_moving(motion) if state['auto'] else state['selected']
                cv2.imshow(window,render(source,playback,index,tracker_name,selected,show_gt,show_pose,show_history,kpt_conf,paused,
                                         motion,state['auto'],motion_min_pairs))
                key=cv2.waitKey(40 if paused else max(1,int(1000/source.fps)))&0xff
                if key in (ord('q'),27):break
                if key==ord(' '):paused=not paused
                elif key==ord('a'):index,paused=max(0,index-1),True
                elif key==ord('d'):index,paused=min(source.count-1,index+1),True
                elif key==ord('r'):index,paused=0,True
                elif key==ord('c'):state['selected']=None;state['auto']=False
                elif key==ord('m'):
                    state['auto']=not state['auto']
                    print('Tu chon theo chuyen dong:', 'BAT' if state['auto'] else 'TAT',flush=True)
                elif key==ord('i'):
                    values=movement_scores(playback.cache,index,motion_window,motion_conf,motion_min_pairs)
                    print(f'Frame {index+1}: diem chuyen dong YOLO, khong dung GT:',flush=True)
                    for tid,stat in sorted(values.items()):
                        score='chua du' if stat['score'] is None else f"{stat['score']:.4f}"
                        print(f"  ID {tid}: M={score}, {stat['pairs']} cap frame",flush=True)
                elif key==ord('g'):show_gt=not show_gt
                elif key==ord('p'):show_pose=not show_pose
                elif key==ord('h'):show_history=not show_history
                elif not paused:
                    if index==source.count-1:paused=True
                    else:index+=1
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
        finally:cv2.destroyWindow(window)
    finally:source.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',help='ID Penn Action, vi du 0018')
    p.add_argument('--video',type=Path,help='Video hoac thu muc frame rieng, khong co GT')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/raw')
    p.add_argument('--action',help='Loc lop Penn Action trong menu')
    p.add_argument('--model',type=Path,default=ROOT/'models/yolo11n-pose.pt')
    p.add_argument('--tracker',choices=('bytetrack','botsort'),default='bytetrack')
    p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--conf',type=float,default=.10,help='Nguong detection, mac dinh 0.10 de tracker co the nhan detection yeu')
    p.add_argument('--kpt-conf',type=float,default=.5,help='Nguong VE khop, khong anh huong tracker')
    p.add_argument('--frame',type=int,default=1,help='Frame bat dau; tracker xu ly tu frame 1')
    p.add_argument('--save-frame',type=Path,help='Luu anh frame thay vi mo cua so')
    p.add_argument('--auto-motion',action='store_true',help='Tu chon ID co chuyen dong lon nhat')
    p.add_argument('--motion-window',type=int,default=20,help='So cap frame gan day de tinh chuyen dong')
    p.add_argument('--motion-min-pairs',type=int,default=5,help='So cap frame hop le toi thieu de chon ID')
    p.add_argument('--motion-conf',type=float,default=.3,help='Nguong confidence khop khi tinh chuyen dong')
    args=p.parse_args()
    if args.video and (args.sequence or args.action):p.error('--video khong dung cung --sequence/--action')
    if not (0<args.conf<=1 and 0<=args.kpt_conf<=1):p.error('Nguong khong hop le')
    if args.motion_window<1 or not 1<=args.motion_min_pairs<=args.motion_window or not 0<=args.motion_conf<=1:
        p.error('Cau hinh motion khong hop le')
    if not args.model.is_file():p.error('Khong tim thay model '+str(args.model))
    if args.device!='cpu' and not torch.cuda.is_available():p.error('CUDA khong kha dung; dung --device cpu')
    if args.video and not args.video.exists():p.error('Khong tim thay video hoac thu muc frame')
    if args.save_frame and not (args.video or args.sequence):p.error('--save-frame can --video hoac --sequence')
    if args.conf>=.25:print('Luu y: conf >= 0.25 loai detection duoi high threshold; khong con nhieu dau vao yeu cho ByteTrack.',flush=True)
    model=YOLO(str(args.model),task='pose')
    options={'auto_motion':args.auto_motion,'motion_window':args.motion_window,'motion_min_pairs':args.motion_min_pairs,'motion_conf':args.motion_conf}
    if args.video:play(ExternalSource(args.video),model,args.tracker,args.conf,args.device,args.kpt_conf,args.frame,args.save_frame,**options);return
    if args.sequence:
        sid=args.sequence.zfill(4);root=args.data_root/'Penn_Action';label=root/'labels'/f'{sid}.mat';source=root/'frames'/sid
        if not label.is_file() or not source.is_dir():p.error('Khong tim thay video Penn '+sid)
        action=str(sio.loadmat(label,variable_names=['action'])['action'].flat[0])
        item={'id':sid,'action':action,'label':label,'source':source}
        if args.action and args.action!=action:p.error('Video khong thuoc lop '+args.action)
        play(PennSource(item),model,args.tracker,args.conf,args.device,args.kpt_conf,args.frame,args.save_frame,**options);return
    try:
        entries=catalog(args.data_root)['penn']
        if not entries:print('Khong tim thay Penn Action.');return
        while True:
            action=args.action or choose('Chon hanh dong',sorted({v['action'] for v in entries}))
            candidates=[v for v in entries if v['action']==action]
            if not candidates:p.error('Khong tim thay hanh dong '+action)
            item=choose('Chon video',candidates,lambda v:v['id'])
            play(PennSource(item),model,args.tracker,args.conf,args.device,args.kpt_conf,args.frame,None,**options)
    except (KeyboardInterrupt,EOFError):print('\nDa thoat.')

if __name__=='__main__':main()
