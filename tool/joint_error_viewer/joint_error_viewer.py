"""Rank and review GT-crop YOLO joint errors on Penn validation (no inference)."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tool/dataset_viewer'))
from dataset_viewer import Sequence, choose
sys.path.insert(0, str(ROOT / 'src/gt_baselines'))
from eda_joint_confidence import MANIFEST, SOURCE, LABELS, NAMES, SWAP, sha

FRAMES = ROOT / 'data/raw/Penn_Action/frames'
DEFAULT_NOTES = ROOT / 'outputs/joint_error_review/annotations_v2.csv'
GROUPS = ('low_wrong', 'high_wrong', 'low_correct', 'missing', 'all')
JOINTS = ('left_wrist', 'right_wrist', 'left_ankle', 'right_ankle')
EDGES = [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
TAGS = {'1':'occlusion', '2':'sudden_jump', '3':'missing_or_gap',
        '4':'possible_swap', '5':'possible_wrong_person', '6':'uncertain', '7':'other'}
TAG_LABELS = ('1. Che khuat', '2. Nhay dot ngot', '3. Thieu khop / pose',
              '4. Nghi dao trai-phai', '5. Nghi sai nguoi', '6. Chua ro', '7. Loi khac')
PANEL_WIDTH = 235
STATUS = ('invalid_gt_bbox', 'no_person_detection', 'no_gt_iou_match', 'matched')
FIELDS = ['timestamp_utc','id','action','frame','joint','group','tag','note','confidence',
          'error_px','error_bbox_diagonal','frame_status','gt_eligible',
          'low_threshold','high_threshold','pck_threshold','mapping','cache_sha256','annotation_sha256']


def load_samples():
    provenance = json.loads((SOURCE / 'provenance.json').read_text(encoding='utf-8'))
    if sha(MANIFEST) != provenance['manifest_sha256']:
        raise ValueError('Manifest khong khop provenance cache YOLO.')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['seed0_split'] == 'validation']
    if len(rows) != 253:
        raise ValueError('Manifest khong co 253 video validation.')
    return rows


def load_case(row):
    sid = row['id']; label = LABELS / f'{sid}.mat'; path = SOURCE / 'poses' / f'{sid}.npz'
    if sha(label) != row['label_sha256']:
        raise ValueError(f'Hash annotation khong khop: {sid}')
    with np.load(path) as d:
        if str(d['annotation_sha256']) != row['label_sha256']:
            raise ValueError(f'Hash annotation trong cache khong khop: {sid}')
        xy, conf, status = d['xy'][:, SWAP].copy(), d['conf'][:, SWAP].copy(), d['status'].copy()
        frame_hash = str(d['image_bytes_sha256'])
    item = {'id':sid, 'action':row['action'], 'label':label, 'source':FRAMES/sid}
    seq = Sequence('penn', item)
    if seq.count != int(row['nframes']) or len(status) != seq.count or xy.shape != (seq.count,12,2) or conf.shape != (seq.count,12):
        seq.close(); raise ValueError(f'So frame/shape khong khop: {sid}')
    # Read annotation separately for GT bounding boxes; bbox can be shorter than frames.
    from scipy.io import loadmat
    mat = loadmat(label, variable_names=['bbox'])
    bbox = np.full((seq.count,4), np.nan)
    n = min(seq.count, len(mat['bbox'])); bbox[:n] = mat['bbox'][:n]
    wh = bbox[:,2:] - bbox[:,:2]; diag = np.linalg.norm(wh, axis=1)
    box_ok = np.isfinite(bbox).all(1) & (wh > 0).all(1)
    gt = seq.points[:,1:13].astype(float)
    eligible = seq.visible[:,1:13] & np.isfinite(gt).all(2) & (gt > 0).all(2) & box_ok[:,None]
    observed = (status[:,None] == 3) & np.isfinite(xy).all(2) & (xy > 0).all(2) & np.isfinite(conf)
    error = np.linalg.norm(xy - gt,axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = error / diag[:,None]
    return dict(row=row, seq=seq, xy=xy, conf=conf, status=status, gt=gt, bbox=bbox,
                eligible=eligible, observed=observed, error=error, norm=norm,
                cache_path=path, frame_hash=frame_hash)


def case_mask(data, j, group, low, high, pck):
    eligible = data['eligible'][:,j]; observed = data['observed'][:,j]
    c = data['conf'][:,j]; wrong = data['norm'][:,j] > pck
    if group == 'missing': return eligible & ~observed
    if group == 'all': return np.ones(len(eligible), dtype=bool)
    usable = eligible & observed
    if group == 'low_wrong': return usable & (c < low) & wrong
    if group == 'high_wrong': return usable & (c >= high) & wrong
    if group == 'low_correct': return usable & (c < low) & ~wrong
    raise ValueError(group)


def rank(rows, j, group, args):
    results = []
    for row in rows:
        data = load_case(row)
        try:
            mask = case_mask(data,j,group,args.low,args.high,args.pck)
            eligible = int(data['eligible'][:,j].sum())
            if mask.any():
                results.append((row, int(mask.sum()), eligible, int(np.flatnonzero(mask)[0])+1))
        finally: data['seq'].close()
    return sorted(results,key=lambda x:(-x[1],x[0]['id']))


def label(img, text, x, y, color=(235,235,235), scale=.43):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def draw_pose(img, xy, valid, selected):
    h,w = img.shape[:2]
    valid = valid & np.isfinite(xy).all(1) & (xy > 0).all(1) & (xy[:,0] < w) & (xy[:,1] < h)
    for a,b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(img,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),(100,210,110),2,cv2.LINE_AA)
    for k in np.flatnonzero(valid):
        p=tuple(np.rint(xy[k]).astype(int)); color=(0,180,255) if k == selected else (0,255,255)
        cv2.circle(img,p,6 if k == selected else 3,color,-1,cv2.LINE_AA)


def button_rects(frame_width):
    x = frame_width * 2 + 10
    return [(x, 50 + i * 39, x + PANEL_WIDTH - 20, 82 + i * 39) for i in range(len(TAG_LABELS))]


def button_at(x, y, frame_width):
    return next((str(i + 1) for i, (x1, y1, x2, y2) in enumerate(button_rects(frame_width))
                 if x1 <= x <= x2 and y1 <= y <= y2), None)


def render(data, t, j, group, args, paused, feedback=''):
    seq=data['seq']; raw=seq.frame(t,False); h,w=raw.shape[:2]
    left=raw.copy();right=raw.copy()
    draw_pose(left,data['gt'][t],seq.visible[t,1:13],j)
    draw_pose(right,data['xy'][t],data['observed'][t],j)
    if data['eligible'][t,j]:
        p=tuple(np.rint(data['gt'][t,j]).astype(int))
        if 0 <= p[0] < w and 0 <= p[1] < h: cv2.circle(right,p,8,(255,200,20),2,cv2.LINE_AA)
    header=100; canvas=np.zeros((max(h+header, 415),w*2+PANEL_WIDTH,3),dtype=np.uint8)
    canvas[header:header+h,:w]=left;canvas[header:header+h,w:2*w]=right
    label(canvas,f"{seq.item['id']} {seq.item['action']} | frame {t+1}/{seq.count} | {'PAUSE' if paused else 'PLAY'}",5,16)
    label(canvas,f"Joint: {NAMES[j]} | filter: {group}",5,34,(0,200,255))
    valid=data['eligible'][t,j] and data['observed'][t,j]
    c=f"{data['conf'][t,j]:.3f}" if data['status'][t] == 3 else 'N/A'
    error=f"{data['error'][t,j]:.1f}px / {data['norm'][t,j]:.3f} bbox diag" if valid else 'N/A'
    label(canvas,f"conf {c} | error {error}",5,52)
    label(canvas,f"status {STATUS[int(data['status'][t])]} | GT eligible {int(data['eligible'][t,j])} | YOLO xy {int(data['observed'][t,j])}",5,70)
    label(canvas,'GT | orange = selected joint',5,92,(60,220,60))
    label(canvas,'YOLO swap | ring = GT',w+5,92,(0,200,255))
    x=w*2
    cv2.rectangle(canvas,(x,0),(canvas.shape[1]-1,canvas.shape[0]-1),(35,35,35),-1)
    label(canvas,'CHON LOI BANG CHUOT',x+10,22,(0,210,255),.44)
    label(canvas,'Bam nut de ghi nhan',x+10,39,scale=.38)
    for rect, name in zip(button_rects(w),TAG_LABELS):
        x1,y1,x2,y2=rect
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(75,75,75),-1)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(170,170,170),1)
        label(canvas,name,x1+6,y1+21,scale=.41)
    label(canvas,'A/D: frame | N/P: loc',x+10,350,scale=.38)
    label(canvas,'Space: phat/dung',x+10,368,scale=.38)
    if feedback:label(canvas,feedback,x+10,394,(80,230,120),.36)
    return canvas


def verify_frames(data):
    digest=hashlib.sha256()
    for frame in data['seq'].frames:
        raw=frame.read_bytes();digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    if digest.hexdigest() != data['frame_hash']:
        raise ValueError('Anh hien tai khong khop hash anh da dung tao cache pose.')


def annotate(data,t,j,group,tag,args,note=''):
    path=args.notes;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding='utf-8-sig',newline='') as f:
            if next(csv.reader(f),[]) != FIELDS: raise ValueError('File ghi chu co header khong dung; chon --notes khac.')
    valid=data['eligible'][t,j] and data['observed'][t,j]
    row={'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'id':data['row']['id'],
        'action':data['row']['action'],'frame':t+1,'joint':NAMES[j],'group':group,'tag':tag,'note':note,
        'confidence':f"{data['conf'][t,j]:.6f}" if data['status'][t]==3 else '',
        'error_px':f"{data['error'][t,j]:.6f}" if valid else '',
        'error_bbox_diagonal':f"{data['norm'][t,j]:.6f}" if valid else '',
        'frame_status':STATUS[int(data['status'][t])],'gt_eligible':int(data['eligible'][t,j]),
        'low_threshold':args.low,'high_threshold':args.high,'pck_threshold':args.pck,
        'mapping':'fixed_swap','cache_sha256':sha(data['cache_path']),'annotation_sha256':data['row']['label_sha256']}
    new=not path.exists() or path.stat().st_size==0
    with path.open('a',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if new:writer.writeheader()
        writer.writerow(row)
    print(f'Da ghi {tag}: {path}',flush=True)


def ask_note(key, frame):
    """Use Windows Forms without Tkinter; Cancel does not save a label."""
    env=os.environ.copy()
    env['JOINT_REVIEW_LABEL']=TAG_LABELS[int(key)-1]
    env['JOINT_REVIEW_FRAME']=str(frame)
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-STA',
        '-ExecutionPolicy','Bypass','-File',str(Path(__file__).with_name('note_dialog.ps1'))],
        env=env,capture_output=True, text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise OSError(f'Khong mo duoc hop ghi chu: {result.stderr.strip()}')
    output=result.stdout.strip()
    if output == 'CANCEL':return None
    if output.startswith('SAVE:'):
        try:return base64.b64decode(output[5:],validate=True).decode('utf-8')
        except (ValueError, UnicodeError) as exc:raise ValueError('Du lieu ghi chu khong hop le') from exc
    raise ValueError('Hop ghi chu khong tra ve ket qua hop le.')


def play(row,j,group,args,start=None):
    data=load_case(row);seq=data['seq']
    try:
        verify_frames(data)
        mask=case_mask(data,j,group,args.low,args.high,args.pck)
        if start is None:
            if not mask.any(): raise ValueError('Video khong co frame trong nhom da chon.')
            start=int(np.flatnonzero(mask)[0])+1
        if not 1 <= start <= seq.count: raise ValueError('Frame nam ngoai video.')
        t=start-1; paused=True; feedback=''
        if args.save_frame:
            if args.save_frame.exists(): raise FileExistsError('Anh da ton tai; chon duong dan khac.')
            if not cv2.imwrite(str(args.save_frame),render(data,t,j,group,args,paused)):raise OSError('Khong luu duoc anh.')
            print('Da luu',args.save_frame);return
        print('Space: phat/dung | A/D: lui/tien | N/P: frame cung bo loc | Q: thoat')
        print('Ghi nhan quan sat: 1 che khuat, 2 nhay dot ngot, 3 thieu, 4 nghi swap, 5 nghi sai nguoi, 6 chua ro, 7 khac')
        window='Phan tich loi khop - GT vs YOLO';cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1400,700)
        selected=[]
        frame_width=seq.frame(t,False).shape[1]
        def on_mouse(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                chosen=button_at(x,y,frame_width)
                if chosen:selected.append(chosen)
        cv2.setMouseCallback(window,on_mouse)
        try:
            while True:
                cv2.imshow(window,render(data,t,j,group,args,paused,feedback))
                key=cv2.waitKey(40 if paused else max(1,int(1000/seq.fps))) & 0xff
                if key in (ord('q'),27):break
                chosen = selected.pop(0) if selected else (chr(key) if chr(key) in TAGS else None)
                if chosen:
                    paused=True
                    try:
                        note=ask_note(chosen,t+1)
                        if note is not None:
                            annotate(data,t,j,group,TAGS[chosen],args,note)
                            feedback=f"Da luu: {TAGS[chosen]}"
                        else:feedback='Da huy, chua luu'
                    except (OSError, ValueError) as exc:
                        feedback='LOI LUU - xem terminal'
                        print(f'Khong luu duoc ghi chu: {exc}',flush=True)
                    continue
                if key==ord(' '):paused=not paused
                elif key in (ord('a'),ord('d')):t=max(0,min(seq.count-1,t+(-1 if key==ord('a') else 1)));paused=True
                elif key in (ord('n'),ord('p')):
                    step=1 if key==ord('n') else -1
                    matches=[i for i in range(t+step,seq.count if step==1 else -1,step) if mask[i]]
                    if matches:t=matches[0]
                    else:print('Khong con frame thoa bo loc theo huong nay.',flush=True)
                    paused=True
                elif not paused:t=(t+1)%seq.count
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
        finally:cv2.destroyWindow(window)
    finally:seq.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint',choices=JOINTS,default='left_wrist')
    p.add_argument('--group',choices=GROUPS,default='low_wrong')
    p.add_argument('--sequence',help='ID video validation')
    p.add_argument('--frame',type=int,help='Frame bat dau, danh so tu 1')
    p.add_argument('--low',type=float,default=.3);p.add_argument('--high',type=float,default=.7)
    p.add_argument('--pck',type=float,default=.2,help='Nguong sai so theo duong cheo bbox GT')
    p.add_argument('--list',action='store_true',help='In xep hang video')
    p.add_argument('--export',type=Path,help='Ghi xep hang CSV moi, khong ghi de')
    p.add_argument('--notes',type=Path,default=DEFAULT_NOTES)
    p.add_argument('--save-frame',type=Path,help='Luu frame de kiem tra; can --sequence')
    args=p.parse_args()
    if not 0 <= args.low <= args.high <= 1 or args.pck <= 0:p.error('Nguong khong hop le.')
    if args.save_frame and not args.sequence:p.error('--save-frame can --sequence.')
    rows=load_samples();j=NAMES.index(args.joint)
    if args.sequence:
        matches=[r for r in rows if r['id']==args.sequence.zfill(4)]
        if not matches:p.error('ID khong co trong validation.')
        play(matches[0],j,args.group,args,args.frame);return
    if args.frame:p.error('--frame can --sequence.')
    if not args.list and not args.export:
        try:
            args.joint=choose('Chon khop',list(JOINTS));j=NAMES.index(args.joint)
            args.group=choose('Chon nhom',list(GROUPS))
        except (KeyboardInterrupt,EOFError):return
    print('Dang xep hang video validation...',flush=True)
    ranked=rank(rows,j,args.group,args)
    if args.export:
        with args.export.open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','action','joint','group','matching_frames','gt_eligible_frames','first_frame','low','high','pck'])
            for row,count,eligible,first in ranked:writer.writerow([row['id'],row['action'],args.joint,args.group,count,eligible,first,args.low,args.high,args.pck])
        print('Da ghi',args.export)
    if args.list:
        for row,count,eligible,first in ranked:print(f"{row['id']} {row['action']:16s} cases={count:3d} gt_eligible={eligible:3d} first={first}")
    if args.export or args.list:return
    if not ranked:print('Khong tim thay truong hop.');return
    try:
        while True:
            item=choose('Chon video',ranked,lambda x:f"{x[0]['id']} {x[0]['action']} | {x[1]} frame | GT hop le {x[2]} | dau tien {x[3]}")
            play(item[0],j,args.group,args)
    except (KeyboardInterrupt,EOFError):print('\nDa thoat.')


if __name__=='__main__':main()
