"""Render GT-crop cached pose diagnostics for manual selection, not switch counts."""
import csv,json
from pathlib import Path
import numpy as np, cv2
from scipy.io import loadmat
from targeted_pose_pilot_20261113 import ROOT, MAN, SRC, LABEL, FRAMES, SWAP, valid

OUT=ROOT/'outputs/gt_person_switch_search_outside_20261113'
EDGES=[(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
def draw(sid,t):
    with np.load(SRC/f'{sid}.npz') as d: x=d['xy'][:,SWAP]; conf=d['conf'][:,SWAP]; st=d['status']
    m=loadmat(LABEL/f'{sid}.mat'); gt=np.stack([m['x'][:,1:13],m['y'][:,1:13]],-1); vis=m['visibility'][:,1:13]; bbox=m['bbox']
    im=cv2.imread(str(FRAMES/sid/f'{t+1:06d}.jpg'))
    if t<len(bbox):
        b=bbox[t].astype(int); cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(0,230,0),2)
    for a,b in EDGES:
        if vis[t,a] and vis[t,b] and valid(gt[t,a]) and valid(gt[t,b]): cv2.line(im,tuple(gt[t,a].astype(int)),tuple(gt[t,b].astype(int)),(0,210,0),2,cv2.LINE_AA)
        if st[t]==3 and conf[t,a]>.3 and conf[t,b]>.3 and valid(x[t,a]) and valid(x[t,b]): cv2.line(im,tuple(x[t,a].astype(int)),tuple(x[t,b].astype(int)),(0,0,255),2,cv2.LINE_AA)
    for j in range(12):
        if st[t]==3 and conf[t,j]>.3 and valid(x[t,j]): cv2.circle(im,tuple(x[t,j].astype(int)),3,(0,0,255),-1)
    cv2.rectangle(im,(0,0),(im.shape[1],26),(0,0,0),-1)
    cv2.putText(im,f'{sid} frame {t+1} | GT green  YOLO red | status {st[t]}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return im
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[r for r in csv.DictReader(MAN.open(encoding='utf-8-sig')) if r['seed0_split']=='validation']; candidates=[]
    for r in rows:
        sid=r['id']; m=loadmat(LABEL/f'{sid}.mat'); gt=np.stack([m['x'][:,1:13],m['y'][:,1:13]],-1); vis=m['visibility'][:,1:13]>0
        with np.load(SRC/f'{sid}.npz') as d: x=d['xy'][:,SWAP]; st=d['status']
        bb=np.full((len(x),4),np.nan); bb[:min(len(x),len(m['bbox']))]=m['bbox'][:len(x)]; wh=bb[:,2:]-bb[:,:2]; diag=np.linalg.norm(wh,axis=1); ok=vis&valid(gt)&valid(x)&(st[:,None]==3)&(diag[:,None]>0)
        err=np.linalg.norm(x-gt,axis=2)/diag[:,None]; n=ok.sum(1); score=np.sum(np.where(ok,err,0),axis=1)/np.maximum(1,n); score[n<5]=np.nan
        diff=np.abs(np.diff(score)); diff[~np.isfinite(diff)]=-1
        outside=((x[:,:,0]<bb[:,None,0])|(x[:,:,0]>bb[:,None,2])|(x[:,:,1]<bb[:,None,1])|(x[:,:,1]>bb[:,None,3])) & ok
        counts=outside.sum(axis=1); counts[~np.isfinite(score)]=0
        if counts.max()>=3:
            worst=int(np.argmax(counts)); candidates.append((float(counts[worst]+score[worst]),sid,worst,r['action']))
    candidates.sort(reverse=True); candidates=candidates[:60]
    tiles=[]
    for _,sid,t,_ in candidates:
        im=draw(sid,t); tiles.append(cv2.resize(im,(384,240)))
    for start in range(0,len(tiles),16):
        group=tiles[start:start+16]
        if len(group)%4: group += [np.zeros_like(group[0])]*(4-len(group)%4)
        sheet=np.vstack([np.hstack(group[i:i+4]) for i in range(0,len(group),4)]); cv2.imwrite(str(OUT/f'candidates_{start//16}.jpg'),sheet)
    (OUT/'candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    print(json.dumps(candidates[:24],indent=2))
if __name__=='__main__':main()
