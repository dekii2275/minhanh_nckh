import sys
sys.path.insert(0,'src')
import audit_left_right_step1 as a
import scipy.io as sio, numpy as np, cv2
cases={'0757':[111,176,183,221],'0864':[27,41,43,46],'2104':[16,22,37,61],'0604':[15,28,36,41],'2203':[1,23,29,69],'2062':[1,32,47,63],'1897':[1,13,25,48],'1691':[1,15,29,44]}
for sid,frames in cases.items():
 m=sio.loadmat(a.DATA/'labels'/f'{sid}.mat');z=np.load(a.OUT/f'{sid}.npz');rows=[]
 for frame in frames:
  t=frame-1;gt=np.stack([m['x'][t,1:],m['y'][t,1:]],-1);vis=m['visibility'][t,1:].astype(bool);xy=z['xy'][t];conf=z['conf'][t]
  orig=cv2.imread(str(a.DATA/'frames'/sid/f'{frame:06d}.jpg'))
  p=gt[vis]; lo=np.maximum(0,p.min(0)-35).astype(int);hi=np.minimum([orig.shape[1],orig.shape[0]],p.max(0)+35).astype(int)
  panels=[]
  for title,pts,cf in [('GT',gt,None),('Identity',xy,conf),('Fixed Swap',xy[a.SWAP],conf[a.SWAP])]:
   im=orig.copy()
   if cf is not None:a.draw(im,gt,vis,True)
   a.draw(im,pts,vis if cf is None else (cf>=.5)&np.isfinite(pts).all(1)&(pts>0).all(1),cf is None)
   im=cv2.resize(im[lo[1]:hi[1],lo[0]:hi[0]],(400,320));ban=np.zeros((30,400,3),np.uint8);cv2.putText(ban,f'{sid} f{frame} {title}',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(np.vstack([ban,im]))
  rows.append(np.hstack(panels))
 cv2.imwrite(str(a.OUT/'figures'/f'evidence_{sid}.jpg'),np.vstack(rows))
