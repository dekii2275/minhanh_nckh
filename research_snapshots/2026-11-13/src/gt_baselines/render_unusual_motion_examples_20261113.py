"""Render candidate clips from cached validation detections and the existing event CSV."""
import csv, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from targeted_pose_pilot_20261113 import ROOT, SRC, SWAP, NAMES, valid, sha
from render_gt_person_switch_examples import draw

OUT=ROOT/'outputs/unusual_joint_motion_examples_20261113'
EVENTS=ROOT/'outputs/unusual_joint_motion_validation_20261113/joint_events.csv'
CASES=[('1625',89,97),('0425',13,21),('1534',31,39),('0463',14,22)]

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    ev=list(csv.DictReader(EVENTS.open(encoding='utf-8-sig',newline=''))); OUT.mkdir(parents=True)
    for sid,start,end in CASES:
        with np.load(SRC/f'{sid}.npz') as d: xy=d['xy'][:,SWAP]
        frames=[]
        for frame in range(start,end+1):
            im=draw(sid,frame-1)
            targets=[e for e in ev if e['id']==sid and int(e['frame_current'])==frame]
            for e in targets:
                j=NAMES.index(e['joint']); now=xy[frame-1,j]; prev=xy[frame-2,j]
                if valid(now) and valid(prev):
                    cv2.arrowedLine(im,tuple(prev.astype(int)),tuple(now.astype(int)),(0,200,255),2,cv2.LINE_AA,tipLength=.15)
                    cv2.circle(im,tuple(now.astype(int)),7,(0,200,255),2,cv2.LINE_AA)
            h,w=im.shape[:2]; im=cv2.copyMakeBorder(im,0,28,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            cv2.putText(im,f'Orange = flagged jump from previous frame | joints: {len(targets)}',(5,h+19),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,210,255),1,cv2.LINE_AA)
            frames.append(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        frames[0].save(OUT/f'{sid}_{start}_{end}.gif',save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False)
    (OUT/'metadata.json').write_text(json.dumps({'cases':CASES,'display_fps':4,'event_sha256':sha(EVENTS),'script_sha256':sha(Path(__file__)),'note':'orange marks candidate current coordinates and movement from previous frame; threshold flags not confirmed errors; GT-assisted crop and selection'},indent=2),encoding='utf-8')
    print('Rendered',len(CASES),'GIFs')

if __name__=='__main__': main()
