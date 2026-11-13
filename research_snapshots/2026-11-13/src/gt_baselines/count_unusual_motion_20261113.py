"""Count high-confidence large YOLO joint displacements on validation; not error prevalence."""
import csv, json, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from targeted_pose_pilot_20261113 import ROOT, MAN, SRC, LABEL, SWAP, NAMES, valid, sha

OUT=ROOT/'outputs/unusual_joint_motion_validation_20261113'

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    rows=[r for r in csv.DictReader(MAN.open(encoding='utf-8-sig',newline='')) if r['seed0_split']=='validation']
    assert len(rows)==253
    events=[]; pervideo=[]; input_hashes={}; joint_counter=Counter(); action_counter=Counter(); total_eligible_frames=0; total_eligible_pairs=0; scale_missing=0
    for r in rows:
        sid=r['id']; path=SRC/f'{sid}.npz'; assert sha(LABEL/f'{sid}.mat')==r['label_sha256']; input_hashes[sid]=sha(path)
        with np.load(path) as d:
            assert str(d['annotation_sha256'])==r['label_sha256']
            xy=d['xy'][:,SWAP].astype(float); conf=d['conf'][:,SWAP].astype(float); status=d['status']
        assert len(xy)==int(r['nframes'])
        good=valid(xy)&np.isfinite(conf)&(conf>=.5)&(status[:,None]==3)
        torso=[]
        for a,b in [(0,6),(1,7)]:
            m=valid(xy[:,a])&valid(xy[:,b])&(status==3)
            torso.extend(np.linalg.norm(xy[m,a]-xy[m,b],axis=1))
        torso=np.asarray(torso); torso=torso[np.isfinite(torso)&(torso>1)]
        if len(torso)<5:
            scale_missing+=1; pervideo.append({'id':sid,'action':r['action'],'frames':len(xy),'scale_px':'','eligible_frames':0,'jump_frames':0,'spike_frames':0,'joint_events':0,'status':'insufficient_scale'}); continue
        scale=float(np.median(torso)); pair=good[1:]&good[:-1]; total_eligible_pairs+=int(pair.sum()); eligible=np.any(pair,axis=1); total_eligible_frames+=int(eligible.sum())
        delta=np.linalg.norm(xy[1:]-xy[:-1],axis=2)/scale; jump=pair&(delta>1.5)
        spike=np.zeros_like(good); triple=good[2:]&good[1:-1]&good[:-2]
        distance=np.linalg.norm(xy[2:]-xy[:-2],axis=2)/scale
        deviation=np.linalg.norm(xy[1:-1]-(xy[2:]+xy[:-2])/2,axis=2)/scale
        spike[1:-1]=triple&(distance<=2)&(deviation>.75)
        for t,j in np.argwhere(jump):
            events.append({'id':sid,'action':r['action'],'frame_previous':int(t+1),'frame_current':int(t+2),'joint':NAMES[j],'movement_in_torso_scales':float(delta[t,j]),'previous_confidence':float(conf[t,j]),'current_confidence':float(conf[t+1,j]),'scale_px':scale,'isolated_spike_at_current':int(spike[t+1,j])}); joint_counter[NAMES[j]]+=1
        jf=int(np.any(jump,axis=1).sum()); sf=int(np.any(spike,axis=1).sum())
        if jf: action_counter[r['action']]+=1
        pervideo.append({'id':sid,'action':r['action'],'frames':len(xy),'scale_px':scale,'eligible_frames':int(eligible.sum()),'jump_frames':jf,'spike_frames':sf,'joint_events':int(jump.sum()),'status':'analyzed'})
    pervideo.sort(key=lambda x:(-x['jump_frames'],-x['joint_events'],x['id']))
    OUT.mkdir(parents=True)
    for name,data in [('per_video.csv',pervideo),('joint_events.csv',events)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary={'validation_videos':len(rows),'videos_insufficient_scale':scale_missing,'total_frames':sum(v['frames'] for v in pervideo),'eligible_current_frames':total_eligible_frames,'eligible_joint_pairs':total_eligible_pairs,'videos_with_jump':sum(v['jump_frames']>0 for v in pervideo),'jump_frames':sum(v['jump_frames'] for v in pervideo),'jump_joint_events':len(events),'videos_with_spike':sum(v['spike_frames']>0 for v in pervideo),'spike_frames':sum(v['spike_frames'] for v in pervideo),'joint_counts_jump':dict(joint_counter),'action_video_counts_jump':dict(action_counter),'threshold':'adjacent valid matched points confidence >=0.5, Euclidean displacement >1.5 * clip median predicted same-side shoulder-to-hip length; frame indexed as later frame','spike_definition':'three valid matched points confidence >=0.5; center deviation from midpoint >0.75 torso scale and endpoint distance <=2 torso scales','scale':'median over matched frames of valid predicted left/right shoulder-hip lengths >1 pixel, require >=5; no GT keypoints used','caveats':'GT bbox crop and IoU matching, fixed left-right swap; pixel displacement includes subject and camera movement; isolated spike and jump are different rules, overlap possible; misses low confidence, missing pose, slow drift; not a count of true errors or identity switches; exploratory thresholds','manifest_sha256':sha(MAN),'script_sha256':sha(Path(__file__)),'common_script_sha256':sha(ROOT/'src/gt_baselines/targeted_pose_pilot_20261113.py'),'pose_hashes':input_hashes}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='pose_hashes'},indent=2))

if __name__=='__main__': main()
