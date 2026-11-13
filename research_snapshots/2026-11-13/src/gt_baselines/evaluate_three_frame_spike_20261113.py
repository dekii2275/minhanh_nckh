"""Isolate the previously piloted 3-frame spike rule, then train and compare."""
import csv,json,time,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
import compare_person_selection as base
import gt_crop_pose_experiment as source
import train_drop_jump_frames_20261113 as experiment
import gt_crop_correction_experiment as correction_experiment
from targeted_pose_pilot_20261113 import correct,sha,valid,NAMES

ROOT=source.ROOT
OUT=ROOT/'outputs/three_frame_spike_isolated_20261113'
OLD=ROOT/'outputs/gt_crop_corrections_20261103'
experiment.OUT=OUT

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    rows=source.rows(); old=np.load(OLD/'features_fixed_swap.npz');assert sha(OLD/'features_fixed_swap.npz')==json.loads((OLD/'prepared.json').read_text())['feature_sha256']['fixed_swap'];assert np.array_equal(old['ids'],[r['id'] for r in rows])
    OUT.mkdir(parents=True)
    protocol={'definition':'three adjacent valid YOLO points confidence >=0.5; outside distance <=2 * clip median YOLO shoulder-hip scale; center deviation from midpoint >0.75 scale; replace center with midpoint; decisions simultaneous based on original coordinates',
       'warning':'same spike rule explored in combined pilot; isolating it is exploratory, validation reused; future frame and clip scale used, offline',
       'split':'seed0 1005 train and 253 validation; no test','model':'BiLSTM attention 24 coordinates fixed swap; 3 seeds, 50 epochs, batch 32, Adam lr .001 wd .0001 cosine eta_min .00001; best validation accuracy, earliest tie',
       'baseline':'prior fixed-swap 3 seed training on identical split, GPU, and budget','inference':'YOLO GT bbox crop + IoU selection; GT keypoints only for pose evaluation',
       'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'manifest_sha256':sha(base.MANIFEST),'source_script_sha256':sha(ROOT/'src/gt_baselines/targeted_pose_pilot_20261113.py'),'script_sha256':sha(Path(__file__)), 'trainer_sha256':sha(Path(experiment.trainer.__file__))}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    features=[]; removals=[]; records=[]; pck_video=[]; gt_n=good0=good1=common_n=0;err0=err1=0.; motion_n=0; motion0=motion1=0.; start=time.time()
    for i,r in enumerate(rows):
        sid=r['id'];path=source.OUT/'poses'/f'{sid}.npz';assert sha(base.LABELS/f'{sid}.mat')==r['label_sha256']
        with np.load(path) as a:
            assert str(a['annotation_sha256'])==r['label_sha256'];raw=a['xy'];conf=a['conf'];status=a['status'];assert len(raw)==int(r['nframes']); x=raw[:,source.SWAP]
        transformed,kind=correct(x,conf[:,source.SWAP]);flags=kind==2;y=x.copy();y[flags]=transformed[flags]
        assert not np.any(flags & (status[:,None]!=3)) and not np.any(flags[1:] & (status[:-1,None]!=3)) and not np.any(flags[:-1] & (status[1:,None]!=3))
        assert np.array_equal(base.resample(x),old['x'][i]);f=base.resample(y);assert np.isfinite(f).all();features.append(f)
        removals.append({'id':sid,'action':r['action'],'split':r['seed0_split'],'frames':len(x),'modified_frames':int(flags.any(1).sum()),'modified_joints':int(flags.sum()),'pose_sha256':sha(path)})
        if r['seed0_split']=='validation':
            m=loadmat(base.LABELS/f'{sid}.mat');gt=np.stack([m['x'][:,1:13],m['y'][:,1:13]],-1).astype(float);visible=m['visibility'][:,1:13]>0;bb=np.full((len(x),4),np.nan); n=min(len(x),len(m['bbox']));bb[:n]=m['bbox'][:n];wh=bb[:,2:]-bb[:,:2];bok=np.isfinite(bb).all(1)&(wh>0).all(1);diag=np.linalg.norm(wh,axis=1)
            eligible=visible&valid(gt)&bok[:,None];obs=valid(x)&(status[:,None]==3);common=eligible&obs
            d0=np.linalg.norm(x-gt,axis=2)/diag[:,None];d1=np.linalg.norm(y-gt,axis=2)/diag[:,None]
            g0=common&(d0<=.2);g1=common&(d1<=.2);gt_n+=int(eligible.sum());good0+=int(g0.sum());good1+=int(g1.sum());common_n+=int(common.sum());err0+=float(d0[common].sum());err1+=float(d1[common].sum())
            pair=common[1:]&common[:-1]; gm=gt[1:]-gt[:-1];m0=np.linalg.norm((x[1:]-x[:-1])-gm,axis=2)/diag[1:,None];m1=np.linalg.norm((y[1:]-y[:-1])-gm,axis=2)/diag[1:,None];motion_n+=int(pair.sum());motion0+=float(m0[pair].sum());motion1+=float(m1[pair].sum())
            pck_video.append({'id':sid,'eligible':int(eligible.sum()),'before':int(g0.sum()),'after':int(g1.sum())})
            for t,j in np.argwhere(flags&common):records.append({'id':sid,'action':r['action'],'frame':int(t+1),'joint':NAMES[j],'error_before':float(d0[t,j]),'error_after':float(d1[t,j]),'pck_delta':int(g1[t,j])-int(g0[t,j])})
        if (i+1)%250==0:print(f'prepared {i+1}/{len(rows)} ({time.time()-start:.1f}s)',flush=True)
    np.savez_compressed(OUT/'features_dropped.npz',x=np.stack(features),ids=old['ids'],y=old['y'],split=old['split'])
    for name,data in [('per_video.csv',removals),('validation_changed_joints.csv',records)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    splits={s:{'videos':sum(r['split']==s for r in removals),'videos_changed':sum(r['split']==s and r['modified_frames']>0 for r in removals),'frames_changed':sum(r['modified_frames'] for r in removals if r['split']==s),'joints_changed':sum(r['modified_joints'] for r in removals if r['split']==s)} for s in ['train','validation']}
    rng=np.random.default_rng(20261113);delta=np.array([(v['after']-v['before'])/v['eligible']*100 if v['eligible'] else 0 for v in pck_video]);boot=rng.integers(0,len(delta),(10000,len(delta)))
    result={'splits':splits,'gt_eligible':gt_n,'common_observed_gt':common_n,'changed_gt_eligible':len(records),'changed_closer':sum(r['error_after']<r['error_before'] for r in records),'changed_farther':sum(r['error_after']>r['error_before'] for r in records),'changed_pck_improved':sum(r['pck_delta']>0 for r in records),'changed_pck_worsened':sum(r['pck_delta']<0 for r in records),'pck_all_before_pct':good0/gt_n*100,'pck_all_after_pct':good1/gt_n*100,'video_mean_pck_delta_pp':float(delta.mean()),'video_bootstrap95_pp':np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),'mean_normalized_error_common_before':err0/common_n,'mean_normalized_error_common_after':err1/common_n,'motion_pairs':motion_n,'mean_displacement_error_before':motion0/motion_n,'mean_displacement_error_after':motion1/motion_n,'motion_definition':'mean norm of predicted 2D inter-frame displacement minus GT displacement, divided by current frame GT bbox diagonal; all common valid GT adjacent pairs; not physical velocity'}
    (OUT/'pose_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(OUT/'prepared.json').write_text(json.dumps({'features_sha256':sha(OUT/'features_dropped.npz'),'baseline_features_reproduced':True,'splits':splits},indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

def frozen():
    data=experiment.dataset()['validation'];records=[]
    for seed in (0,1,2):
        path=source.OUT/'yolo_trained_fixed_swap/bilstm_attention'/f'seed{seed}'/'best.pt';result=correction_experiment.predict(path,data,'cuda');records.append({'seed':seed,**result});print(f'frozen seed {seed} accuracy {result["accuracy_pct"]:.3f}',flush=True)
    (OUT/'frozen_results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    {'prepare':prepare,'train':experiment.train,'summary':experiment.summarize,'frozen':frozen}[sys.argv[1]]()
