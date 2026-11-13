"""Validation-only GPU experiment: delete flagged whole frames then retrain."""
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np
import torch
import compare_person_selection as base
import gt_crop_pose_experiment as source
import train_gt_baselines as trainer
from targeted_pose_pilot_20261113 import valid, sha

OUT=source.ROOT/'outputs/drop_jump_frames_20261113'
PREVIOUS=source.ROOT/'outputs/gt_crop_corrections_20261103'
SEEDS=(0,1,2)

def dump(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def load(path): return json.loads(path.read_text(encoding='utf-8'))

def flag_frames(xy,conf,status):
    good=valid(xy)&np.isfinite(conf)&(conf>=.5)&(status[:,None]==3)
    distances=[]
    for a,b in ((0,6),(1,7)):
        mask=valid(xy[:,a])&valid(xy[:,b])&(status==3)
        distances.extend(np.linalg.norm(xy[mask,a]-xy[mask,b],axis=1))
    distances=np.asarray(distances); distances=distances[np.isfinite(distances)&(distances>1)]
    flags=np.zeros(len(xy),bool)
    if len(distances)<5: return flags,None
    scale=float(np.median(distances)); delta=np.linalg.norm(xy[1:]-xy[:-1],axis=2)
    flags[1:]=np.any((good[1:]&good[:-1])&(delta>1.5*scale),axis=1)
    return flags,scale

def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; CPU fallback prohibited')
    rows=source.rows(); old=np.load(PREVIOUS/'features_fixed_swap.npz')
    assert sha(PREVIOUS/'features_fixed_swap.npz')==load(PREVIOUS/'prepared.json')['feature_sha256']['fixed_swap']
    assert np.array_equal(old['ids'],np.array([r['id'] for r in rows]))
    OUT.mkdir(parents=True)
    protocol={'description':'drop later original frame with any high-confidence joint displacement >1.5 clip median YOLO shoulder-hip scale; compare consecutive ORIGINAL frames, not previous retained frame; same train/validation; resample kept indices uniformly to 64 steps compressing time',
        'scale':'median matched valid predicted shoulder-to-hip distances >1px, minimum 5 values; otherwise no frames dropped',
        'threshold':'confidence >=0.5 both frames, valid coords, status matched; no GT keypoints in deletion rule',
        'split':'seed0, 1005 train / 253 validation; no test', 'model':'BiLSTM attention on 24 coordinates, fixed swap',
        'epochs':50,'seeds':list(SEEDS),'batch':32,'optimizer':'Adam lr=.001 wd=.0001, cosine T=50 eta_min=.00001, gradient clip 1',
        'selection':'max validation accuracy, earliest tie', 'baseline':'existing original input trained same budget 3 seeds; recheck features and predictions',
        'caveats':'offline, time compression; validation reused for design and checkpoint selection; GT bbox assisted YOLO crop and matching; threshold heuristic; no true-error labels',
        'manifest_sha256':sha(base.MANIFEST),'source_provenance_sha256':sha(source.OUT/'provenance.json'), 'baseline_features_sha256':sha(PREVIOUS/'features_fixed_swap.npz'),
        'script_sha256':sha(Path(__file__)),'trainer_sha256':sha(Path(trainer.__file__)), 'resampler_sha256':sha(Path(base.__file__)),
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'python':platform.python_version()}
    dump(OUT/'protocol.json',protocol)
    xs=[]; counts=[]; removed=[]; identities=[]; start=time.time()
    for i,r in enumerate(rows):
        sid=r['id']; path=source.OUT/'poses'/f'{sid}.npz'; assert sha(base.LABELS/f'{sid}.mat')==r['label_sha256']
        with np.load(path) as d:
            assert str(d['annotation_sha256'])==r['label_sha256']; xy=d['xy'][:,source.SWAP].astype(float); conf=d['conf'][:,source.SWAP]; status=d['status']
        assert len(xy)==int(r['nframes'])
        # Baseline computed from original dtype for bitwise identity with stored artifact.
        with np.load(path) as d: raw=base.resample(d['xy'][:,source.SWAP])
        if not np.array_equal(raw,old['x'][i]): raise ValueError('Baseline features mismatch '+sid)
        flags,scale=flag_frames(xy,conf,status); assert len(xy[~flags])>=1
        f=base.resample(xy[~flags].astype(np.float32)); assert f.shape==(64,24) and np.isfinite(f).all()
        if not flags.any() and not np.array_equal(f,old['x'][i]): raise ValueError('Unchanged clip feature mismatch '+sid)
        xs.append(f); counts.append({'id':sid,'action':r['action'],'split':r['seed0_split'],'frames_original':len(xy),'frames_dropped':int(flags.sum()),'frames_kept':int((~flags).sum()),'scale_px':scale,'dropped_frame_numbers':' '.join(map(str,np.flatnonzero(flags)+1)),'pose_sha256':sha(path)})
        for t in np.flatnonzero(flags): removed.append({'id':sid,'split':r['seed0_split'],'frame':int(t+1)})
        if (i+1)%200==0: print(f'prepared {i+1}/{len(rows)} seconds={time.time()-start:.0f}',flush=True)
    np.savez_compressed(OUT/'features_dropped.npz',x=np.stack(xs),ids=old['ids'],y=old['y'],split=old['split'])
    with (OUT/'per_video_removals.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    with (OUT/'removed_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['id','split','frame']);w.writeheader();w.writerows(removed)
    # Check exact agreement with independently generated validation counts.
    prev=list(csv.DictReader((source.ROOT/'outputs/unusual_joint_motion_validation_20261113/per_video.csv').open(encoding='utf-8-sig')))
    expected={r['id']:int(r['jump_frames']) for r in prev}
    assert all(expected[r['id']]==r['frames_dropped'] for r in counts if r['split']=='validation')
    dump(OUT/'prepared.json',{'features_sha256':sha(OUT/'features_dropped.npz'),'baseline_all_features_identical':True,'validation_counts_identical_to_prior':True,
        'splits':{s:{'videos':sum(r['split']==s for r in counts),'videos_with_removal':sum(r['split']==s and r['frames_dropped']>0 for r in counts),'frames':sum(r['frames_original'] for r in counts if r['split']==s),'frames_dropped':sum(r['frames_dropped'] for r in counts if r['split']==s),'insufficient_scale':sum(r['split']==s and r['scale_px'] is None for r in counts)} for s in ('train','validation')}})
    print(json.dumps(load(OUT/'prepared.json'),indent=2),flush=True)

def dataset():
    assert sha(OUT/'features_dropped.npz')==load(OUT/'prepared.json')['features_sha256']
    with np.load(OUT/'features_dropped.npz') as a:
        return {s:[(torch.from_numpy(x.copy()),int(y),str(i)) for x,y,i,z in zip(a['x'],a['y'],a['ids'],a['split']) if z==s] for s in ('train','validation')}

def train():
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(4); data=dataset(); args=argparse.Namespace(epochs=50,batch=32)
    for seed in SEEDS: trainer.run('bilstm_attention',seed,data,args,OUT/'trained')

def summarize():
    torch.set_num_threads(4); data=dataset()['validation']; x=torch.stack([v[0] for v in data]); allrows=[]; per=[]; rng=np.random.default_rng(20261113)
    for seed in SEEDS:
        newdir=OUT/'trained/bilstm_attention'/f'seed{seed}'; olddir=PREVIOUS/'retrained/fixed_swap/bilstm_attention'/f'seed{seed}'
        new=load(newdir/'best_validation.json'); old=load(olddir/'best_validation.json'); meta=load(newdir/'provenance.json')
        assert meta['device']=='cuda' and meta['epochs']==50 and len(load(newdir/'history.json'))==50 and sha(newdir/'best.pt')==meta['checkpoint_sha256']
        assert [(i['id'],i['target']) for i in old['predictions']]==[(i['id'],i['target']) for i in new['predictions']]==[(i[2],i[1]) for i in data]
        # Independent CPU checkpoint reload on the saved features.
        net=trainer.model_for('bilstm_attention'); net.load_state_dict(torch.load(newdir/'best.pt',map_location='cpu',weights_only=True));net.eval()
        with torch.inference_mode(): cpu=torch.cat([net(b).argmax(-1) for b in x.split(32)]).tolist()
        mismatch=sum(a!=p['predicted'] for a,p in zip(cpu,new['predictions']))
        changes=[]
        for a,b in zip(new['predictions'],old['predictions']):
            delta=int(a['predicted']==a['target'])-int(b['predicted']==b['target']); changes.append(delta)
            per.append({'seed':seed,'id':a['id'],'target':a['target'],'baseline_prediction':b['predicted'],'drop_prediction':a['predicted'],'delta_correct':delta})
        allrows.append({'seed':seed,'baseline_epoch':old['epoch'],'baseline_accuracy':old['accuracy'],'baseline_macro_f1':old['macro_f1'],'drop_epoch':new['epoch'],'drop_accuracy':new['accuracy'],'drop_macro_f1':new['macro_f1'],'delta_accuracy_pp':new['accuracy']-old['accuracy'],'improved_videos':sum(v>0 for v in changes),'worsened_videos':sum(v<0 for v in changes),'cpu_prediction_mismatches':mismatch,'checkpoint_sha256':meta['checkpoint_sha256'],'seconds':meta['seconds']})
    with (OUT/'paired_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    arr=np.array([[p['delta_correct'] for p in per if p['seed']==s] for s in SEEDS]).mean(axis=0)
    boot=rng.integers(0,len(arr),(10000,len(arr)))
    result={'seeds':allrows,'baseline_accuracy_mean':float(np.mean([r['baseline_accuracy'] for r in allrows])),'drop_accuracy_mean':float(np.mean([r['drop_accuracy'] for r in allrows])),'baseline_macro_f1_mean':float(np.mean([r['baseline_macro_f1'] for r in allrows])),'drop_macro_f1_mean':float(np.mean([r['drop_macro_f1'] for r in allrows])),'delta_accuracy_pp':float(arr.mean()*100),'paired_video_bootstrap95_pp':np.quantile(arr[boot].mean(axis=1)*100,[.025,.975]).tolist(),'bootstrap':'10000 paired video resamples; seed averaged per video; conditional on trained models; excludes seed and selection uncertainty', 'caution':'exploratory reused validation, GT bbox assistance, clip-wide scale uses future frames; no test, no online claim'}
    dump(OUT/'summary.json',result); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train','summary'));a=p.parse_args()
    {'prepare':prepare,'train':train,'summary':summarize}[a.stage]()
