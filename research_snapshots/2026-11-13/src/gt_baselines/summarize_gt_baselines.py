"""Summarize saved validation predictions and verify confusion/manifest alignment."""
import csv, hashlib, json, platform, statistics, sys
from pathlib import Path
import numpy as np
import scipy
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/gt_baselines_20261102_v1'
MANIFEST=ROOT/'report/02112026_nckh/penn_audit_artifacts/video_manifest.csv'
rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')))
val={r['id']:r for r in rows if r['seed0_split']=='validation'}
sys.path.insert(0,str(ROOT/'src'))
from data.action_dataset import ACTION_CLASSES

summary={}
for directory in sorted(OUT.iterdir()):
    if not directory.is_dir(): continue
    runs=[]
    for p in sorted(directory.glob('seed*/best_validation.json')):
        meta=p.parent/'provenance.json'
        if not meta.exists(): continue
        d=json.loads(p.read_text(encoding='utf-8')); preds=d['predictions']
        assert len(preds)==253 and {r['id'] for r in preds}==set(val)
        assert all(r['target']==ACTION_CLASSES.index(val[r['id']]['action']) for r in preds)
        cm=np.zeros((15,15),dtype=int)
        for r in preds: cm[r['target'],r['predicted']]+=1
        assert cm.tolist()==d['confusion_matrix']
        assert abs(100*np.trace(cm)/cm.sum()-d['accuracy'])<1e-8
        assert all(len(r['probabilities'])==15 and abs(sum(r['probabilities'])-1)<1e-5 for r in preds)
        provenance=json.loads(meta.read_text(encoding='utf-8'))
        assert hashlib.sha256((p.parent/'best.pt').read_bytes()).hexdigest()==provenance['checkpoint_sha256']
        runs.append({'seed':d['seed'],'epoch':d['epoch'],'accuracy':d['accuracy'],'macro_f1':d['macro_f1'],'seconds':provenance['seconds'],'parameters':provenance['parameters']})
    if runs:
        summary[directory.name]={'runs':runs,'accuracy_mean':statistics.mean(x['accuracy'] for x in runs),'accuracy_sd_sample':statistics.stdev(x['accuracy'] for x in runs) if len(runs)>1 else None,'macro_f1_mean':statistics.mean(x['macro_f1'] for x in runs)}
files=['src/gt_baselines/train_gt_baselines.py','src/gt_baselines/train_gt_cascade_scratch.py','src/data/action_dataset.py','src/models/action_bilstm.py','src/train_action_recognition.py']
result={'summary':summary,'versions':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},'note':'Checkpoint selected using this same validation set. No test evaluation.'}
(OUT/'summary_verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
