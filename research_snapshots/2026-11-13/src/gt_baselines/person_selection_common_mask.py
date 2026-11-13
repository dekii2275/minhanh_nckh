"""Sensitivity analysis: compare person choices on identical available frames."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import compare_person_selection as base

OUT=base.ROOT/'outputs/person_selection_validation_20261102'
samples=base.rows(); data=[]
for row in samples:
    d=np.load(OUT/'detections'/f"{row['id']}.npz")
    offsets=d['offsets']; xy=d['xy']; boxes=d['boxes']; gt=d['gt_bbox']; T=int(row['nframes'])
    baseline=np.full((T,12,2),np.nan,np.float32); assisted=baseline.copy()
    for t in range(T):
        lo,hi=offsets[t:t+2]; b=boxes[lo:hi]
        score=base.ious(b[:,:4],gt[t]);
        if len(b)==0 or not np.isfinite(score).all() or score.max()<.3: continue
        baseline[t]=xy[lo+int(np.argmax(b[:,4])),5:17]
        assisted[t]=xy[lo+int(np.argmax(score)),5:17]
    data.append((row,base.resample(baseline),base.resample(assisted)))
device='cuda' if torch.cuda.is_available() else 'cpu'
target=[base.ACTION_CLASSES.index(v[0]['action']) for v in data]
inp=[torch.tensor(np.stack([v[i] for v in data]),device=device) for i in (1,2)]
summary=[]; predictions=[]
for seed in (0,1,2):
    net=base.ActionBiLSTMAttentionClassifier().to(device)
    net.load_state_dict(torch.load(base.CKPTS/f'seed{seed}'/'best.pt',map_location=device,weights_only=True)); net.eval()
    pred=[]
    with torch.inference_mode():
        for x in inp:
            y=[]
            for i in range(0,len(x),32): y.extend(net(x[i:i+32]).argmax(-1).cpu().tolist())
            pred.append(y)
    delta=(np.asarray(pred[1])==target).astype(int)-(np.asarray(pred[0])==target).astype(int)
    summary.append({'seed':seed,'confidence_common_mask':base.metric(target,pred[0]),'gt_assisted_common_mask':base.metric(target,pred[1]),'accuracy_difference_pp':100*float(delta.mean()),'improved':int((delta>0).sum()),'worsened':int((delta<0).sum())})
    predictions.extend({'seed':seed,'id':v[0]['id'],'confidence_prediction':a,'gt_assisted_prediction':b} for v,a,b in zip(data,*pred))
result={'explanation':'Both branches use missing for all frames with no GT-bbox match at IoU 0.3. Sensitivity analysis, not autonomous deployment.', 'samples':len(data),'runs':summary,'mean_difference_pp':float(np.mean([v['accuracy_difference_pp'] for v in summary])), 'script_sha256':base.sha(Path(__file__))}
(OUT/'summary_common_mask_iou_0_3.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (OUT/'predictions_common_mask_iou_0_3.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
print(json.dumps(result,indent=2))
