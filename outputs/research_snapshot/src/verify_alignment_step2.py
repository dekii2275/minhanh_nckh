"""Independent artifact checks after the alignment validation experiment."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/alignment_step2'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    provenance = json.loads((OUT/'provenance.json').read_text())
    result = json.loads((OUT/'results.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    manifest = json.loads((OUT/'validation_manifest.json').read_text())
    df = pd.read_csv(OUT/'per_video_predictions.csv', dtype={'id':str})
    assert len(df) == 253 == df.id.nunique()
    assert set(df.id) == set(protocol['validation_ids']) == {r['id'] for r in manifest}
    assert set(df.id).isdisjoint(protocol['train_ids'])
    assert set(df.label) == set(range(15))
    assert df.frames.sum() == 17747 == result['frames']
    assert all(r['official_train']==1 for r in manifest)
    for key,p in [('checkpoint_sha256',provenance['checkpoint']),
                  ('yolo_sha256','models/yolo11n-pose.pt'),
                  ('calibrator_sha256','src/skeleton_calibrator.py'),
                  ('classifier_code_sha256','src/models/action_bilstm.py'),
                  ('feature_code_sha256','src/data/action_dataset.py'),
                  ('script_sha256','src/evaluate_alignment_step2.py')]:
        assert sha(ROOT/p) == provenance[key], p
    probs = np.load(OUT/'probabilities.npz')
    assert np.array_equal(probs['ids'],df.id.to_numpy())
    assert np.array_equal(probs['labels'],df.label.to_numpy())
    y = df.label.to_numpy()
    for c in ['C1','C2','C3','C4']:
        p = df[f'{c}_prediction'].to_numpy()
        assert np.array_equal(p,probs[c].argmax(1))
        assert np.isfinite(probs[c]).all()
        np.testing.assert_allclose(probs[c].sum(1),1,atol=1e-6)
        correct = y==p
        assert np.array_equal(correct,df[f'{c}_correct'])
        accuracy = correct.mean()*100
        # Independent F1 expression; no import of the experiment's metric helper.
        f1s = []
        for k in range(15):
            tp = ((y==k)&(p==k)).sum()
            denom = (y==k).sum()+(p==k).sum()
            f1s.append(2*tp/denom if denom else 0)
        f1 = np.mean(f1s)*100
        assert abs(accuracy-result['metrics'][c]['top1_accuracy']) < 1e-10
        assert abs(f1-result['metrics'][c]['macro_f1']) < 1e-10
    swap = df.production_swap.to_numpy(dtype=bool)
    np.testing.assert_allclose(probs['C3'][swap],probs['C2'][swap],atol=1e-6)
    np.testing.assert_allclose(probs['C3'][~swap],probs['C1'][~swap],atol=1e-6)
    for pair in result['comparisons']:
        a,b = pair['comparison'].split('-')
        diff = df[f'{a}_correct'].to_numpy(dtype=int)-df[f'{b}_correct'].to_numpy(dtype=int)
        assert (diff==1).sum()==pair['improved_videos']
        assert (diff==-1).sum()==pair['harmed_videos']
        assert abs(diff.mean()*100-pair['accuracy_delta_pp'])<1e-10
    report = ROOT/'reports/LEFT_RIGHT_ALIGNMENT_STEP2_2026-10-29.md'
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        assert (report.parent/link.split('#')[0]).exists(),link
    verification = {'passed':True,'validation_videos':len(df),'validation_frames':int(df.frames.sum()),
        'checks':['unique validation IDs and no train overlap',
            'unchanged model, calibration, features, evaluation-script hashes',
            'all four metrics independently recomputed',
            'posterior and per-video predictions consistent',
            'C3 posterior matches C2 on swap and C1 on identity clips',
            'paired improvement/harm counts reproduce deltas',
            'report artifact links valid']}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification))

if __name__ == '__main__':
    main()
