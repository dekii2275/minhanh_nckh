"""Frozen Protocol-v2 GT-trained 24D classifier: four Penn validation inputs.
No training, parameter search, official-test loading, or GT-based swap decisions.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from data.action_dataset import (ACTION_CLASSES, ACTION_TO_ID, COCO_TO_CANONICAL,
    PennActionSkeletonDataset, collate_skeleton_batch, normalize_skeleton_sequence)
from models.action_bilstm import ActionBiLSTMAttentionClassifier
from run_inference import COCO_KEYPOINTS
from skeleton_calibrator import FullCalibrationPipeline
from train_action_recognition import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/alignment_step2'
DATA = ROOT / 'data/raw/Penn_Action'
CHECKPOINT = ROOT / 'outputs/action_recognition/protocol_v2/multi_seed/baseline/seed0/best_bilstm_model.pt'
YOLO_PATH = ROOT / 'models/yolo11n-pose.pt'
CAN = COCO_TO_CANONICAL
SWAP = np.arange(12).reshape(6, 2)[:, ::-1].ravel()
CONFIGS = ['C1', 'C2', 'C3', 'C4']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))

def metadata():
    summary = json.loads(CHECKPOINT.with_name('action_recognition_summary.json').read_text())
    protocol = summary['protocol']
    assert protocol['version'] == 2 and protocol['seed'] == 0
    assert protocol['train_modality'] == 'ground_truth'
    train = set(protocol['sequence_ids']['train_gt'])
    val = sorted(protocol['sequence_ids']['validation_gt'])
    assert len(train) == 1005 and len(val) == 253 and len(set(val)) == len(val)
    assert train.isdisjoint(val)
    # Access only official-training labels; never build/load the mixed train/test cache.
    by_action = {}
    annotations = {}
    for sid in sorted(train | set(val)):
        m = sio.loadmat(DATA / 'labels' / f'{sid}.mat')
        assert int(m['train'].item()) == 1, sid
        action = str(m['action'][0])
        assert action in ACTION_TO_ID
        by_action.setdefault(action, []).append(sid)
        if sid in val:
            annotations[sid] = m
    # Independently reconstruct the original action-stratified seed0 holdout.
    rng = random.Random(protocol['seed'])
    reconstructed = []
    for action in ACTION_CLASSES:
        ids = sorted(by_action[action]); rng.shuffle(ids)
        n = max(1, min(len(ids)-1, round(len(ids)*protocol['validation_fraction'])))
        reconstructed.extend(ids[:n])
    assert set(reconstructed) == set(val)
    manifest = []
    for sid in val:
        m = annotations[sid]
        manifest.append({'id': sid, 'action': str(m['action'][0]),
            'label': ACTION_TO_ID[str(m['action'][0])],
            'frames': int(m['nframes'].item()), 'official_train': 1,
            'annotation_sha256': sha(DATA / 'labels' / f'{sid}.mat')})
    dump(OUT / 'validation_manifest.json', manifest)
    dump(OUT / 'protocol.json', {k: v for k, v in protocol.items() if k != 'sequence_ids'} | {
        'train_ids': sorted(train), 'validation_ids': val,
        'checkpoint_best_epoch': summary['best_epoch'],
        'checkpoint_best_validation_gt_top1': summary['best_validation_top1_acc']})
    return manifest, annotations, summary['best_validation_top1_acc']

def classifier():
    model = ActionBiLSTMAttentionClassifier(input_size=24, hidden_size=128, num_layers=2,
        num_classes=15, attention_heads=4, dropout=0.3, bidirectional=True, pooling='mean')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert tuple(state['lstm.weight_ih_l0'].shape) == (512, 24)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def predict(model, samples):
    loader = DataLoader(PennActionSkeletonDataset(samples), batch_size=32, shuffle=False,
        collate_fn=collate_skeleton_batch)
    records = []
    with torch.inference_mode():
        for features, lengths, labels, ids in loader:
            assert torch.isfinite(features).all()
            probs = torch.softmax(model(features, lengths=lengths), dim=1).cpu().numpy()
            for sid, y, p in zip(ids, labels.tolist(), probs):
                records.append({'id': sid, 'label': y, 'prediction': int(p.argmax()),
                    'probabilities': p.tolist()})
    return records

def check_gt(model, manifest, annotations, expected):
    samples = []
    for item in manifest:
        m = annotations[item['id']]
        coords = np.stack([m['x'][:, 1:], m['y'][:, 1:]], axis=-1).astype(np.float32)
        invisible = (m['visibility'][:, 1:] == 0) | (coords[:, :, 0] <= 0)
        coords[invisible] = np.nan
        samples.append({'seq_id': item['id'], 'features': normalize_skeleton_sequence(coords), 'label': item['label']})
    rec = predict(model, samples)
    accuracy = np.mean([r['prediction'] == r['label'] for r in rec])*100
    assert abs(accuracy-expected) < 1e-6, (accuracy, expected)
    dump(OUT / 'gt_reproduction_check.json', {'expected_top1': expected, 'observed_top1': float(accuracy),
        'videos': len(rec), 'purpose': 'checkpoint/feature/split validation only; GT never enters swap decisions'})
    print(f'GT CHECK PASS {accuracy:.8f}%', flush=True)

def infer(manifest, fingerprint):
    model = None
    for index, item in enumerate(manifest):
        sid = item['id']; path = OUT / 'raw_predictions' / f'{sid}.npz'
        if path.exists():
            z = np.load(path)
            assert str(z['fingerprint'].item()) == fingerprint
            assert z['xy'].shape == (item['frames'], 17, 2)
            continue
        if model is None:
            model = YOLO(str(YOLO_PATH))
        paths = sorted((DATA / 'frames' / sid).glob('*.jpg'))
        assert [p.name for p in paths] == [f'{f:06d}.jpg' for f in range(1, item['frames']+1)]
        xy = []; conf = []; n = []; boxconf = []; boxes = []
        for start in range(0, len(paths), 16):
            results = model.predict([str(p) for p in paths[start:start+16]], imgsz=640,
                conf=0.25, device='cpu', verbose=False)
            for r in results:
                count = len(r.boxes); n.append(count)
                if count:
                    k = int(r.boxes.conf.argmax())
                    xy.append(r.keypoints.xy[k].cpu().numpy())
                    conf.append(r.keypoints.conf[k].cpu().numpy())
                    boxconf.append(float(r.boxes.conf[k])); boxes.append(r.boxes.xyxy[k].cpu().numpy())
                else:
                    xy.append(np.full((17,2), np.nan)); conf.append(np.full(17, np.nan))
                    boxconf.append(np.nan); boxes.append(np.full(4, np.nan))
        np.savez_compressed(path, xy=xy, conf=conf, n=n, boxconf=boxconf, boxes=boxes,
            fingerprint=np.array(fingerprint))
        print(f'YOLO {index+1}/{len(manifest)} {sid} {item["action"]} {len(paths)} frames', flush=True)

def as_df(xy, conf):
    return pd.DataFrame({f'pred_{name}_{suffix}': (conf[:, j] if suffix == 'conf' else xy[:, j, k])
        for j, name in enumerate(COCO_KEYPOINTS) for k, suffix in enumerate(['x','y','conf'])})

def coords(df):
    # Match existing action_dataset: float32 before normalization, no confidence mask on raw.
    return np.stack([df[[f'pred_{name}_{s}' for name in CAN]].to_numpy(dtype=np.float32)
        for s in ['x','y']], axis=-1)

def calibration_inputs(raw):
    pipe = FullCalibrationPipeline()
    pre = pipe.stage1_interp.calibrate_sequence(raw)
    pre = pipe.stage2_bone.calibrate_sequence(pre)
    pre = pipe.stage3_smooth.calibrate_sequence(pre)
    # Record the permutation by feeding stage4 marker coordinates with the exact
    # production shoulder columns. This executes the existing decision branch,
    # including short clips/insufficient valid shoulders; it never consults GT.
    marker = pre.copy()
    marker['pred_left_wrist_x'] = 1.0
    marker['pred_right_wrist_x'] = 2.0
    decision = pipe.stage4_lr.calibrate_sequence(marker)
    swap = bool((decision['pred_left_wrist_x'] == 2.0).all())
    full = pipe.stage4_lr.calibrate_sequence(pre)
    # Independently check the public production entrypoint exactly.
    direct = FullCalibrationPipeline().calibrate_sequence(raw)
    for name in COCO_KEYPOINTS:
        for suffix in ['x','y','conf']:
            col = f'pred_{name}_{suffix}'
            assert np.allclose(full[col], direct[col], equal_nan=True), col
    base = coords(raw)
    c3 = base[:, SWAP].copy() if swap else base.copy()
    processed = coords(pre)
    assert np.allclose(coords(full), processed[:, SWAP] if swap else processed, equal_nan=True)
    valid = pre.pred_left_shoulder_x.notna() & pre.pred_right_shoulder_x.notna()
    median = float(np.median((pre.pred_left_shoulder_x-pre.pred_right_shoulder_x)[valid])) if valid.any() else None
    return {'C1': base, 'C2': base[:, SWAP].copy(), 'C3': c3, 'C4': coords(full)}, swap, median

def evaluate(model, manifest):
    samples = {c: [] for c in CONFIGS}; decisions = []
    for item in manifest:
        z = np.load(OUT/'raw_predictions'/f'{item["id"]}.npz')
        inputs, swap, delta = calibration_inputs(as_df(z['xy'], z['conf']))
        for c, arr in inputs.items():
            feat = normalize_skeleton_sequence(arr)
            assert feat.shape == (item['frames'], 24) and np.isfinite(feat).all()
            samples[c].append({'seq_id': item['id'], 'label': item['label'], 'features': feat})
        decisions.append({'id': item['id'], 'production_swap': swap, 'production_median_dx': delta,
            'no_detection_frames': int((z['n'] == 0).sum()), 'multiple_people_frames': int((z['n'] > 1).sum())})
    pd.DataFrame(decisions).to_csv(OUT/'alignment_decisions.csv', index=False)
    wide = pd.DataFrame(manifest).merge(pd.DataFrame(decisions), on='id', validate='one_to_one')
    metrics = {}; all_probs = {}
    for c in CONFIGS:
        rec = predict(model, samples[c]); assert [r['id'] for r in rec] == wide.id.tolist()
        pred = [r['prediction'] for r in rec]; y = [r['label'] for r in rec]
        acc, f1, _, cm = compute_metrics(y, pred, 15)
        metrics[c] = {'top1_accuracy': acc, 'macro_f1': f1, 'correct': int(sum(np.array(y)==pred)), 'videos': len(y)}
        np.savetxt(OUT/f'{c}_confusion_matrix.csv', cm, delimiter=',', fmt='%d')
        wide[f'{c}_prediction'] = pred
        wide[f'{c}_predicted_action'] = [ACTION_CLASSES[p] for p in pred]
        wide[f'{c}_correct'] = np.array(pred) == np.array(y)
        all_probs[c] = np.array([r['probabilities'] for r in rec])
        print(c, metrics[c], flush=True)
    np.savez_compressed(OUT/'probabilities.npz', ids=wide.id.to_numpy(dtype=str), labels=wide.label.to_numpy(), **all_probs)
    for c in CONFIGS:
        metrics[c]['improvement_vs_raw_pp'] = metrics[c]['top1_accuracy']-metrics['C1']['top1_accuracy']
    return wide, metrics

def comparisons(wide, metrics):
    # Paired bootstrap: same resampled videos for A and B, stratified by GT class
    # to retain the validation class composition. No model refits or parameter tuning.
    rng = np.random.default_rng(20261029)
    y = wide.label.to_numpy(); n = len(y); B = 20000
    groups = [np.where(y == k)[0] for k in range(15)]
    indices = np.concatenate([rng.choice(g, size=(B,len(g)), replace=True) for g in groups], axis=1)
    result = []
    for a, b in [('C2','C1'),('C3','C1'),('C4','C3'),('C4','C2')]:
        ca = wide[f'{a}_correct'].to_numpy(dtype=int); cb = wide[f'{b}_correct'].to_numpy(dtype=int)
        boot = (ca-cb)[indices].mean(1)*100
        delta = metrics[a]['top1_accuracy']-metrics[b]['top1_accuracy']
        low, high = np.percentile(boot, [2.5,97.5])
        improved = (ca == 1)&(cb == 0); harmed = (ca == 0)&(cb == 1)
        name = f'{a}-{b}'
        wide[f'{name}_outcome'] = np.where(improved,'improved',np.where(harmed,'harmed',np.where(ca==1,'both_correct','both_wrong')))
        result.append({'comparison': name, 'accuracy_delta_pp': delta,
            'accuracy_ci95_low_pp': float(low), 'accuracy_ci95_high_pp': float(high),
            'macro_f1_delta_pp': metrics[a]['macro_f1']-metrics[b]['macro_f1'],
            'improved_videos': int(improved.sum()), 'harmed_videos': int(harmed.sum()),
            'both_correct': int(((ca==1)&(cb==1)).sum()), 'both_wrong': int(((ca==0)&(cb==0)).sum())})
        assert abs(delta-(improved.sum()-harmed.sum())/n*100) < 1e-8
    wide.to_csv(OUT/'per_video_predictions.csv', index=False)
    pd.DataFrame(result).to_csv(OUT/'paired_comparisons.csv', index=False)
    pd.DataFrame(metrics).T.to_csv(OUT/'configuration_metrics.csv', index_label='configuration')
    perclass = []
    for k, action in enumerate(ACTION_CLASSES):
        mask = y == k
        for c in CONFIGS:
            perclass.append({'action':action,'configuration':c,'videos':int(mask.sum()),'top1_accuracy':float(wide.loc[mask,f'{c}_correct'].mean()*100)})
    pd.DataFrame(perclass).to_csv(OUT/'per_class_accuracy.csv', index=False)
    return result

def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/'raw_predictions').mkdir(exist_ok=True)
    manifest, annotations, expected = metadata()
    # Freeze experiment choices before generating predictions or computing results.
    provenance = {'checkpoint':str(CHECKPOINT.relative_to(ROOT)), 'checkpoint_sha256':sha(CHECKPOINT),
        'yolo_sha256':sha(YOLO_PATH), 'calibrator_sha256':sha(ROOT/'src/skeleton_calibrator.py'),
        'classifier_code_sha256':sha(ROOT/'src/models/action_bilstm.py'),
        'feature_code_sha256':sha(ROOT/'src/data/action_dataset.py'),
        'script_sha256':sha(__file__), 'torch_version':torch.__version__,
        'ultralytics_version':__import__('ultralytics').__version__,
        'numpy_version':np.__version__, 'pandas_version':pd.__version__,
        'classifier':{'input_size':24,'hidden_size':128,'num_layers':2,'num_classes':15,
            'attention_heads':4,'dropout':0.3,'bidirectional':True,'pooling':'mean'},
        'yolo':{'imgsz':640,'conf':0.25,'device':'cpu','batch_size':16,
            'person_selection':'highest box confidence; no GT'},
        'calibration':{'conf_threshold':0.35,'max_gap':15,'min_bone_ratio':0.4,
            'max_bone_ratio':1.2,'freq':30.0,'min_cutoff':1.0,'beta':0.05,'d_cutoff':1.0,
            'lr_mode':'auto_viewer_centric'},
        'evaluation_batch_size':32,'training_performed':False,'official_test_evaluated':False,
        'bootstrap':{'unit':'video','paired':True,'stratified_by':'GT action','replicates':20000,
            'seed':20261029,'ci':'percentile 95%; unadjusted; conditional on fixed checkpoint'}}
    fingerprint = hashlib.sha256(json.dumps({'model':provenance['yolo_sha256'],'yolo':provenance['yolo'],
        'version':provenance['ultralytics_version']},sort_keys=True).encode()).hexdigest()
    dump(OUT/'provenance.json', provenance)
    model = classifier();check_gt(model, manifest, annotations, expected)
    infer(manifest, fingerprint)
    wide, metrics = evaluate(model, manifest)
    pairs = comparisons(wide, metrics)
    assert sha(CHECKPOINT) == provenance['checkpoint_sha256']
    assert sha(ROOT/'src/skeleton_calibrator.py') == provenance['calibrator_sha256']
    dump(OUT/'results.json', {'videos':len(wide),'frames':int(wide.frames.sum()),
        'production_swap_clips':int(wide.production_swap.sum()),'metrics':metrics,'comparisons':pairs})
    print('COMPLETE', json.dumps(pairs), flush=True)

if __name__ == '__main__':
    main()
