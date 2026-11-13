"""Paired validation ablation of corrections on the GT-cropped YOLO pipeline.

No official test access. Compare frozen YOLO-trained classifiers and classifiers
trained from scratch on each corrected representation using the same budget.
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import compare_person_selection as base
import evaluate_pose_corrections as corrections
import gt_crop_pose_experiment as source
import train_gt_baselines as trainer

METHODS = ('fixed_swap', 'interpolate3', 'mean5', 'median5', 'interpolate3_mean5')
SEEDS = (0, 1, 2)
ROOT = source.ROOT
DEFAULT_OUT = ROOT / 'outputs/gt_crop_corrections_20261103'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def protocol():
    return {
        'methods': list(METHODS), 'seeds': list(SEEDS), 'epochs': 50, 'batch': 32,
        'learning_rate': .001, 'weight_decay': .0001,
        'scheduler': 'CosineAnnealingLR T_max=50 eta_min=0.00001',
        'selection': 'highest validation accuracy; earliest tie',
        'split': 'fixed seed0 manifest: 1005 train / 253 validation; no test',
        'source': str(source.OUT), 'source_provenance_sha256': base.sha(source.OUT/'provenance.json'),
        'manifest_sha256': base.sha(base.MANIFEST),
        'script_sha256': base.sha(Path(__file__)),
        'correction_script_sha256': base.sha(Path(corrections.__file__)),
        'trainer_sha256': base.sha(Path(trainer.__file__)),
        'normalizer_sha256': base.sha(ROOT/'src/data/action_dataset.py'),
        'classifier_sha256': base.sha(ROOT/'src/models/action_bilstm.py'),
        'resampler_sha256': base.sha(Path(base.__file__)),
        'confidence_threshold': corrections.THRESHOLD,
        'max_internal_gap_frames': corrections.MAX_GAP,
        'smoothing': 'centered +/-2 frames, >=3 reliable observations; reliable targets only',
        'missing': 'unresolved predictions unchanged, including NaN; same zeroing/resampling as source',
        'mapping': 'fixed left/right swap exactly once before every correction',
        'inference_information': 'GT bbox crop and GT IoU selection retained; corrections use predicted xy/conf only',
        'caution': 'exploratory validation reused for model selection; centered operations use future frames; not online',
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'frozen_checkpoints': {str(s): base.sha(old_run(s)/'best.pt') for s in SEEDS},
    }


def old_run(seed):
    return source.OUT/'yolo_trained_fixed_swap/bilstm_attention'/f'seed{seed}'


def check_protocol(out):
    if read_json(out/'protocol.json') != protocol():
        raise ValueError('Protocol or source code changed; choose a new output directory')


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', protocol())
    rows = source.rows()
    features = {m: [] for m in METHODS}
    changes = []
    inputs = []
    start = time.perf_counter()
    for k, row in enumerate(rows, 1):
        sid = row['id']
        path = source.OUT/'poses'/f'{sid}.npz'
        with np.load(path) as cache:
            xy, conf = cache['xy'], cache['conf']
            if xy.shape != (int(row['nframes']), 12, 2) or conf.shape != xy.shape[:2]:
                raise ValueError('Shape mismatch '+sid)
            if str(cache['annotation_sha256']) != row['label_sha256']:
                raise ValueError('Annotation mismatch '+sid)
        if base.sha(base.LABELS/f'{sid}.mat') != row['label_sha256']:
            raise ValueError('Annotation changed '+sid)
        variants, filled = corrections.make_variants(xy, conf)
        baseline = xy[:, source.SWAP]
        if not np.array_equal(variants['fixed_swap'][0], baseline, equal_nan=True):
            raise ValueError('Baseline mapping mismatch')
        for name in METHODS:
            pose = variants[name][0]
            # No correction may create values in an entirely absent video.
            if not np.isfinite(xy).any() and np.isfinite(pose).any():
                raise ValueError('Invented pose for absent video '+sid)
            f = base.resample(pose)
            if f.shape != (64, 24) or not np.isfinite(f).all():
                raise ValueError('Invalid features '+sid+' '+name)
            features[name].append(f)
            same = np.all((pose == baseline) | (np.isnan(pose) & np.isnan(baseline)), axis=2)
            changes.append({'id': sid, 'split': row['seed0_split'], 'method': name,
                            'changed_joint_frames': int((~same).sum()),
                            'filled_joint_frames': int(filled.sum()) if 'interpolate' in name else 0})
        inputs.append({'id': sid, 'split': row['seed0_split'], 'annotation_sha256': row['label_sha256'],
                       'pose_sha256': base.sha(path)})
        if k % 100 == 0:
            print(f'prepare {k}/{len(rows)} seconds={time.perf_counter()-start:.1f}', flush=True)
    for name in METHODS:
        np.savez_compressed(out/f'features_{name}.npz', x=np.stack(features[name]),
                            ids=np.array([r['id'] for r in rows]),
                            y=np.array([base.ACTION_CLASSES.index(r['action']) for r in rows]),
                            split=np.array([r['seed0_split'] for r in rows]))
    write_json(out/'input_manifest.json', inputs)
    write_json(out/'transformation_counts.json', changes)
    write_json(out/'prepared.json', {'videos': len(rows), 'seconds': time.perf_counter()-start,
               'feature_sha256': {m: base.sha(out/f'features_{m}.npz') for m in METHODS}})


def load_data(out, name):
    prepared = read_json(out/'prepared.json')
    path = out/f'features_{name}.npz'
    if base.sha(path) != prepared['feature_sha256'][name]:
        raise ValueError('Features changed '+name)
    with np.load(path) as a:
        data = {s: [(torch.from_numpy(x.copy()), int(y), str(sid))
                    for x, y, sid, split in zip(a['x'], a['y'], a['ids'], a['split']) if split == s]
                for s in ('train', 'validation')}
    if len(data['train']) != 1005 or len(data['validation']) != 253:
        raise ValueError('Split mismatch')
    return data


def predict(path, dataset, device):
    net = trainer.model_for('bilstm_attention').to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    x = torch.stack([v[0] for v in dataset])
    pred = []
    with torch.inference_mode():
        for batch in x.split(32):
            pred.extend(net(batch.to(device)).argmax(-1).cpu().tolist())
    y = [v[1] for v in dataset]
    return {**base.metric(y, pred), 'checkpoint_sha256': base.sha(path),
            'predictions': [{'id': v[2], 'target': v[1], 'predicted': p} for v, p in zip(dataset, pred)]}


def frozen(out):
    records = []
    for name in METHODS:
        data = load_data(out, name)['validation']
        for seed in SEEDS:
            result = predict(old_run(seed)/'best.pt', data, 'cuda' if torch.cuda.is_available() else 'cpu')
            if name == 'fixed_swap':
                old = read_json(old_run(seed)/'best_validation.json')
                if any(a[k] != b[k] for a, b in zip(result['predictions'], old['predictions'])
                       for k in ('id', 'target', 'predicted')) or len(old['predictions']) != len(data):
                    raise ValueError('Historical baseline did not reproduce')
            records.append({'method': name, 'seed': seed, **result})
            print(f'frozen {name} seed={seed} accuracy={result["accuracy_pct"]:.4f}', flush=True)
    write_json(out/'frozen_results.json', records)


def train(out):
    args = argparse.Namespace(epochs=50, batch=32)
    for name in METHODS:
        data = load_data(out, name)
        for seed in SEEDS:
            directory = out/'retrained'/name/'bilstm_attention'/f'seed{seed}'
            if directory.exists():
                # Only resume fully completed jobs, never silently accept partial runs.
                meta = read_json(directory/'provenance.json')
                if meta['epochs'] != 50 or len(read_json(directory/'history.json')) != 50:
                    raise ValueError('Incomplete training '+str(directory))
                if base.sha(directory/'best.pt') != meta['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed')
                continue
            print(f'START retrained {name} seed={seed}', flush=True)
            trainer.run('bilstm_attention', seed, data, args, out/'retrained'/name)


def paired_summary(records):
    baseline = {r['seed']: r for r in records if r['method'] == 'fixed_swap'}
    results = {}
    rng = np.random.default_rng(20261103)
    boot = rng.integers(0, 253, size=(10000, 253))
    for name in METHODS:
        group = [r for r in records if r['method'] == name]
        differences, seeds = [], []
        for r in group:
            b = baseline[r['seed']]
            if [(p['id'], p['target']) for p in r['predictions']] != [(p['id'], p['target']) for p in b['predictions']]:
                raise ValueError('Unpaired predictions')
            delta = np.array([int(p['predicted'] == p['target']) - int(q['predicted'] == q['target'])
                              for p, q in zip(r['predictions'], b['predictions'])])
            differences.append(delta)
            seeds.append({'seed': r['seed'], 'accuracy_pct': r['accuracy_pct'],
                          'macro_f1_pct': r['macro_f1_pct'], 'epoch': r.get('epoch'),
                          'improved_videos': int((delta > 0).sum()), 'worsened_videos': int((delta < 0).sum())})
        delta = np.stack(differences).mean(axis=0)
        results[name] = {'accuracy_mean_pct': float(np.mean([r['accuracy_pct'] for r in group])),
                         'accuracy_sd_pct': float(np.std([r['accuracy_pct'] for r in group], ddof=1)),
                         'macro_f1_mean_pct': float(np.mean([r['macro_f1_pct'] for r in group])),
                         'gain_vs_baseline_pp': float(delta.mean()*100),
                         'paired_video_bootstrap_95_pp': np.quantile(delta[boot].mean(axis=1)*100, [.025, .975]).tolist(),
                         'seeds': seeds}
    return results


def summarize(out):
    frozen_records = read_json(out/'frozen_results.json')
    retrained = []
    verified = []
    for name in METHODS:
        data = load_data(out, name)['validation']
        for seed in SEEDS:
            directory = out/'retrained'/name/'bilstm_attention'/f'seed{seed}'
            saved = read_json(directory/'best_validation.json')
            meta = read_json(directory/'provenance.json')
            actual = predict(directory/'best.pt', data, 'cpu')
            if actual['checkpoint_sha256'] != meta['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            if any(p[k] != q[k] for p, q in zip(actual['predictions'], saved['predictions'])
                   for k in ('id', 'target', 'predicted')):
                raise ValueError('Checkpoint reload prediction mismatch')
            retrained.append({'method': name, 'seed': seed, 'epoch': saved['epoch'], **actual})
            verified.append({'method': name, 'seed': seed, 'predictions_matched': len(data),
                             'checkpoint_sha256': actual['checkpoint_sha256']})
            print(f'verified {name} seed={seed} accuracy={actual["accuracy_pct"]:.4f}', flush=True)
    write_json(out/'retrained_results.json', retrained)
    summary = {'validation_videos': 253, 'train_videos': 1005,
               'frozen': paired_summary(frozen_records), 'retrained': paired_summary(retrained),
               'caution': 'Exploratory validation, GT bbox assistance, future frames. Bootstrap over video with seeds averaged; excludes model selection and method search bias. No official test evaluated.'}
    write_json(out/'summary.json', summary)
    write_json(out/'verification.json', verified)
    events = []
    for scenario, records in [('frozen', frozen_records), ('retrained', retrained)]:
        baseline = {r['seed']: r for r in records if r['method'] == 'fixed_swap'}
        for r in records:
            for p, b in zip(r['predictions'], baseline[r['seed']]['predictions']):
                events.append({'scenario': scenario, 'method': r['method'], 'seed': r['seed'],
                               **p, 'baseline_predicted': b['predicted'],
                               'correctness_delta': int(p['predicted'] == p['target']) - int(b['predicted'] == b['target'])})
    with (out/'per_video_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)
    print(json.dumps({s: {m: {k: v for k, v in r.items() if k != 'seeds'} for m, r in summary[s].items()}
                      for s in ('frozen', 'retrained')}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('prepare', 'frozen', 'train', 'summarize', 'all'), required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage in ('prepare', 'all'):
        prepare(args.output)
    check_protocol(args.output)
    if args.stage in ('frozen', 'all'):
        frozen(args.output)
    if args.stage in ('train', 'all'):
        train(args.output)
    if args.stage in ('summarize', 'all'):
        summarize(args.output)


if __name__ == '__main__':
    main()
