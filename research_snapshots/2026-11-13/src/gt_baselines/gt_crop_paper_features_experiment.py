"""Controlled Ma et al. kinematic-feature adaptation on YOLO-crop BiLSTM.

Uses the existing train/validation cache only. It is not a paper reproduction:
retain 15 classes, 12 joints, project's normalizer, BiLSTM+attention and budget.
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import gt_crop_correction_experiment as previous
import paper_kinematic_features as kinematics

base = previous.base
ROOT = previous.ROOT
OUT = ROOT/'outputs/gt_crop_paper_features_20261103'
PAPER = 'https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1925220/full'
METHODS = {'coordinates': ('fixed_swap', 'coordinates'), 'angles': ('fixed_swap', 'angles'),
           'velocity': ('fixed_swap', 'velocity'), 'full': ('fixed_swap', 'full'),
           'capacity_control': ('fixed_swap', 'capacity_control'),
           'mean5_coordinates': ('mean5', 'coordinates'), 'mean5_full': ('mean5', 'full')}
SEEDS = (0, 1, 2)


def old_dir(method, seed):
    kind = METHODS[method][0]
    return previous.DEFAULT_OUT/'retrained'/kind/'bilstm_attention'/f'seed{seed}'


def protocol():
    return {'paper': PAPER, 'paper_sections': '3.6, 3.8, Table 7',
            'adaptation': 'kinematic component feature concatenation; not coordinate correction or aggregate profiles',
            'methods': METHODS, 'seeds': list(SEEDS), 'epochs': 50, 'batch': 32,
            'optimizer': 'Adam lr=.001 weight_decay=.0001', 'clip_norm': 1.,
            'scheduler': 'cosine 50 epochs eta_min=.00001',
            'selection': 'highest validation accuracy; earliest tie; same as existing baseline',
            'scope': '1005 train / 253 validation, 15 classes, no official test',
            'coordinates': 'unchanged baseline: hip centering, shoulder scaling with fallback, resampled 64 steps',
            'angles': {'triplets': kinematics.ANGLE_TRIPLETS, 'names': kinematics.ANGLE_NAMES,
                       'units': 'radians/pi; invalid or degenerate triplet -> 0'},
            'velocity': 'first difference of normalized coordinates per resampled step; first/invalid transitions -> 0',
            'com': 'mean normalized shoulders and hips (4 torso joints), xy and scalar speed; invalid -> 0',
            'missing': 'coordinate channels unchanged; kinematic channels mask invalid contributing native-frame joints',
            'capacity_control': '24 coordinate channels plus 35 zeros; same architecture/parameters/initialization shape as full',
            'source_features_sha256': {m: base.sha(previous.DEFAULT_OUT/f'features_{m}.npz') for m in ('fixed_swap', 'mean5')},
            'source_input_manifest_sha256': base.sha(previous.DEFAULT_OUT/'input_manifest.json'),
            'manifest_sha256': base.sha(base.MANIFEST),
            'script_sha256': base.sha(Path(__file__)), 'features_script_sha256': base.sha(Path(kinematics.__file__)),
            'classifier_sha256': base.sha(ROOT/'src/models/action_bilstm.py'),
            'source_protocol_sha256': base.sha(previous.DEFAULT_OUT/'protocol.json'),
            'baseline_checkpoints': {m: {str(s): base.sha(old_dir(m, s)/'best.pt') for s in SEEDS}
                                     for m in ('coordinates', 'mean5_coordinates')},
            'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'limitations': 'independent implementation; different model/split/joints/normalization from paper; GT bbox assisted; validation reused; no expert quality labels'}


def check_protocol(out):
    # Normalize tuples through JSON to compare with persisted arrays.
    expected = json.loads(json.dumps(protocol()))
    if previous.read_json(out/'protocol.json') != expected:
        raise ValueError('Protocol or input changed')


def prepare(out):
    previous.check_protocol(previous.DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=False)
    previous.write_json(out/'protocol.json', protocol())
    masks = []
    inputs = previous.read_json(previous.DEFAULT_OUT/'input_manifest.json')
    for row in inputs:
        path = previous.source.OUT/'poses'/f'{row["id"]}.npz'
        if base.sha(path) != row['pose_sha256']:
            raise ValueError('Pose source changed '+row['id'])
        with np.load(path) as a:
            if str(a['annotation_sha256']) != row['annotation_sha256']:
                raise ValueError('Annotation mismatch')
            masks.append(kinematics.resampled_validity(a['xy'][:, previous.source.SWAP]))
    hashes = {}
    for method, (source_mode, feature_mode) in METHODS.items():
        with np.load(previous.DEFAULT_OUT/f'features_{source_mode}.npz') as a:
            if a['ids'].tolist() != [r['id'] for r in inputs]:
                raise ValueError('Order mismatch')
            x = np.stack([kinematics.enrich(p, mask, feature_mode) for p, mask in zip(a['x'], masks)])
            if not np.array_equal(x[:, :, :24], a['x']):
                raise ValueError('Changed coordinate channels')
            path = out/f'features_{method}.npz'
            np.savez_compressed(path, x=x, y=a['y'], ids=a['ids'], split=a['split'])
        hashes[method] = base.sha(path)
        print(f'prepared {method} {x.shape}', flush=True)
    previous.write_json(out/'prepared.json', {'feature_sha256': hashes, 'videos': len(inputs)})


def data_for(out, method):
    path = out/f'features_{method}.npz'
    if base.sha(path) != previous.read_json(out/'prepared.json')['feature_sha256'][method]:
        raise ValueError('Features changed')
    with np.load(path) as a:
        result = {s: [(torch.from_numpy(x.copy()), int(y), str(sid))
                      for x, y, sid, split in zip(a['x'], a['y'], a['ids'], a['split']) if split == s]
                  for s in ('train', 'validation')}
    if len(result['train']) != 1005 or len(result['validation']) != 253:
        raise ValueError('Split mismatch')
    return result


def model_for(method):
    return previous.trainer.ActionBiLSTMAttentionClassifier(input_size=kinematics.DIMENSIONS[METHODS[method][1]])


def train_run(out, method, seed, data):
    directory = out/'trained'/method/f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_for(method).to(device)
    tr = DataLoader(data['train'], batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    va = DataLoader(data['validation'], batch_size=32, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=.00001)
    loss_function = nn.CrossEntropyLoss()
    best = -1.; history = []; start = time.perf_counter()
    for epoch in range(1, 51):
        model.train(); loss_sum = 0.
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); logits = model(x); loss = loss_function(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            loss_sum += loss.item()*len(y)
        sched.step(); model.eval(); predictions = []; target = []; ids = []; probs = []
        with torch.inference_mode():
            for x, y, sid in va:
                logits = model(x.to(device))
                predictions.extend(logits.argmax(-1).cpu().tolist()); target.extend(y.tolist()); ids.extend(sid)
                probs.extend(logits.softmax(-1).cpu().tolist())
        metrics = base.metric(target, predictions)
        accuracy = metrics['accuracy_pct']
        history.append({'epoch': epoch, 'train_loss': loss_sum/1005, **metrics})
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), directory/'best.pt')
            previous.write_json(directory/'best_validation.json', {'method': method, 'seed': seed,
                'epoch': epoch, **metrics, 'predictions': [{'id': sid, 'target': y, 'predicted': pred, 'probabilities': prob}
                    for sid, y, pred, prob in zip(ids, target, predictions, probs)]})
        previous.write_json(directory/'history.json', history)
        print(f'{method} seed={seed} epoch={epoch}/50 val={accuracy:.2f} best={best:.2f} seconds={time.perf_counter()-start:.1f}', flush=True)
    previous.write_json(directory/'provenance.json', {'method': method, 'seed': seed, 'epochs': 50,
        'parameters': sum(p.numel() for p in model.parameters()), 'input_size': kinematics.DIMENSIONS[METHODS[method][1]],
        'checkpoint_sha256': base.sha(directory/'best.pt'), 'protocol_sha256': base.sha(out/'protocol.json'),
        'features_sha256': base.sha(out/f'features_{method}.npz'), 'seconds': time.perf_counter()-start,
        'train_ids': [r[2] for r in data['train']], 'validation_ids': [r[2] for r in data['validation']]})


def predict(out, method, seed, dataset):
    baseline = method in ('coordinates', 'mean5_coordinates')
    directory = old_dir(method, seed) if baseline else out/'trained'/method/f'seed{seed}'
    meta = previous.read_json(directory/'provenance.json')
    checkpoint = directory/'best.pt'
    if base.sha(checkpoint) != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    model = model_for(method)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)); model.eval()
    x = torch.stack([v[0] for v in dataset]); pred = []
    with torch.inference_mode():
        for batch in x.split(32):
            pred.extend(model(batch).argmax(-1).tolist())
    saved = previous.read_json(directory/'best_validation.json')
    actual = [{'id': v[2], 'target': v[1], 'predicted': p} for v, p in zip(dataset, pred)]
    if len(actual) != len(saved['predictions']) or any(p[k] != q[k] for p, q in zip(actual, saved['predictions'])
                                                     for k in ('id', 'target', 'predicted')):
        raise ValueError('CPU reload mismatch')
    return {'method': method, 'seed': seed, 'epoch': saved['epoch'],
            'parameters': sum(p.numel() for p in model.parameters()),
            **base.metric([v[1] for v in dataset], pred), 'predictions': actual,
            'checkpoint_sha256': base.sha(checkpoint), 'reused_baseline': baseline}


def comparison(group, reference):
    rng = np.random.default_rng(20261103)
    differences, seeds = [], []
    for r, b in zip(group, reference):
        if r['seed'] != b['seed'] or [(p['id'], p['target']) for p in r['predictions']] != [(p['id'], p['target']) for p in b['predictions']]:
            raise ValueError('Pair mismatch')
        delta = np.array([int(p['predicted'] == p['target'])-int(q['predicted'] == q['target'])
                          for p, q in zip(r['predictions'], b['predictions'])])
        differences.append(delta)
        seeds.append({'seed': r['seed'], 'accuracy_pct': r['accuracy_pct'], 'macro_f1_pct': r['macro_f1_pct'],
                      'epoch': r['epoch'], 'improved_videos': int((delta > 0).sum()), 'worsened_videos': int((delta < 0).sum())})
    mean = np.stack(differences).mean(0)
    boot = rng.integers(0, len(mean), size=(10000, len(mean)))
    return {'accuracy_mean_pct': float(np.mean([r['accuracy_pct'] for r in group])),
            'accuracy_sd_pct': float(np.std([r['accuracy_pct'] for r in group], ddof=1)),
            'macro_f1_mean_pct': float(np.mean([r['macro_f1_pct'] for r in group])),
            'parameters': group[0]['parameters'],
            'gain_pp': float(100*mean.mean()),
            'paired_video_bootstrap_95_pp': np.quantile(mean[boot].mean(1)*100, [.025, .975]).tolist(),
            'seeds': seeds}


def summarize(out):
    records = []
    for method in METHODS:
        data = data_for(out, method)['validation']
        for seed in SEEDS:
            records.append(predict(out, method, seed, data))
            print(f'verified {method} seed={seed} accuracy={records[-1]["accuracy_pct"]:.4f}', flush=True)
    groups = {m: [r for r in records if r['method'] == m] for m in METHODS}
    summary = {'methods_vs_coordinates': {m: comparison(group, groups['coordinates']) for m, group in groups.items()},
               'full_vs_capacity_control': comparison(groups['full'], groups['capacity_control']),
               'mean5_full_vs_mean5_coordinates': comparison(groups['mean5_full'], groups['mean5_coordinates']),
               'mean5_full_vs_raw_full': comparison(groups['mean5_full'], groups['full']),
               'validation_videos': 253, 'train_videos': 1005,
               'caution': 'Exploratory validation, GT bbox assistance. Three seeds same split. Video bootstrap excludes checkpoint/method selection and seed uncertainty; intervals unadjusted for multiple comparisons.'}
    previous.write_json(out/'results.json', records)
    previous.write_json(out/'summary.json', summary)
    previous.write_json(out/'verification.json', {'runs': len(records), 'new_checkpoints': 15,
                        'cpu_prediction_matches': sum(len(r['predictions']) for r in records)})
    events = []
    for r in records:
        b = groups['coordinates'][r['seed']]
        for p, q in zip(r['predictions'], b['predictions']):
            events.append({'method': r['method'], 'seed': r['seed'], **p, 'baseline_predicted': q['predicted'],
                           'correctness_delta': int(p['predicted'] == p['target'])-int(q['predicted'] == q['target'])})
    with (out/'per_video_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)
    print(json.dumps({m: {k: v for k, v in r.items() if k != 'seeds'} for m, r in summary['methods_vs_coordinates'].items()}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--stage', choices=('prepare', 'train', 'summarize', 'all'), required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage in ('prepare', 'all'):
        prepare(args.output)
    check_protocol(args.output)
    if args.stage in ('train', 'all'):
        for method in METHODS:
            if method in ('coordinates', 'mean5_coordinates'):
                continue
            data = data_for(args.output, method)
            for seed in SEEDS:
                directory = args.output/'trained'/method/f'seed{seed}'
                if directory.exists():
                    meta = previous.read_json(directory/'provenance.json')
                    if meta['protocol_sha256'] != base.sha(args.output/'protocol.json') or len(previous.read_json(directory/'history.json')) != 50:
                        raise ValueError('Partial/changed previous run')
                    continue
                train_run(args.output, method, seed, data)
    if args.stage in ('summarize', 'all'):
        summarize(args.output)


if __name__ == '__main__':
    main()
