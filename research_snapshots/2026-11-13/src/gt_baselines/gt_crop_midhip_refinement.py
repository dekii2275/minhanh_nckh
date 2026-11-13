"""Bounded midhip normalization ablations against the 94.86% reference."""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch

import fixed_bilstm_training as training
import gt_crop_confidence_experiment as previous
import midhip_refinement_features as features

base = previous.base
ROOT = previous.ROOT
OUT = ROOT/'outputs/gt_crop_midhip_refinement_20261103'
METHODS = ('baseline', 'reference', *features.METHODS)
SEEDS = (0, 1, 2)


def protocol():
    return {'methods': METHODS, 'new_methods': features.METHODS, 'seeds': SEEDS,
        'scope': '1005 train / 253 validation, fixed seed0 split, 15 classes, no official test',
        'training': '50 epochs, batch32, Adam lr=.001 wd=.0001, cosine eta_min=.00001, clip norm1; earliest highest validation accuracy',
        'input_size': 'all new methods/reference 48, baseline24; same BiLSTM+attention',
        'strict_midhip_shoulder': 'paired reliable hips+shoulders each native frame, shoulder width>5px; zero invalid frames; coordinates clipped +/-5; zero confidence/masks for invalid geometry',
        'median_shoulder': 'same reliable/interpolated root as 94.86% reference, median reliable shoulder width>5px over clip; fallback reference torso scale; original confidence/mask unchanged',
        'smooth_midhip': 'same reference torso scale and confidence/mask; centered +/-2-frame mean of root only',
        'interpolate3': 'bounded 1-3 native-frame gaps using predicted xy/conf; recompute root/torso scale on imputed pose; append original confidence/mask so imputed joints are not marked observed',
        'mapping': 'same fixed swap; predicted pose only; GT bbox crop/selection retained',
        'paper': {'title': 'A BiLSTM with Attention Approach Using 2D Skeletons for Autism Activity Recognition',
                  'doi': '10.1109/ICCA66035.2025.11431081',
                  'status': 'identified in author/conference sources; original full text not accessed; own presets, not exact reproduction',
                  'primary_related_method': 'https://www.mdpi.com/1424-8220/26/14/4638'},
        'confidence_threshold': .30, 'strict_clip': features.CLIP,
        'script_sha256': base.sha(Path(__file__)), 'feature_script_sha256': base.sha(Path(features.__file__)),
        'training_script_sha256': base.sha(Path(training.__file__)),
        'source_protocol_sha256': base.sha(previous.OUT/'protocol.json'),
        'source_input_manifest_sha256': base.sha(previous.source.DEFAULT_OUT/'input_manifest.json'),
        'source_reference_features_sha256': base.sha(previous.OUT/'features_torso_confidence_mask.npz'),
        'source_baseline_features_sha256': base.sha(previous.OUT/'features_baseline.npz'),
        'classifier_sha256': base.sha(ROOT/'src/models/action_bilstm.py'),
        'source_checkpoint_sha256': {str(seed): base.sha(previous.OUT/'trained/torso_confidence_mask'/f'seed{seed}'/'best.pt') for seed in SEEDS},
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'limitations': 'exploratory reused validation, GT-assisted target, future frames, project-defined missing rules; no exact paper replication or online claim'}


def check(out):
    if previous.source.read_json(out/'protocol.json') != json.loads(json.dumps(protocol())):
        raise ValueError('Protocol/input changed')


def prepare(out):
    previous.check(previous.OUT)
    out.mkdir(parents=True, exist_ok=False)
    previous.source.write_json(out/'protocol.json', protocol())
    rows = previous.source.read_json(previous.source.DEFAULT_OUT/'input_manifest.json')
    with np.load(previous.OUT/'features_torso_confidence_mask.npz') as a:
        reference = a['x']; ids, labels, splits = a['ids'], a['y'], a['split']
    with np.load(previous.OUT/'features_baseline.npz') as a:
        baseline = a['x']
        if not np.array_equal(ids, a['ids']):
            raise ValueError('Baseline order mismatch')
    if ids.tolist() != [r['id'] for r in rows]:
        raise ValueError('Manifest order mismatch')
    accum = {m: [] for m in features.METHODS}; diagnostics = []
    for i, row in enumerate(rows):
        path = previous.source.source.OUT/'poses'/f'{row["id"]}.npz'
        if base.sha(path) != row['pose_sha256']:
            raise ValueError('Input cache changed')
        with np.load(path) as a:
            if str(a['annotation_sha256']) != row['annotation_sha256']:
                raise ValueError('Annotation mismatch')
            xy = a['xy'][:, previous.source.source.SWAP]
            conf = a['conf'][:, previous.source.source.SWAP]
        variants, diag = features.variants(xy, conf)
        if not np.array_equal(variants['reference'], reference[i]):
            raise ValueError('94.86% preprocessing did not reproduce '+row['id'])
        for m in accum:
            accum[m].append(variants[m])
        diagnostics.append({'id': row['id'], 'split': row['split'], **diag})
        if (i+1) % 250 == 0:
            print(f'prepared {i+1}/{len(rows)}', flush=True)
    arrays = {'baseline': baseline, 'reference': reference, **{m: np.stack(x) for m, x in accum.items()}}
    hashes = {}
    for m, x in arrays.items():
        path = out/f'features_{m}.npz'
        np.savez_compressed(path, x=x, ids=ids, y=labels, split=splits)
        hashes[m] = base.sha(path)
    previous.source.write_json(out/'prepared.json', {'feature_sha256': hashes})
    previous.source.write_json(out/'transformation_diagnostics.json', diagnostics)


def predict(out, method, seed, dataset):
    if method in ('baseline', 'reference'):
        old = 'baseline' if method == 'baseline' else 'torso_confidence_mask'
        result = previous.predict(previous.OUT, old, seed, dataset)
        return {**result, 'method': method}
    directory = out/'trained'/method/f'seed{seed}'
    meta = previous.source.read_json(directory/'provenance.json'); checkpoint = directory/'best.pt'
    if base.sha(checkpoint) != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    model = previous.source.trainer.ActionBiLSTMAttentionClassifier(input_size=48)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)); model.eval()
    pred = []; x = torch.stack([v[0] for v in dataset])
    with torch.inference_mode():
        for batch in x.split(32):
            pred.extend(model(batch).argmax(-1).tolist())
    predictions = [{'id': v[2], 'target': v[1], 'predicted': p} for v, p in zip(dataset, pred)]
    saved = previous.source.read_json(directory/'best_validation.json')
    if len(predictions) != len(saved['predictions']) or any(p[k] != q[k] for p, q in zip(predictions, saved['predictions']) for k in ('id', 'target', 'predicted')):
        raise ValueError('CPU checkpoint reload mismatch')
    return {'method': method, 'seed': seed, 'epoch': saved['epoch'], 'predictions': predictions,
            'parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_sha256': base.sha(checkpoint),
            **base.metric([v[1] for v in dataset], pred)}


def summarize(out):
    records = []
    for method in METHODS:
        data = previous.data_for(out, method)['validation']
        for seed in SEEDS:
            result = predict(out, method, seed, data); records.append(result)
            print(f'verified {method} seed={seed} accuracy={result["accuracy_pct"]:.4f}', flush=True)
    groups = {m: [r for r in records if r['method'] == m] for m in METHODS}
    compare = previous.comparisons.comparison
    summary = {'methods_vs_reference': {m: compare(r, groups['reference']) for m, r in groups.items()},
               'methods_vs_baseline': {m: compare(r, groups['baseline']) for m, r in groups.items()},
               'train_videos': 1005, 'validation_videos': 253,
               'caution': 'Exploratory reused validation; GT-assisted target; video bootstrap excludes model selection and seed uncertainty; unadjusted intervals. No independent test.'}
    previous.source.write_json(out/'results.json', records); previous.source.write_json(out/'summary.json', summary)
    previous.source.write_json(out/'verification.json', {'runs': len(records), 'new_checkpoints': 12,
        'cpu_prediction_matches': sum(len(r['predictions']) for r in records), 'reference_preprocessing_matches': 1258})
    rows = []
    for r in records:
        b = groups['reference'][r['seed']]
        for p, q in zip(r['predictions'], b['predictions']):
            rows.append({'method': r['method'], 'seed': r['seed'], **p, 'reference_predicted': q['predicted'],
                         'correctness_delta': int(p['predicted'] == p['target'])-int(q['predicted'] == q['target'])})
    with (out/'per_video_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({m: {k: v for k, v in r.items() if k != 'seeds'} for m, r in summary['methods_vs_reference'].items()}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--stage', choices=('prepare', 'train', 'summarize', 'all'), required=True)
    args = p.parse_args(); torch.set_num_threads(4)
    if args.stage in ('prepare', 'all'):
        prepare(args.output)
    check(args.output)
    if args.stage in ('train', 'all'):
        for method in features.METHODS:
            data = previous.data_for(args.output, method)
            for seed in SEEDS:
                directory = args.output/'trained'/method/f'seed{seed}'
                if directory.exists():
                    meta = previous.source.read_json(directory/'provenance.json')
                    if meta['protocol_sha256'] != base.sha(args.output/'protocol.json') or len(previous.source.read_json(directory/'history.json')) != 50:
                        raise ValueError('Partial/changed old training run')
                    continue
                training.run(args.output, method, seed, data, 48)
    if args.stage in ('summarize', 'all'):
        summarize(args.output)


if __name__ == '__main__':
    main()
