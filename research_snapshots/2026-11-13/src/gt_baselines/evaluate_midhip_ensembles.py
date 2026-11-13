"""Secondary fixed equal-weight 3-seed ensembles; CPU-verified probabilities."""
from pathlib import Path

import numpy as np
import torch

import gt_crop_midhip_refinement as study


def member_directory(method, seed):
    if method == 'baseline':
        return study.previous.old_dir('baseline', seed)
    if method == 'reference':
        return study.previous.OUT/'trained/torso_confidence_mask'/f'seed{seed}'
    return study.OUT/'trained'/method/f'seed{seed}'


def main():
    torch.set_num_threads(4)
    spec = study.previous.source.read_json(study.OUT/'ensemble_protocol.json')
    if spec['methods'] != list(study.METHODS) or spec['seeds'] != list(study.SEEDS):
        raise ValueError('Ensemble protocol mismatch')
    ensembles = {}; provenance = []
    for method in study.METHODS:
        data = study.previous.data_for(study.OUT, method)['validation']
        ids = [v[2] for v in data]; targets = np.array([v[1] for v in data])
        x = torch.stack([v[0] for v in data]); members = []; cpu_members = []
        for seed in study.SEEDS:
            directory = member_directory(method, seed)
            saved = study.previous.source.read_json(directory/'best_validation.json')
            meta = study.previous.source.read_json(directory/'provenance.json')
            if study.base.sha(directory/'best.pt') != meta['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            if ids != [p['id'] for p in saved['predictions']] or targets.tolist() != [p['target'] for p in saved['predictions']]:
                raise ValueError('Member pairing mismatch')
            model = study.previous.source.trainer.ActionBiLSTMAttentionClassifier(input_size=x.shape[2])
            model.load_state_dict(torch.load(directory/'best.pt', map_location='cpu', weights_only=True)); model.eval()
            with torch.inference_mode():
                probabilities = torch.cat([model(batch).softmax(-1) for batch in x.split(32)]).numpy()
            original = np.array([p['probabilities'] for p in saved['predictions']])
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1., atol=1e-5):
                raise ValueError('Invalid CPU probability distribution')
            if not np.array_equal(probabilities.argmax(1), np.array([p['predicted'] for p in saved['predictions']])):
                raise ValueError('Member prediction mismatch')
            cpu_members.append(probabilities)
            # GPU reproduces the original backend; CPU LSTM can differ slightly
            # in softmax values even when every predicted class is identical.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            with torch.inference_mode():
                checked = torch.cat([model(batch.to(device)).softmax(-1).cpu() for batch in x.split(32)]).numpy()
            if device == 'cuda' and not np.allclose(checked, original, atol=1e-5, rtol=1e-4):
                raise ValueError('Original-backend probability reload mismatch')
            members.append(checked)
            provenance.append({'method': method, 'seed': seed, 'checkpoint_sha256': meta['checkpoint_sha256'],
                'prediction_json_sha256': study.base.sha(directory/'best_validation.json'),
                'max_abs_probability_difference': float(np.max(np.abs(probabilities-original))),
                'original_backend': device,
                'max_abs_original_backend_probability_difference': float(np.max(np.abs(checked-original)))})
        mean = np.stack(members).mean(0); pred = mean.argmax(1)
        cpu_pred = np.stack(cpu_members).mean(0).argmax(1)
        ensembles[method] = {**study.base.metric(targets, pred),
            'cpu_ensemble_predicted_labels_match': bool(np.array_equal(cpu_pred, pred)),
            'cpu_ensemble_accuracy_pct': study.base.metric(targets, cpu_pred)['accuracy_pct'],
            'predictions': [{'id': sid, 'target': int(y), 'predicted': int(p), 'probabilities': prob.tolist()}
                            for sid, y, p, prob in zip(ids, targets, pred, mean)]}
        print(f'ensemble {method}: {ensembles[method]["accuracy_pct"]:.4f}', flush=True)
    reference = ensembles['reference']['predictions']
    rng = np.random.default_rng(20261103); boot = rng.integers(0, 253, size=(10000, 253))
    for method, result in ensembles.items():
        delta = np.array([int(p['predicted'] == p['target'])-int(q['predicted'] == q['target'])
                          for p, q in zip(result['predictions'], reference)])
        result.update({'gain_vs_reference_ensemble_pp': float(delta.mean()*100),
            'paired_video_bootstrap_95_pp': np.quantile(delta[boot].mean(1)*100, [.025, .975]).tolist(),
            'improved_videos': int((delta > 0).sum()), 'worsened_videos': int((delta < 0).sum())})
    study.previous.source.write_json(study.OUT/'ensemble_results.json', {
        'methods': ensembles, 'verification': provenance,
        'script_sha256': study.base.sha(Path(__file__)),
        'ensemble_protocol_sha256': study.base.sha(study.OUT/'ensemble_protocol.json'),
        'caution': 'Secondary exploratory validation analysis. Three models per prediction; no fitted weights; unequal cost versus one member. No independent test.'})


if __name__ == '__main__':
    main()
