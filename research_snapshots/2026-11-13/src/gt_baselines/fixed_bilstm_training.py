"""Fixed 50-epoch BiLSTM budget shared by new controlled ablations."""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import gt_crop_correction_experiment as base_study


def run(out, method, seed, data, input_size):
    directory = out/'trained'/method/f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = base_study.trainer.ActionBiLSTMAttentionClassifier(input_size=input_size).to(device)
    tr = DataLoader(data['train'], batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    va = DataLoader(data['validation'], batch_size=32, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=.00001)
    criterion = nn.CrossEntropyLoss(); best = -1.; history = []; start = time.perf_counter()
    for epoch in range(1, 51):
        model.train(); loss_sum = 0.
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); z = model(x); loss = criterion(z, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            loss_sum += loss.item()*len(y)
        sched.step(); model.eval(); pred = []; target = []; ids = []; probabilities = []
        with torch.inference_mode():
            for x, y, sid in va:
                z = model(x.to(device)); pred.extend(z.argmax(-1).cpu().tolist())
                target.extend(y.tolist()); ids.extend(sid); probabilities.extend(z.softmax(-1).cpu().tolist())
        metrics = base_study.base.metric(target, pred)
        history.append({'epoch': epoch, 'train_loss': loss_sum/len(data['train']), **metrics})
        if metrics['accuracy_pct'] > best:
            best = metrics['accuracy_pct']; torch.save(model.state_dict(), directory/'best.pt')
            base_study.write_json(directory/'best_validation.json', {'method': method, 'seed': seed, 'epoch': epoch,
                **metrics, 'predictions': [{'id': sid, 'target': y, 'predicted': p, 'probabilities': prob}
                                         for sid, y, p, prob in zip(ids, target, pred, probabilities)]})
        base_study.write_json(directory/'history.json', history)
        print(f'{method} seed={seed} epoch={epoch}/50 val={metrics["accuracy_pct"]:.2f} best={best:.2f} seconds={time.perf_counter()-start:.1f}', flush=True)
    base_study.write_json(directory/'provenance.json', {'method': method, 'seed': seed, 'epochs': 50,
        'parameters': sum(p.numel() for p in model.parameters()), 'input_size': input_size,
        'checkpoint_sha256': base_study.base.sha(directory/'best.pt'),
        'protocol_sha256': base_study.base.sha(out/'protocol.json'),
        'feature_sha256': base_study.base.sha(out/f'features_{method}.npz'), 'seconds': time.perf_counter()-start,
        'train_ids': [r[2] for r in data['train']], 'validation_ids': [r[2] for r in data['validation']]})
