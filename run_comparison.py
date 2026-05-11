"""Compare softmax baseline vs evidential model on test set.

Produces:
  - evidential_vs_softmax.png — accuracy / precision / recall / F1 by certainty decile
  - terminal-printed comparison table
"""
import math
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from deeplob_data import (
    CachedArrayView, ConcatenatedCachedArrayView, SlidingWindowDataset,
    ensure_cached_dataset, open_memmap, split_train_validation,
)

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
amp_enabled = device.type == 'cuda'
def amp_ctx():
    return torch.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()

cache_dir = Path('.cache/deeplob')
checkpoint_dir = Path('./checkpoints')
softmax_ckpt = checkpoint_dir / 'best_val_model_pytorch.pt'
evidential_ckpt = checkpoint_dir / 'best_val_model_evidential.pt'

# ---- Data (test only) ----
test_views = []
for n in ['Test_Dst_NoAuction_DecPre_CF_7.txt',
         'Test_Dst_NoAuction_DecPre_CF_8.txt',
         'Test_Dst_NoAuction_DecPre_CF_9.txt']:
    ci = ensure_cached_dataset(n, cache_dir)
    test_views.append(CachedArrayView(open_memmap(ci.cache_path)))
test_view = ConcatenatedCachedArrayView(test_views)
# We also need val for the threshold sweep
train_cache = ensure_cached_dataset('Train_Dst_NoAuction_DecPre_CF_7.txt', cache_dir)
train_array = open_memmap(train_cache.cache_path)
_, val_view = split_train_validation(train_array, train_fraction=0.8)

ds_test = SlidingWindowDataset(view=test_view, k=4, num_classes=3, T=100)
ds_val  = SlidingWindowDataset(view=val_view,  k=4, num_classes=3, T=100)
test_loader = data.DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)
val_loader  = data.DataLoader(ds_val,  batch_size=256, shuffle=False, num_workers=0)
K = ds_test.num_classes

# ---- Model architectures (identical except head activation) ----
def _backbone():
    return (
        nn.Sequential(
            nn.Conv2d(1,32,(1,2),stride=(1,2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32,32,(1,2),stride=(1,2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.Tanh(), nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32,32,(1,10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,(4,1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        ),
        nn.Sequential(
            nn.Conv2d(32,64,(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,(3,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        ),
        nn.Sequential(
            nn.Conv2d(32,64,(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,(5,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        ),
        nn.Sequential(
            nn.MaxPool2d((3,1),stride=(1,1),padding=(1,0)),
            nn.Conv2d(32,64,(1,1),padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        ),
    )

class deeplob_softmax(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        (self.conv1, self.conv2, self.conv3,
         self.inp1, self.inp2, self.inp3) = _backbone()
        self.lstm = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0,2,1,3).reshape(-1, x.shape[2], x.shape[1])
        x, _ = self.lstm(x)
        return torch.softmax(self.fc1(x[:, -1, :]), dim=1)

class deeplob_evidential(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        (self.conv1, self.conv2, self.conv3,
         self.inp1, self.inp2, self.inp3) = _backbone()
        self.lstm = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0,2,1,3).reshape(-1, x.shape[2], x.shape[1])
        x, _ = self.lstm(x)
        return F.softplus(self.fc1(x[:, -1, :]))


@torch.no_grad()
def predict_softmax(model, loader):
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        with amp_ctx():
            p = model(xb)
        ps.append(p.float().cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)

@torch.no_grad()
def predict_evidential(model, loader):
    model.eval()
    ps, us, ys = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        with amp_ctx():
            ev = model(xb)
        ev = ev.float()
        alpha = ev + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p = (alpha / S).cpu().numpy()
        u = (K / S.squeeze(1)).cpu().numpy()
        ps.append(p); us.append(u); ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(us), np.concatenate(ys)


# ---- Predict with both models ----
m_soft = deeplob_softmax(y_len=K).to(device)
m_soft.load_state_dict(torch.load(softmax_ckpt, map_location=device), strict=True)
p_soft, y_soft = predict_softmax(m_soft, test_loader)
pred_soft = p_soft.argmax(1)

m_ev = deeplob_evidential(y_len=K).to(device)
m_ev.load_state_dict(torch.load(evidential_ckpt, map_location=device), strict=True)
p_ev, u_ev, y_ev = predict_evidential(m_ev, test_loader)
pred_ev = p_ev.argmax(1); maxp_ev = p_ev.max(1)

# Also val for threshold pick
p_ev_v, u_ev_v, y_ev_v = predict_evidential(m_ev, val_loader)
pred_ev_v = p_ev_v.argmax(1); maxp_ev_v = p_ev_v.max(1); correct_v = (pred_ev_v == y_ev_v)

# Pick (beta, tau) on val: max accuracy s.t. coverage >= 10%
betas = np.linspace(0.34, 0.95, 25); taus = np.linspace(0.05, 1.00, 25)
best = (None, None, -np.inf, 0)
for b in betas:
    for t in taus:
        m = (maxp_ev_v > b) & (u_ev_v < t)
        if m.sum() < 0.10 * len(maxp_ev_v):
            continue
        acc = correct_v[m].mean()
        if acc > best[2]:
            best = (b, t, acc, m.mean())
best_b, best_t, best_acc_v, best_cov_v = best

mask_test = (maxp_ev > best_b) & (u_ev < best_t)

assert (y_soft == y_ev).all(), 'test labels diverged between runs'
y_test = y_ev

# ---- Metrics ----
def m(label, y_true, y_pred, n_total=None):
    cov = (len(y_true) / n_total) if n_total else 1.0
    return {
        'label': label,
        'coverage': cov,
        'n': len(y_true),
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'rec':  recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1':   f1_score(y_true, y_pred, average='macro', zero_division=0),
        'prec_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
    }

n_total = len(y_test)
rows = [
    m('Softmax (baseline)',           y_test, pred_soft, n_total),
    m('Evidential (all)',             y_test, pred_ev,   n_total),
    m(f'Evidential (β={best_b:.2f}, τ={best_t:.2f})',
      y_test[mask_test], pred_ev[mask_test], n_total),
]

print('\n=== Head-to-head on test set ===')
print(f'{"model":<40s} {"cov":>6s} {"n":>9s} {"acc":>6s} {"prec":>6s} {"rec":>6s} {"f1":>6s}')
for r in rows:
    print(f'{r["label"]:<40s} {r["coverage"]*100:5.1f}% {r["n"]:>9,d} '
          f'{r["acc"]:.4f} {r["prec"]:.4f} {r["rec"]:.4f} {r["f1"]:.4f}')
print()
print('Per-class precision:')
for r in rows:
    print(f'  {r["label"]:<40s} {np.round(r["prec_per_class"], 4)}')

# ---- Certainty-bucket analysis ----
certainty = 1.0 - u_ev  # higher = more certain
# Use 10 quantile-based buckets so each contains ~equal mass
edges = np.quantile(certainty, np.linspace(0, 1, 11))
edges[0] -= 1e-9; edges[-1] += 1e-9
bins = np.digitize(certainty, edges, right=False) - 1
bins = np.clip(bins, 0, 9)

per_bin = []
for b in range(10):
    sel = (bins == b)
    if sel.sum() == 0:
        per_bin.append(dict(lo=np.nan, hi=np.nan, n=0, acc=np.nan, prec=np.nan, rec=np.nan, f1=np.nan))
        continue
    per_bin.append(dict(
        lo=edges[b], hi=edges[b+1], n=int(sel.sum()),
        acc=accuracy_score(y_test[sel], pred_ev[sel]),
        prec=precision_score(y_test[sel], pred_ev[sel], average='macro', zero_division=0),
        rec=recall_score(y_test[sel], pred_ev[sel], average='macro', zero_division=0),
        f1=f1_score(y_test[sel], pred_ev[sel], average='macro', zero_division=0),
    ))

print('\n=== Test accuracy by certainty decile (1 - u, low → high) ===')
print(f'{"bin":>3s} {"cert range":<22s} {"n":>8s} {"acc":>6s} {"prec":>6s} {"rec":>6s} {"f1":>6s}')
for i, r in enumerate(per_bin):
    rng = f'[{r["lo"]:.3f}, {r["hi"]:.3f}]'
    print(f'{i:>3d} {rng:<22s} {r["n"]:>8,d} {r["acc"]:.4f} {r["prec"]:.4f} {r["rec"]:.4f} {r["f1"]:.4f}')

# ---- Plots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Accuracy / precision / recall / f1 by certainty decile
ax = axes[0, 0]
xs = np.arange(10)
ax.plot(xs, [r['acc']  for r in per_bin], 'o-', label='accuracy',  lw=2)
ax.plot(xs, [r['prec'] for r in per_bin], 's-', label='precision (macro)', lw=2)
ax.plot(xs, [r['rec']  for r in per_bin], '^-', label='recall (macro)',    lw=2)
ax.plot(xs, [r['f1']   for r in per_bin], 'd-', label='F1 (macro)',        lw=2)
ax.set_xticks(xs)
ax.set_xticklabels([f'{i+1}' for i in xs])
ax.set_xlabel('certainty decile (1 = least certain, 10 = most certain)')
ax.set_ylabel('score')
ax.set_title('Test metrics by certainty bucket (evidential)')
ax.grid(alpha=0.3); ax.legend()

# (2) Bar chart per certainty decile — accuracy with sample count overlay
ax = axes[0, 1]
ax.bar(xs, [r['acc'] for r in per_bin], color='#3b82f6', alpha=0.85)
for i, r in enumerate(per_bin):
    ax.text(i, r['acc'] + 0.01, f'{r["acc"]:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([f'{i+1}' for i in xs])
ax.set_xlabel('certainty decile')
ax.set_ylabel('accuracy on bucket')
ax.set_title('Accuracy per certainty decile')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3, axis='y')
ax2 = ax.twinx()
ax2.plot(xs, [r['n'] for r in per_bin], color='gray', ls=':', marker='x', label='# samples')
ax2.set_ylabel('# samples (dotted)')
ax2.legend(loc='lower right')

# (3) Head-to-head bar chart
ax = axes[1, 0]
metric_names = ['acc', 'prec', 'rec', 'f1']
labels_pretty = ['Softmax\n(baseline)', 'Evidential\n(all)',
                 f'Evidential\nfiltered\nβ={best_b:.2f}, τ={best_t:.2f}']
width = 0.22
xpos = np.arange(len(metric_names))
colors = ['#94a3b8', '#3b82f6', '#16a34a']
for i, r in enumerate(rows):
    vals = [r['acc'], r['prec'], r['rec'], r['f1']]
    ax.bar(xpos + (i - 1) * width, vals, width, label=labels_pretty[i], color=colors[i])
    for j, v in enumerate(vals):
        ax.text(xpos[j] + (i - 1) * width, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(['accuracy', 'macro prec', 'macro rec', 'macro F1'])
ax.set_ylim(0, 1.0); ax.set_ylabel('score')
ax.set_title('Softmax vs Evidential on test set')
ax.grid(alpha=0.3, axis='y'); ax.legend(loc='lower right')

# (4) Cumulative accuracy as certainty threshold sweeps (continuous version of buckets)
ax = axes[1, 1]
order = np.argsort(-certainty)  # most certain first
cum_correct = np.cumsum((pred_ev[order] == y_test[order]).astype(float))
n_seq = np.arange(1, len(y_test) + 1)
cum_acc = cum_correct / n_seq
coverage = n_seq / len(y_test)
ax.plot(coverage, cum_acc, lw=2, color='#7c3aed')
ax.axhline(rows[0]['acc'], ls='--', color='#94a3b8', label=f'softmax baseline = {rows[0]["acc"]:.3f}')
ax.axhline(rows[1]['acc'], ls='--', color='#3b82f6', label=f'evidential (all) = {rows[1]["acc"]:.3f}')
ax.scatter([mask_test.mean()], [rows[2]['acc']], c='#16a34a', s=80, zorder=5,
           label=f'chosen β/τ → acc {rows[2]["acc"]:.3f} @ {mask_test.mean()*100:.1f}% cov')
ax.set_xlabel('coverage (fraction kept, most certain first)')
ax.set_ylabel('cumulative accuracy on kept')
ax.set_title('Accuracy vs coverage (sorted by certainty)')
ax.grid(alpha=0.3); ax.legend(loc='lower left')

plt.tight_layout()
fig.savefig('evidential_vs_softmax.png', dpi=120, bbox_inches='tight')
print('\nSaved: evidential_vs_softmax.png')
