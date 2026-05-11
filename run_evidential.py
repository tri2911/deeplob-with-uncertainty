"""Standalone runner mirroring the notebook's evidential cells.

Loads the existing LSTM softmax checkpoint, fine-tunes a Dirichlet/evidential
head, then evaluates uncertainty behavior on the test set and saves plots.
"""
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from deeplob_data import (
    CachedArrayView,
    ConcatenatedCachedArrayView,
    SlidingWindowDataset,
    ensure_cached_dataset,
    open_memmap,
    split_train_validation,
)

# ---- Setup ----
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
amp_enabled = device.type == 'cuda'
batch_size = 256
cache_dir = Path('.cache/deeplob')
checkpoint_dir = Path('./checkpoints')
best_model_path = checkpoint_dir / 'best_val_model_pytorch.pt'
evidential_ckpt_path = checkpoint_dir / 'best_val_model_evidential.pt'
pin_memory = False

if amp_enabled:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def get_autocast_context():
    return torch.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()

if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
else:
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

# ---- Data ----
train_cache = ensure_cached_dataset('Train_Dst_NoAuction_DecPre_CF_7.txt', cache_dir)
train_array = open_memmap(train_cache.cache_path)
train_view, val_view = split_train_validation(train_array, train_fraction=0.8)

test_views = []
for n in ['Test_Dst_NoAuction_DecPre_CF_7.txt',
         'Test_Dst_NoAuction_DecPre_CF_8.txt',
         'Test_Dst_NoAuction_DecPre_CF_9.txt']:
    ci = ensure_cached_dataset(n, cache_dir)
    test_views.append(CachedArrayView(open_memmap(ci.cache_path)))
test_view = ConcatenatedCachedArrayView(test_views)

dataset_train = SlidingWindowDataset(view=train_view, k=4, num_classes=3, T=100)
dataset_val   = SlidingWindowDataset(view=val_view,   k=4, num_classes=3, T=100)
dataset_test  = SlidingWindowDataset(view=test_view,  k=4, num_classes=3, T=100)

train_loader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, num_workers=0)

K = dataset_train.num_classes

# ---- Model ----
class deeplob_evidential(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3).reshape(-1, x.shape[2], x.shape[1])
        x, _ = self.lstm(x)
        return F.softplus(self.fc1(x[:, -1, :]))


def evidential_outputs(evidence):
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    u = K / S.squeeze(1)
    return alpha, S.squeeze(1), p, u


def kl_dirichlet_uniform(alpha):
    K_ = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    term1 = torch.lgamma(S.squeeze(1)) - math.lgamma(K_) - torch.lgamma(alpha).sum(dim=1)
    term2 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return term1 + term2


def evidential_mse_loss(evidence, target_onehot, kl_weight):
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = (target_onehot - p).pow(2).sum(dim=1)
    var = (p * (1.0 - p) / (S + 1.0)).sum(dim=1)
    mse = err + var
    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    kl = kl_dirichlet_uniform(alpha_tilde)
    return (mse + kl_weight * kl).mean(), mse.mean().item(), kl.mean().item()


# ---- Fine-tune ----
model = deeplob_evidential(y_len=K).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device), strict=True)
print('Loaded softmax checkpoint as warm start')

epochs = 15
annealing_epochs = 10
lr = 1e-4
opt = optim.Adam(model.parameters(), lr=lr)
best_val_loss = float('inf')
for ep in range(epochs):
    model.train()
    kl_weight = min(1.0, (ep + 1) / max(1, annealing_epochs))
    ep_loss = []; ep_mse = []; ep_kl = []
    t0 = datetime.now()
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=pin_memory)
        targets = targets.to(device, non_blocking=pin_memory)
        target_onehot = F.one_hot(targets, num_classes=K).float()
        opt.zero_grad(set_to_none=True)
        with get_autocast_context():
            evidence = model(inputs)
            loss, m, k = evidential_mse_loss(evidence, target_onehot, kl_weight)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ep_loss.append(loss.item()); ep_mse.append(m); ep_kl.append(k)
    # val
    model.eval()
    v_loss = []; n_correct = n_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device); targets = targets.to(device)
            target_onehot = F.one_hot(targets, num_classes=K).float()
            with get_autocast_context():
                evidence = model(inputs)
                loss, _, _ = evidential_mse_loss(evidence, target_onehot, kl_weight)
            v_loss.append(loss.item())
            _, _, p, _ = evidential_outputs(evidence.float())
            n_correct += (p.argmax(1) == targets).sum().item()
            n_total += targets.size(0)
    mean_v = float(np.mean(v_loss))
    saved = ''
    if mean_v < best_val_loss:
        best_val_loss = mean_v
        torch.save(model.state_dict(), evidential_ckpt_path)
        saved = ' [saved]'
    print(f'Epoch {ep+1}/{epochs} | kl_w={kl_weight:.2f} | '
          f'train {np.mean(ep_loss):.4f} (mse {np.mean(ep_mse):.4f} kl {np.mean(ep_kl):.4f}) | '
          f'val {mean_v:.4f} | val_acc {n_correct/max(1,n_total):.4f} | {datetime.now()-t0}{saved}')

# ---- Evaluate ----
@torch.no_grad()
def collect(model, loader):
    model.eval()
    ps, us, ys = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        with get_autocast_context():
            ev = model(inputs)
        _, _, p, u = evidential_outputs(ev.float())
        ps.append(p.cpu().numpy()); us.append(u.cpu().numpy()); ys.append(targets.numpy())
    return np.concatenate(ps), np.concatenate(us), np.concatenate(ys)

ev_eval = deeplob_evidential(y_len=K).to(device)
ev_eval.load_state_dict(torch.load(evidential_ckpt_path, map_location=device), strict=True)
p_test, u_test, y_test = collect(ev_eval, test_loader)
pred_test = p_test.argmax(1); maxp_test = p_test.max(1); correct = (pred_test == y_test)

print(f'\nTest samples: {len(y_test):,}')
print(f'Overall accuracy: {correct.mean():.4f}')
print(f'Uncertainty stats — mean={u_test.mean():.3f} median={np.median(u_test):.3f} '
      f'p10={np.quantile(u_test, 0.1):.3f} p90={np.quantile(u_test, 0.9):.3f}')
print(f'Uncertainty — correct: mean={u_test[correct].mean():.3f} | '
      f'incorrect: mean={u_test[~correct].mean():.3f}')

p_val, u_val, y_val = collect(ev_eval, val_loader)
pred_val = p_val.argmax(1); maxp_val = p_val.max(1); correct_val = (pred_val == y_val)

betas = np.linspace(0.34, 0.95, 25); taus = np.linspace(0.05, 1.00, 25)
rows = []
for b in betas:
    for t in taus:
        m = (maxp_val > b) & (u_val < t)
        cov = m.mean()
        rows.append((b, t, cov, correct_val[m].mean() if m.sum() else np.nan))
rows = np.array(rows)
ok = (rows[:, 2] >= 0.10)
best_idx = np.argmax(np.where(ok, rows[:, 3], -np.inf)) if ok.any() else np.argmax(rows[:, 3])
best_b, best_t, _, best_acc_val = rows[best_idx]
print(f'\nChosen on val: beta={best_b:.3f}, tau={best_t:.3f} -> val_acc_on_kept={best_acc_val:.4f}')

mask_test = (maxp_test > best_b) & (u_test < best_t)
print(f'Applied to test: coverage={mask_test.mean():.3f}, acc_on_kept={correct[mask_test].mean():.4f} '
      f'(baseline acc = {correct.mean():.4f})')
prec_full = precision_score(y_test, pred_test, average=None, zero_division=0)
prec_kept = precision_score(y_test[mask_test], pred_test[mask_test], average=None, zero_division=0) if mask_test.sum() else np.full(K, np.nan)
print(f'Precision per class — full: {np.round(prec_full, 4)} | kept: {np.round(prec_kept, 4)}')

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
ax.hist(u_test[correct], bins=40, alpha=0.6, label=f'correct (n={correct.sum():,})', density=True)
ax.hist(u_test[~correct], bins=40, alpha=0.6, label=f'incorrect (n={(~correct).sum():,})', density=True)
ax.set_xlabel('uncertainty u = K / S'); ax.set_ylabel('density')
ax.set_title('Uncertainty: correct vs incorrect'); ax.legend()

ax = axes[0, 1]
tau_grid = np.linspace(0.05, 1.0, 100)
covs, accs = [], []
for t in tau_grid:
    m = u_test < t
    covs.append(m.mean())
    accs.append(correct[m].mean() if m.any() else np.nan)
ax.plot(covs, accs, lw=2)
ax.axhline(correct.mean(), color='gray', ls='--', label=f'baseline acc={correct.mean():.3f}')
if mask_test.any():
    ax.scatter([mask_test.mean()], [correct[mask_test].mean()], c='red', s=80, zorder=5,
               label=f'chosen β={best_b:.2f}, τ={best_t:.2f}')
ax.set_xlabel('coverage'); ax.set_ylabel('accuracy on kept')
ax.set_title('Accuracy vs coverage (sweep τ)'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
edges = np.linspace(1.0/K, 1.0, 16)
centers = 0.5 * (edges[:-1] + edges[1:])
emp_acc = []; counts = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (maxp_test >= lo) & (maxp_test < hi)
    emp_acc.append(correct[m].mean() if m.any() else np.nan); counts.append(m.sum())
ax.plot([1/K, 1.0], [1/K, 1.0], 'k--', alpha=0.5, label='perfect')
ax.plot(centers, emp_acc, 'o-', label='empirical')
ax2 = ax.twinx()
ax2.bar(centers, counts, width=(edges[1]-edges[0])*0.9, alpha=0.15, color='gray')
ax2.set_ylabel('# samples (bar)')
ax.set_xlabel('predicted prob (max p)'); ax.set_ylabel('empirical accuracy')
ax.set_title('Reliability'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)

ax = axes[1, 1]
prec_by_cov = []
for t in tau_grid:
    m = u_test < t
    prec_by_cov.append(precision_score(y_test[m], pred_test[m], average='macro', zero_division=0) if m.sum() > 10 else np.nan)
ax.plot(covs, prec_by_cov, lw=2, color='C2')
base_prec = precision_score(y_test, pred_test, average='macro', zero_division=0)
ax.axhline(base_prec, color='gray', ls='--', label=f'baseline macro-prec={base_prec:.3f}')
ax.set_xlabel('coverage'); ax.set_ylabel('macro precision on kept')
ax.set_title('Precision vs coverage (sweep τ)'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig('evidential_uncertainty.png', dpi=120, bbox_inches='tight')
print('\nSaved: evidential_uncertainty.png')
