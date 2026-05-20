"""
Per-Appliance Ensemble: RL LNN (PPO) vs Advanced LNN Fine-Tune — UKDALE

Pipeline
--------
  1. Train RL LNN on fine_tuning_dataset/pretrain  (supervised warm-start + PPO)
  2. Train Advanced LNN on fine_tuning_dataset/pretrain + finetune  (supervised → FT)
  3. Evaluate both on fine_tuning_dataset/validation
  4. For each appliance select the model with higher  score = f1 - 0.001 * mae
  5. Report selected model's test metrics  (no test leakage in selection)

Both models evaluate on the SAME val/test splits so scores are directly comparable.
Advanced LNN has the expected advantage of an explicit fine-tune step on House 5 data.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Source Code')
sys.path.insert(0, _SRC)
from models import AdvancedLiquidNetworkModel
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD  = 10.0
WIN = 100;  STRIDE = 5;  BATCH = 32

DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'fine_tuning_dataset')

# RL hyperparameters
RL_LR            = 3e-4
RL_PRETRAIN_EP   = 20
RL_EPOCHS        = 50
RL_PATIENCE      = 15
RL_GAMMA         = 0.99
RL_GAE_LAMBDA    = 0.95
RL_PPO_CLIP      = 0.2
RL_VALUE_COEF    = 0.5
RL_ENTROPY_COEF  = 0.01
RL_PPO_EPOCHS    = 4
RL_ROLLOUT_SIZE  = 512
RL_ALPHA         = 1.0
RL_BETA_TRANS    = 0.05
RL_NORM_POWER    = 3000.0

# Advanced LNN hyperparameters
ADV_LR          = 1e-3;  ADV_EPOCHS    = 80;  ADV_PATIENCE    = 20
ADV_LR_FT       = 1e-4;  ADV_EPOCHS_FT = 30;  ADV_PATIENCE_FT = 10
ADV_HIDDEN      = 64;    ADV_LAYERS    = 2


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_splits(dataset_dir):
    splits = {}
    for name in ('pretrain', 'validation', 'finetune', 'test'):
        path = os.path.join(dataset_dir, f'UKDALE_HF_{name}.csv')
        splits[name] = pd.read_csv(path)
        print(f"  {name:12s}: {len(splits[name]):6,} rows")
    return splits


def make_sequences_multi(df):
    """All-appliance windows for the RL model."""
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(Y, dtype=np.float32))


def make_sequences_single(df, appliance):
    """Per-appliance windows for the Advanced LNN."""
    mains = df['aggregate'].values
    tgts  = df[appliance].values
    X, y  = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        y.append(tgts[i + WIN // 2])
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(y, dtype=np.float32).reshape(-1, 1))


def nilm_metrics(raw_true, raw_pred):
    m = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLD)
    m['TP'] = int(((raw_true > THRESHOLD) & (raw_pred > THRESHOLD)).sum())
    m['FP'] = int(((raw_true <= THRESHOLD) & (raw_pred > THRESHOLD)).sum())
    m['FN'] = int(((raw_true > THRESHOLD) & (raw_pred <= THRESHOLD)).sum())
    return m


# ---------------------------------------------------------------------------
# RL model architecture  (LNNActorCritic — identical to test_lnn_rl_ukdale.py)
# ---------------------------------------------------------------------------

class LNNActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate          = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm          = nn.LayerNorm(hidden_size)
        self.actor_mean    = nn.Linear(hidden_size, n_appliances)
        self.actor_log_std = nn.Parameter(torch.full((n_appliances,), -1.0))
        self.critic        = nn.Linear(hidden_size, 1)

    def _encode(self, x):
        B, T, _ = x.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            xt  = x[:, t, :]
            inp = self.input_proj(xt)
            rec = torch.matmul(h, self.rec_weights)
            tau = (F.softplus(self.tau_base).unsqueeze(0)
                   * torch.sigmoid(self.tau_mod(xt))).clamp(min=self.dt)
            g   = torch.sigmoid(self.gate(torch.cat([xt, h], dim=1)))
            dh  = ((-h / tau) + g * torch.tanh(inp + rec)) * self.dt
            h   = (h + dh).clamp(-10.0, 10.0)
        return self.norm(h)

    def get_action(self, x, deterministic=False):
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(0.01, 2.0)
        dist = Normal(mean, std)
        action   = mean if deterministic else dist.rsample().clamp(0.0, 1.0)
        log_prob = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)
        return action, log_prob, dist.entropy().sum(-1), self.critic(h).squeeze(-1)

    def evaluate_actions(self, x, actions):
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(0.01, 2.0)
        dist = Normal(mean, std)
        lp   = dist.log_prob(actions.clamp(1e-6, 1 - 1e-6)).sum(-1)
        return lp, dist.entropy().sum(-1), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# RL training
# ---------------------------------------------------------------------------

def _rl_rewards(actions, y_true, prev, y_mins, y_ranges):
    pr = actions * y_ranges + y_mins
    tr = y_true  * y_ranges + y_mins
    pv = prev    * y_ranges + y_mins
    return (-(RL_ALPHA * np.abs(pr - tr).mean(1)
               + RL_BETA_TRANS * np.abs(pr - pv).mean(1)) / RL_NORM_POWER)


def _gae(rewards, values, last_value):
    adv, gae = [], 0.0
    ve = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + RL_GAMMA * ve[t + 1] - ve[t]
        gae   = delta + RL_GAMMA * RL_GAE_LAMBDA * gae
        adv.insert(0, gae)
    return adv, [a + v for a, v in zip(adv, values)]


def _eval_rl(model, X, Y_sc, y_scalers, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            a, _, _, _ = model.get_action(
                torch.FloatTensor(X[i:i + BATCH]).to(device), deterministic=True)
            preds.append(a.cpu().numpy())
            trues.append(Y_sc[i:i + BATCH])
    ps = np.concatenate(preds);  ts = np.concatenate(trues)
    return {a: nilm_metrics(
                y_scalers[i].inverse_transform(ts[:, i:i+1]).flatten(),
                y_scalers[i].inverse_transform(ps[:, i:i+1]).flatten())
            for i, a in enumerate(APPLIANCES)}


def train_rl(splits, device, hidden_size=64, dt=0.1):
    print(f"\n{'='*60}\nRL LNN (PPO)  —  pretrain split only\n{'='*60}")

    X_tr, Y_tr = make_sequences_multi(splits['pretrain'])
    X_va, Y_va = make_sequences_multi(splits['validation'])
    X_te, Y_te = make_sequences_multi(splits['test'])

    xs = MinMaxScaler()
    X_tr = xs.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = xs.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_sc = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_sc.append(ys)

    y_mins   = np.array([float(s.data_min_[0])   for s in y_sc])
    y_ranges = np.array([float(s.data_range_[0]) for s in y_sc])

    model = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=RL_LR, eps=1e-5)

    # Supervised warm-start
    print(f"  Supervised warm-start ({RL_PRETRAIN_EP} epochs)")
    model.train()
    pre_opt = torch.optim.Adam(model.parameters(), lr=RL_LR)
    for ep in range(RL_PRETRAIN_EP):
        perm = np.random.permutation(len(X_tr))
        tot = 0.0;  n = 0
        for s in range(0, len(X_tr), BATCH):
            idx = perm[s:s + BATCH]
            xb  = torch.FloatTensor(X_tr[idx]).to(device)
            yb  = torch.FloatTensor(Y_tr[idx]).to(device)
            h   = model._encode(xb)
            mse = F.mse_loss(torch.sigmoid(model.actor_mean(h)), yb)
            pre_opt.zero_grad();  mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            pre_opt.step()
            tot += mse.item();  n += 1
        print(f"    pretrain {ep+1:2d}/{RL_PRETRAIN_EP}  MSE={tot/n:.6f}")

    # PPO loop
    print(f"  PPO training ({RL_EPOCHS} epochs max)")
    best_f1 = -1.0;  best_state = None;  counter = 0
    for epoch in range(RL_EPOCHS):
        model.train()
        prev_np = np.zeros(len(APPLIANCES), dtype=np.float32)
        for rs in range(0, len(X_tr), RL_ROLLOUT_SIZE):
            re = min(rs + RL_ROLLOUT_SIZE, len(X_tr))
            N  = re - rs
            xc = torch.FloatTensor(X_tr[rs:re]).to(device)
            yc = Y_tr[rs:re]
            model.eval()
            with torch.no_grad():
                acts_t, lp_t, _, vals_t = model.get_action(xc)
            an = acts_t.cpu().numpy();  ln = lp_t.cpu().numpy()
            vn = vals_t.cpu().numpy().tolist()
            pr = np.vstack([prev_np[None, :], an[:-1]])
            rn = _rl_rewards(an, yc, pr, y_mins, y_ranges)
            prev_np = an[-1]
            with torch.no_grad():
                _, _, _, lv = model.get_action(xc[-1:], deterministic=True)
            adv_l, ret_l = _gae(rn.tolist(), vn, lv[0].item())
            adv_t = torch.tensor(adv_l, dtype=torch.float32, device=device)
            ret_t = torch.tensor(ret_l, dtype=torch.float32, device=device)
            olp_t = torch.tensor(ln,    dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            model.train()
            for _ in range(RL_PPO_EPOCHS):
                perm = torch.randperm(N, device=device)
                for s in range(0, N, BATCH):
                    idx = perm[s:s + BATCH]
                    nlp, ent, vals = model.evaluate_actions(xc[idx], acts_t[idx])
                    r   = (nlp - olp_t[idx]).exp()
                    adv = adv_t[idx]
                    pl  = -torch.min(r * adv,
                                     r.clamp(1 - RL_PPO_CLIP,
                                             1 + RL_PPO_CLIP) * adv).mean()
                    vl  = RL_VALUE_COEF * (ret_t[idx] - vals).pow(2).mean()
                    el  = -RL_ENTROPY_COEF * ent.mean()
                    opt.zero_grad();  (pl + vl + el).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()

        vm     = _eval_rl(model, X_va, Y_va, y_sc, device)
        avg_f1 = np.mean([vm[a]['f1'] for a in APPLIANCES])
        avg_mae= np.mean([vm[a]['mae'] for a in APPLIANCES])
        print(f"  RL Ep {epoch+1:3d}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1;  counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= RL_PATIENCE:
                print(f"  RL early stop at epoch {epoch+1}");  break

    model.load_state_dict(best_state)
    return (model,
            _eval_rl(model, X_va, Y_va, y_sc, device),
            _eval_rl(model, X_te, Y_te, y_sc, device))


# ---------------------------------------------------------------------------
# Advanced LNN training
# ---------------------------------------------------------------------------

class _DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X);  self.y = torch.FloatTensor(y)
    def __len__(self):         return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]


def _run_adv(model, loader, crit, opt, device, train=True):
    model.train() if train else model.eval()
    tot = 0.0;  outs, tgts = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train: opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()
            outs.append(out.detach().cpu().numpy())
            tgts.append(yb.cpu().numpy())
    return tot / len(loader), np.concatenate(outs), np.concatenate(tgts)


def train_adv(splits, device, hidden_size=ADV_HIDDEN, num_layers=ADV_LAYERS, dt=0.1):
    print(f"\n{'='*60}\nAdvanced LNN  —  pretrain + fine-tune\n{'='*60}")

    val_metrics  = {};  test_metrics = {}

    for app in APPLIANCES:
        print(f"\n  {app}")
        X_pre, y_pre = make_sequences_single(splits['pretrain'],   app)
        X_val, y_val = make_sequences_single(splits['validation'], app)
        X_ft,  y_ft  = make_sequences_single(splits['finetune'],   app)
        X_te,  y_te  = make_sequences_single(splits['test'],       app)

        xs = MinMaxScaler();  ys = MinMaxScaler()
        X_pre = xs.fit_transform(X_pre.reshape(-1, 1)).reshape(X_pre.shape)
        X_val = xs.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_ft  = xs.transform(X_ft.reshape(-1, 1)).reshape(X_ft.shape)
        X_te  = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)
        y_pre = ys.fit_transform(y_pre);  y_val = ys.transform(y_val)
        y_ft  = ys.transform(y_ft);       y_te  = ys.transform(y_te)

        mk = lambda X, y, sh: torch.utils.data.DataLoader(
            _DS(X, y), batch_size=BATCH, shuffle=sh)
        pre_ldr = mk(X_pre, y_pre, True)
        val_ldr = mk(X_val, y_val, False)
        ft_ldr  = mk(X_ft,  y_ft,  True)
        te_ldr  = mk(X_te,  y_te,  False)

        model = AdvancedLiquidNetworkModel(1, hidden_size, 1, num_layers, dt).to(device)
        crit  = nn.MSELoss()

        # Phase 1: supervised pretrain
        opt   = torch.optim.Adam(model.parameters(), lr=ADV_LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        best_val = float('inf');  best_st = None;  ctr = 0
        for ep in range(ADV_EPOCHS):
            _run_adv(model, pre_ldr, crit, opt, device, True)
            va_l, _, _ = _run_adv(model, val_ldr, crit, opt, device, False)
            sched.step(va_l)
            if va_l < best_val:
                best_val = va_l;  ctr = 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                ctr += 1
                if ctr >= ADV_PATIENCE:
                    print(f"    pretrain early stop ep {ep+1}");  break
        model.load_state_dict(best_st)

        # Phase 2: fine-tune on House 5
        ft_opt = torch.optim.Adam(model.parameters(), lr=ADV_LR_FT)
        best_ft = float('inf');  best_ft_st = None;  ft_ctr = 0
        for ep in range(ADV_EPOCHS_FT):
            tr_l, _, _ = _run_adv(model, ft_ldr, crit, ft_opt, device, True)
            if tr_l < best_ft:
                best_ft = tr_l;  ft_ctr = 0
                best_ft_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                ft_ctr += 1
                if ft_ctr >= ADV_PATIENCE_FT:
                    print(f"    finetune early stop ep {ep+1}");  break
        model.load_state_dict(best_ft_st)

        # Evaluate
        _, vo, vt = _run_adv(model, val_ldr, crit, ft_opt, device, False)
        _, to, tt = _run_adv(model, te_ldr,  crit, ft_opt, device, False)
        val_metrics[app]  = nilm_metrics(ys.inverse_transform(vt).flatten(),
                                          ys.inverse_transform(vo).flatten())
        test_metrics[app] = nilm_metrics(ys.inverse_transform(tt).flatten(),
                                          ys.inverse_transform(to).flatten())
        print(f"    val  F1={val_metrics[app]['f1']:.4f}  MAE={val_metrics[app]['mae']:.2f}")
        print(f"    test F1={test_metrics[app]['f1']:.4f}  MAE={test_metrics[app]['mae']:.2f}")

    return val_metrics, test_metrics


# ---------------------------------------------------------------------------
# Ensemble selection
# ---------------------------------------------------------------------------

def _score(m):
    return m['f1'] - 0.001 * m['mae']


def select_ensemble(val_rl, test_rl, val_adv, test_adv):
    selected = {}
    for app in APPLIANCES:
        sa = _score(val_rl[app]);   sb = _score(val_adv[app])
        if sa >= sb:
            selected[app] = {'model': 'rl_lnn_ppo',
                             'val_score': sa, 'test_metrics': test_rl[app]}
        else:
            selected[app] = {'model': 'advanced_lnn_ft',
                             'val_score': sb, 'test_metrics': test_adv[app]}
    return selected


def print_comparison(val_rl, test_rl, val_adv, test_adv, selected):
    cols = ('F1', 'MAE', 'SAE')
    print(f"\n{'='*72}")
    print("PER-APPLIANCE COMPARISON  (val used for selection, test for reporting)")
    print(f"{'='*72}")
    print(f"  {'App':<22}  {'--- RL LNN ---':^20}  {'-- Adv LNN --':^20}  {'Chosen':^14}")
    print(f"  {'':22}  {'F1':>6} {'MAE':>6} {'SAE':>6}  {'F1':>6} {'MAE':>6} {'SAE':>6}  {'model':^14}")
    print(f"  {'-'*70}")
    for app in APPLIANCES:
        mr = test_rl[app];  ma = test_adv[app];  ch = selected[app]
        flag = '<-' if ch['model'] == 'rl_lnn_ppo' else '  '
        print(f"  {app:<22}  "
              f"{mr['f1']:>6.4f} {mr['mae']:>6.1f} {mr['sae']:>6.4f}  "
              f"{ma['f1']:>6.4f} {ma['mae']:>6.1f} {ma['sae']:>6.4f}  "
              f"{ch['model']:<14} {flag}")

    print(f"\n  ENSEMBLE TEST RESULTS")
    print(f"  {'App':<22}  {'Model':<20}  {'ValScore':>9}  {'F1':>6}  {'P':>6}  {'R':>6}  {'MAE':>7}  {'SAE':>7}")
    print(f"  {'-'*80}")
    for app in APPLIANCES:
        s = selected[app];  m = s['test_metrics']
        print(f"  {app:<22}  {s['model']:<20}  {s['val_score']:>9.4f}  "
              f"{m['f1']:>6.4f}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  "
              f"{m['mae']:>7.2f}  {m['sae']:>7.4f}")
    avg_f1  = np.mean([selected[a]['test_metrics']['f1']  for a in APPLIANCES])
    avg_mae = np.mean([selected[a]['test_metrics']['mae'] for a in APPLIANCES])
    avg_sae = np.mean([selected[a]['test_metrics']['sae'] for a in APPLIANCES])
    print(f"  {'-'*80}")
    print(f"  {'AVERAGE':<22}  {'':20}  {'':9}  "
          f"{avg_f1:>6.4f}  {'':6}  {'':6}  {avg_mae:>7.2f}  {avg_sae:>7.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, dt=0.1, save_dir=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_dir is None:
        save_dir = f'models/ensemble_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  hidden={hidden_size}  dt={dt}")
    print("\nLoading fine_tuning_dataset splits...")
    splits = load_splits(dataset_dir)

    wall = time.time()

    _, val_rl,  test_rl  = train_rl(splits, device, hidden_size, dt)
    val_adv, test_adv    = train_adv(splits, device, hidden_size, dt=dt)

    selected = select_ensemble(val_rl, test_rl, val_adv, test_adv)
    print_comparison(val_rl, test_rl, val_adv, test_adv, selected)

    result = {
        'timestamp':      timestamp,
        'selection_rule': 'f1 - 0.001 * mae  on validation set',
        'ensemble': {
            app: {
                'chosen_model': s['model'],
                'val_score':    float(s['val_score']),
                'test_metrics': {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                 for k, v in s['test_metrics'].items()},
            }
            for app, s in selected.items()
        },
        'rl_val_metrics':   {a: {k: float(v) for k, v in val_rl[a].items()}   for a in APPLIANCES},
        'adv_val_metrics':  {a: {k: float(v) for k, v in val_adv[a].items()}  for a in APPLIANCES},
        'rl_test_metrics':  {a: {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                  for k, v in test_rl[a].items()}  for a in APPLIANCES},
        'adv_test_metrics': {a: {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                  for k, v in test_adv[a].items()} for a in APPLIANCES},
    }
    with open(os.path.join(save_dir, 'ensemble_results.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    print(f"\nResults → {save_dir}/ensemble_results.json")
    print(f"Total time: {(time.time()-wall)/60:.1f} min")
    return selected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Per-appliance ensemble: RL LNN vs Advanced LNN fine-tune')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--dt',           type=float, default=0.1)
    p.add_argument('--save-dir',     default=None)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.dt, args.save_dir)
