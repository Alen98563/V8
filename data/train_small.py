#!/usr/bin/env python3
"""
train_small.py — Lightweight AlphaCast training for N150 (4-core, 15GB RAM)
Matches 567-sample dataset: smaller model, no OOM.

Run: cd /home/jerry/V8 && PYTHONPATH=. python3 data/train_small.py
"""

import torch, torch.nn as nn, torch.optim as optim
import polars as pl
import os, sys, time, json
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/home/jerry/V8')
from models.alphacast.alphacast_model import AlphaCastModel

# =====================================================
# Config (matched to 567 samples, N150 hardware)
# =====================================================
CONFIG = {
    "input_dim": 178, "d_model": 128, "nhead": 4, "num_layers": 3,
    "dim_feedforward": 512, "dropout": 0.2, "seq_len": 60,
    "batch_size": 16, "epochs": 20, "lr": 1e-3, "weight_decay": 1e-4,
    "warmup_epochs": 3, "patience": 8,
    "pos_weight": 1.5, "focal_gamma": 1.5,
    "train_ratio": 0.7, "val_ratio": 0.15,
    "output_dir": "/home/jerry/V8/models/alphacast/checkpoints",
}

# =====================================================
# Load data
# =====================================================
df = pl.read_parquet("/home/jerry/V8/data/train_combined.parquet")
print(f"Loaded {df.height} rows")

features_list = df["features_178d"].to_list()
labels = df["cfl_label"].to_list()

X = torch.tensor(features_list, dtype=torch.float32)
y = torch.tensor([1 if l > 0 else 0 for l in labels], dtype=torch.float32)

# Build sequences (sliding window)
seq_len = CONFIG["seq_len"]
N = len(X)
X_seq, y_seq = [], []
for i in range(seq_len, N):
    X_seq.append(X[i-seq_len:i])
    y_seq.append(y[i])
X_seq = torch.stack(X_seq)  # [N-seq_len, seq_len, 178]
y_seq = torch.tensor(y_seq, dtype=torch.float32)

print(f"Sequences: {X_seq.shape}, pos rate: {y_seq.mean():.3f}")

# Split (时序, no shuffle)
N2 = len(X_seq)
t_end = int(N2 * CONFIG["train_ratio"])
v_end = int(N2 * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))
X_tr, y_tr = X_seq[:t_end], y_seq[:t_end]
X_va, y_va = X_seq[t_end:v_end], y_seq[t_end:v_end]
X_te, y_te = X_seq[v_end:], y_seq[v_end:]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

# =====================================================
# Model
# =====================================================
device = torch.device("cpu")
model = AlphaCastModel(
    input_dim=CONFIG["input_dim"], d_model=CONFIG["d_model"],
    nhead=CONFIG["nhead"], num_layers=CONFIG["num_layers"],
    dim_feedforward=CONFIG["dim_feedforward"], dropout=CONFIG["dropout"],
    seq_len=CONFIG["seq_len"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params, device: {device}")

# =====================================================
# Training
# =====================================================
ds = TensorDataset(X_tr, y_tr)
loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True)
opt = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=2)

pos_w = torch.tensor([CONFIG["pos_weight"]])
def focal_loss(pred, target):
    bce = nn.functional.binary_cross_entropy_with_logits(
        pred, target, pos_weight=pos_w, reduction='none')
    pt = torch.exp(-bce)
    return ((1 - pt) ** CONFIG["focal_gamma"] * bce).mean()

best_val_loss = float("inf")
patience_cnt = 0
hist = {"train_loss": [], "val_loss": [], "val_acc": []}

t0 = time.time()
for epoch in range(CONFIG["epochs"]):
    model.train()
    total_loss, n_batch = 0.0, 0
    for bx, by in loader:
        opt.zero_grad()
        out = model(bx)
        loss = focal_loss(out["predicted_return"], by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batch += 1
    sched.step()

    # Validation
    model.eval()
    with torch.no_grad():
        vo = model(X_va)
        v_loss = focal_loss(vo["predicted_return"], y_va).item()
        v_acc = ((vo["predicted_return"] > 0) == (y_va > 0.5)).float().mean().item()
    
    hist["train_loss"].append(total_loss / max(n_batch, 1))
    hist["val_loss"].append(v_loss)
    hist["val_acc"].append(v_acc)

    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"Ep {epoch+1:2d}/{CONFIG['epochs']}: tloss={hist['train_loss'][-1]:.4f} "
              f"vloss={v_loss:.4f} vacc={v_acc:.3f} "
              f"lr={sched.get_last_lr()[0]:.6f}")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_cnt = 0
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "best_model.pt"))
    else:
        patience_cnt += 1
        if patience_cnt >= CONFIG["patience"]:
            print(f"Early stop at ep {epoch+1}")
            break

elapsed = time.time() - t0

# =====================================================
# Final eval
# =====================================================
model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], "best_model.pt")))
model.eval()
with torch.no_grad():
    to = model(X_te)
    te_acc = ((to["predicted_return"] > 0) == (y_te > 0.5)).float().mean().item()
    te_preds = torch.sigmoid(to["predicted_return"]).numpy()
    te_true = y_te.numpy()
    
    # Directional stats
    correct_long = ((te_preds > 0.5) & (te_true == 1)).sum()
    correct_short = ((te_preds <= 0.5) & (te_true == 0)).sum()
    pred_long = (te_preds > 0.5).sum()
    pred_short = (te_preds <= 0.5).sum()

result = {
    "model_params": n_params,
    "test_accuracy": float(te_acc),
    "best_val_loss": float(best_val_loss),
    "epochs_trained": len(hist["train_loss"]),
    "train_time_sec": round(elapsed, 1),
    "test_correct_long": int(correct_long),
    "test_correct_short": int(correct_short),
    "test_pred_long": int(pred_long),
    "test_pred_short": int(pred_short),
}
print(f"\n{'='*50}")
print(f"DONE in {elapsed:.1f}s | Test acc: {te_acc:.3f}")
print(f"Long:  {correct_long}/{pred_long} correct | Short: {correct_short}/{pred_short} correct")
print(f"Model saved: {CONFIG['output_dir']}/best_model.pt ({n_params:,} params)")
print(json.dumps(result, indent=2))

# Export TorchScript
try:
    from models.alphacast.alphacast_model import export_torchscript
    export_torchscript(model, os.path.join(CONFIG["output_dir"], "alphacast_model.pt"))
    print("TorchScript exported ✓")
except Exception as e:
    print(f"TorchScript export failed: {e}")
