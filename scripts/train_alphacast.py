"""
scripts/train_alphacast.py — Task 5: AlphaCast 训练管线
========================================================

离线训练 AlphaCast Transformer 模型：

    输入: CFL 标签 (labeling/counterfactual_labeler.py)
         + 178d 融合特征 (features/feature_fusion.py)
    模型: AlphaCastModel (models/alphacast/alphacast_model.py)
    输出: TorchScript (.pt) + ONNX (.onnx) + Triton 模型仓库

训练策略:
    - 类别不平衡: CFL 97.4% 负样本 → 加权 CrossEntropy + Focal Loss
    - 时序交叉验证: 滚动窗口 (避免未来信息泄露)
    - 早停: validation NLL 连续 5 epoch 不下降
    - 学习率调度: Cosine Annealing with Warm Restarts

用法:
    python scripts/train_alphacast.py --data data/cfl_labels.parquet --epochs 50
    python scripts/train_alphacast.py --data data/cfl_labels.parquet --export-triton
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.logging_setup import get_logger

_log = get_logger("scripts.train_alphacast")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore
    _log.error("PyTorch required for training; pip install torch")

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# ============================================================
# 配置
# ============================================================

DEFAULT_TRAIN_CONFIG = {
    # 模型
    "input_dim": 178,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "seq_len": 60,

    # 训练
    "batch_size": 64,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    "patience": 5,  # 早停耐心

    # 类别不平衡
    "pos_weight": 10.0,  # 正样本权重
    "focal_gamma": 2.0,  # Focal Loss gamma

    # 数据
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # 输出
    "output_dir": "models/alphacast/checkpoints",
    "export_torchscript": True,
    "export_onnx": True,
    "export_triton": False,
}


# ============================================================
# 数据加载
# ============================================================

def load_training_data(
    parquet_path: str,
    seq_len: int = 60,
    input_dim: int = 178,
) -> Tuple[Any, Any, Any]:
    """
    从 CFL 标签 Parquet 加载训练数据

    期望列: ts_ms, cfl_label, cfl_weight, features_178d (list[float])

    Returns: (X_train, y_train, w_train) as torch tensors
    """
    if pl is None:
        raise ImportError("polars required; pip install polars")

    df = pl.read_parquet(parquet_path)
    _log.info(f"Loaded {len(df)} rows from {parquet_path}")

    # 过滤中性标签 (只保留 +1 / -1 用于二分类)
    df_filtered = df.filter(pl.col("cfl_label") != 0)
    _log.info(f"Non-neutral samples: {len(df_filtered)} "
              f"(+1: {df_filtered.filter(pl.col('cfl_label') == 1).height}, "
              f"-1: {df_filtered.filter(pl.col('cfl_label') == -1).height})")

    # 提取特征和标签
    # features_178d 列应该是 list[float] 类型
    if "features_178d" not in df_filtered.columns:
        raise ValueError("Parquet must contain 'features_178d' column (list[float])")

    features_list = df_filtered["features_178d"].to_list()
    labels = df_filtered["cfl_label"].to_list()
    weights = df_filtered["cfl_weight"].to_list() if "cfl_weight" in df_filtered.columns else [1.0] * len(labels)

    # 转换为 tensor
    X = torch.tensor(features_list, dtype=torch.float32)  # [N, 178]
    y = torch.tensor([1 if l > 0 else 0 for l in labels], dtype=torch.float32)  # 二分类
    w = torch.tensor(weights, dtype=torch.float32)

    # 构建序列 (滑动窗口)
    N = len(X)
    if N <= seq_len:
        raise ValueError(f"Not enough samples ({N}) for seq_len={seq_len}")

    X_seq = []
    y_seq = []
    w_seq = []
    for i in range(seq_len, N):
        X_seq.append(X[i - seq_len:i])  # [seq_len, 178]
        y_seq.append(y[i])
        w_seq.append(w[i])

    X_seq = torch.stack(X_seq)  # [N-seq_len, seq_len, 178]
    y_seq = torch.tensor(y_seq, dtype=torch.float32)
    w_seq = torch.tensor(w_seq, dtype=torch.float32)

    _log.info(f"Sequences: {X_seq.shape}, Labels: {y_seq.shape}, "
              f"Pos rate: {y_seq.mean():.3f}")

    return X_seq, y_seq, w_seq


# ============================================================
# 训练循环
# ============================================================

def train(
    X: Any, y: Any, w: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    训练 AlphaCast 模型

    Returns: 训练结果统计
    """
    from models.alphacast.alphacast_model import AlphaCastModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"Device: {device}")

    # 数据分割 (时序，不 shuffle)
    N = len(X)
    train_end = int(N * config["train_ratio"])
    val_end = int(N * (config["train_ratio"] + config["val_ratio"]))

    X_train, y_train, w_train = X[:train_end].to(device), y[:train_end].to(device), w[:train_end].to(device)
    X_val, y_val = X[train_end:val_end].to(device), y[train_end:val_end].to(device)
    X_test, y_test = X[val_end:].to(device), y[val_end:].to(device)

    _log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 构建 DataLoader
    train_ds = TensorDataset(X_train, y_train, w_train)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

    # 模型
    model = AlphaCastModel(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        seq_len=config["seq_len"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    _log.info(f"Model params: {total_params:,}")

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 损失函数 (BCE + Focal)
    pos_weight = torch.tensor([config["pos_weight"]]).to(device)

    def focal_loss(pred, target, gamma=config["focal_gamma"]):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction='none')
        pt = torch.exp(-bce)
        return ((1 - pt) ** gamma * bce).mean()

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            pred_return = output["predicted_return"]
            loss = focal_loss(pred_return, batch_y)
            # 样本加权
            weighted_loss = (loss * batch_w).mean() if loss.dim() > 0 else loss
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += weighted_loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = val_output["predicted_return"]
            val_loss = focal_loss(val_pred, y_val).item()
            val_acc = ((val_pred > 0) == (y_val > 0.5)).float().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if (epoch + 1) % 5 == 0:
            _log.info(f"Epoch {epoch+1}/{config['epochs']}: "
                      f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.3f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            os.makedirs(config["output_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                _log.info(f"Early stopping at epoch {epoch+1} (patience={config['patience']})")
                break

    # 加载最佳模型并评估
    model.load_state_dict(torch.load(os.path.join(config["output_dir"], "best_model.pt")))
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_pred = test_output["predicted_return"]
        test_acc = ((test_pred > 0) == (y_test > 0.5)).float().mean().item()

    _log.info(f"Test accuracy: {test_acc:.3f}")

    return {
        "test_accuracy": test_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "total_params": total_params,
        "history": history,
    }


# ============================================================
# 模型导出
# ============================================================

def export_model(config: Dict[str, Any]) -> None:
    """导出模型为 TorchScript / ONNX / Triton 格式"""
    from models.alphacast.alphacast_model import AlphaCastModel, export_torchscript, export_onnx

    model = AlphaCastModel(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )

    # 加载最佳权重
    ckpt_path = os.path.join(config["output_dir"], "best_model.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        _log.info(f"Loaded checkpoint: {ckpt_path}")

    if config.get("export_torchscript", True):
        export_torchscript(model, os.path.join(config["output_dir"], "alphacast_model.pt"))

    if config.get("export_onnx", True):
        export_onnx(model, os.path.join(config["output_dir"], "alphacast_model.onnx"))

    if config.get("export_triton", False):
        _setup_triton_repo(model, config)


def _setup_triton_repo(model: Any, config: Dict[str, Any]) -> None:
    """设置 Triton 模型仓库目录结构"""
    triton_dir = os.path.join(_PROJECT_ROOT, "triton_model_repository", "alphacast_resnet")
    version_dir = os.path.join(triton_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    # 导出 TorchScript 到 Triton 目录
    from models.alphacast.alphacast_model import export_torchscript
    export_torchscript(model, os.path.join(version_dir, "model.pt"))

    # 写入 config.pbtxt
    config_pbtxt = f"""name: "alphacast_resnet"
platform: "pytorch_libtorch"
max_batch_size: 64
input [
  {{
    name: "features"
    data_type: TYPE_FP32
    dims: [ {config['seq_len']}, {config['input_dim']} ]
  }}
]
output [
  {{
    name: "predicted_return"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }},
  {{
    name: "uncertainty"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }},
  {{
    name: "confidence"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }},
  {{
    name: "market_state"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
    with open(os.path.join(triton_dir, "config.pbtxt"), "w") as f:
        f.write(config_pbtxt)

    _log.info(f"Triton model repo set up: {triton_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train AlphaCast model")
    parser.add_argument("--data", type=str, required=True, help="Path to CFL labels parquet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--export-triton", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models/alphacast/checkpoints")
    args = parser.parse_args()

    config = DEFAULT_TRAIN_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["output_dir"] = args.output_dir
    config["export_triton"] = args.export_triton

    if torch is None:
        print("ERROR: PyTorch required. pip install torch")
        sys.exit(1)

    # 加载数据
    X, y, w = load_training_data(args.data, seq_len=config["seq_len"])

    # 训练
    result = train(X, y, w, config)
    print(json.dumps(result, indent=2, default=str))

    # 导出
    export_model(config)

    print("✓ Training complete")


if __name__ == "__main__":
    main()
