"""
model_update.py

Train/eval script for next-hour AEP load forecasting (PyTorch LSTM).

Expected input file (if available):
  data/AEP_hourly.csv with columns:
    - Datetime
    - AEP_MW (target)

Optional features:
  - calendar/seasonality features derived from Datetime (enabled via --add-time-features)
"""

from __future__ import annotations

import argparse
import os
import glob
from string import printable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# matplotlib is imported lazily for environments w/ locked caches.
plt = None  # type: ignore


def _get_plt(outdir: Path):
    """Lazy matplotlib import with a writable cache dir."""
    global plt
    if plt is not None:
        return plt

    mpl_config_dir = outdir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(outdir / ".cache"))

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as _plt  # type: ignore

        plt = _plt  # type: ignore
        return plt
    except Exception:
        return None


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data utilities
# -----------------------------
def load_aep_dataframe(csv_path) -> pd.DataFrame:
    """
    Project-specific loader.
    Expects exactly the columns in data/AEP_hourly.csv:
      - Datetime
      - AEP_MW
    """
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns or  len(df.columns) != 2:
        raise ValueError(f"Expected columns ['Datetime','AEP_MW'], got {list(df.columns)}")
    MW_col = ""
    for c in df.columns:
        if c != "Datetime":
            MW_col = c
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    
    df = df.dropna(subset=["Datetime", c]).copy()
    df = df.sort_values("Datetime").set_index("Datetime")
    df[c] = df[c].astype("float32")
    return df


def maybe_add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        hour = out.index.hour.values
        dow = out.index.dayofweek.values
        doy = out.index.dayofyear.values

        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return out
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X_raw: shape (N, num_features)
      y_raw: shape (N,)
      feature_names: list[str]
    """
    df = df.copy()

    feature_cols: List[str] = ["AEP_MW"]

    # Keep derived time features if they exist.
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]:
        if c in df.columns:
            feature_cols.append(c)

    X_raw = df[feature_cols].astype("float32").to_numpy()
    y_raw = df["AEP_MW"].astype("float32").to_numpy()
    return X_raw, y_raw, feature_cols


def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window:
      input:  X[t-seq_len+1 ... t] -> predict y[t+1]
    We implement it as:
      X_seq[i] = X[i : i+seq_len]
      y_seq[i] = y[i+seq_len]  (next hour after the window)
    Shapes:
      X_seq: (N-seq_len, seq_len, num_features)
      y_seq: (N-seq_len,)
    """
    if len(X) != len(y):
        raise ValueError("X and y length mismatch.")
    if len(X) <= seq_len:
        raise ValueError(f"Need more than seq_len={seq_len} rows; got {len(X)}.")

    xs = []
    ys = []
    for i in range(0, len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.stack(xs).astype("float32"), np.asarray(ys, dtype="float32")


# -----------------------------
# Model
# -----------------------------
class LSTMForecaster(nn.Module):
    """
    Input:  (B, T, F)
    Output: (B, 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        b = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=device)

        out, (_hn, _cn) = self.lstm(x, (h0, c0))  # out: (B, T, H)
        y_hat = self.head(out[:, -1, :])  # last time step
        return y_hat


# -----------------------------
# Training / evaluation
# -----------------------------
@dataclass
class TrainConfig:
    seq_len: int = 24
    hidden_dim: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 8
    splits: int = 5
    dropout: float = 0.1
    device: str = "auto"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = X.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], y[sl]


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    

    model = LSTMForecaster(input_dim=X_train.shape[-1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device).view(-1, 1)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device).view(-1, 1)

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}

    for _ in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in batch_iter(Xtr, ytr, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = loss_fn(val_pred, yva).item()

        history["train_mse"].append(float(np.mean(train_losses)))
        history["val_mse"].append(float(val_loss))

    model.eval()
    with torch.no_grad():
        val_pred = model(Xva).detach().cpu().numpy().reshape(-1)
    return history, val_pred, model


def walk_forward_validate(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    cfg: TrainConfig,
    device
) -> Dict[str, object]:
    """Walk-forward validation; scaler is fit on train split only."""
    splits = list(TimeSeriesSplit(n_splits=cfg.splits).split(X_raw))

    fold_metrics = []
    last_fold = {}
    models = []
    

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
        X_val_raw, y_val_raw = X_raw[val_idx], y_raw[val_idx]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = x_scaler.fit_transform(X_train_raw)
        X_val = x_scaler.transform(X_val_raw)

        y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
        y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(-1)

        # Make sequences inside each segment (no bleeding across boundary).
        Xtr_seq, ytr_seq = make_sequences(X_train, y_train, cfg.seq_len)
        Xva_seq, yva_seq = make_sequences(X_val, y_val, cfg.seq_len)

        history, yva_pred_scaled, tempModel = train_one_fold(Xtr_seq, ytr_seq, Xva_seq, yva_seq, cfg, device)
        models.append(tempModel)

        # Inverse-transform predictions to MW space for metrics/plots.
        yva_pred = y_scaler.inverse_transform(yva_pred_scaled.reshape(-1, 1)).reshape(-1)
        yva_true = y_scaler.inverse_transform(yva_seq.reshape(-1, 1)).reshape(-1)

        fold_rmse = rmse(yva_true, yva_pred)
        fold_mae = mae(yva_true, yva_pred)

        fold_metrics.append({"fold": fold, "rmse": fold_rmse, "mae": fold_mae})

        last_fold = {
            "fold": fold,
            "history": history,
            "y_true": yva_true,
            "y_pred": yva_pred,
        }

    rmse_mean = float(np.mean([m["rmse"] for m in fold_metrics]))
    mae_mean = float(np.mean([m["mae"] for m in fold_metrics]))
    totalModel = LSTMForecaster(input_dim=X_train.shape[-1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)

    states = totalModel.state_dict()
    for i in states:
        b = None
        for g in models:
            if (b == None):
                b = g[i]
            else:
                b += g[i]
        states[i] = b/len(models)
    totalModel.load_state_dict(states)


    return {
        "fold_metrics": fold_metrics,
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "last_fold": last_fold,
    }, totalModel


# -----------------------------
# Plotting
# -----------------------------
def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, outdir: Path) -> None:
    _plt = _get_plt(outdir)
    if _plt is None:
        raise RuntimeError("matplotlib is not available, cannot save plots.")
    _plt.figure(figsize=(12, 4))
    n = len(y_true)
    show = min(n, 7 * 24)  # show one week (or less)
    _plt.plot(y_true[-show:], label="Actual MW", linewidth=2)
    _plt.plot(y_pred[-show:], label="Predicted MW", linewidth=2, alpha=0.85)
    _plt.title("Predicted vs. Actual (last validation fold)")
    _plt.xlabel("Time (hours)")
    _plt.ylabel("MW")
    _plt.legend()
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=160)
    _plt.close()


def plot_loss_curve(history: Dict[str, List[float]], out_path: Path, outdir: Path) -> None:
    _plt = _get_plt(outdir)
    if _plt is None:
        raise RuntimeError("matplotlib is not available, cannot save plots.")
    _plt.figure(figsize=(7, 4))
    _plt.plot(history["train_mse"], label="Train MSE")
    _plt.plot(history["val_mse"], label="Val MSE")
    _plt.title("Loss Curve (MSE)")
    _plt.xlabel("Epoch")
    _plt.ylabel("MSE")
    _plt.legend()
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=160)
    _plt.close()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM walk-forward validation for hourly energy forecasting.")
    p.add_argument("--seq-len", type=int, default=24, help="Sequence length in hours.")
    p.add_argument("--splits", type=int, default=5, help="Walk-forward folds (TimeSeriesSplit).")
    p.add_argument("--epochs", type=int, default=8, help="Epochs per fold.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout on last hidden state.")
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps (if available).")
    p.add_argument("--add-time-features", action="store_true", help="Add sin/cos time features from timestamp.")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory for plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)

    


    # csv_paths = Path("data/")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    path = "data/"
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    os.chdir("..")

    # if not csv_paths.exists():
    #     raise FileNotFoundError(f"Missing dataset at {csv_paths}.")
    
    dataframes = []
    for csv in result:
        print(csv)
        
        # new_string = "".join(char for char in (path.join(csv)) if char in printable)
        # print(path.join(csv))
        dataframes.append(load_aep_dataframe(path+csv))
    print(len(dataframes))

    raise ValueError("just testing file loading")

    cfg = TrainConfig(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        splits=args.splits,
        dropout=args.dropout,
        device=args.device,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )


    if args.add_time_features:
        df = maybe_add_time_features(df)

    X_raw, y_raw, feature_names = build_feature_matrix(df)

    

    totalModel = LSTMForecaster(input_dim=X_train.shape[-1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    results, model = walk_forward_validate(X_raw, y_raw, cfg)

    torch.save(totalmod.state_dict(), 'currentmodel.pth')


    print("\n=== Walk-forward results (MW space) ===")
    print(f"Features used: {feature_names}")
    for m in results["fold_metrics"]:
        print(f"Fold {m['fold']:>2}: RMSE={m['rmse']:.2f} MW | MAE={m['mae']:.2f} MW")
    print(f"\nMean: RMSE={results['rmse_mean']:.2f} MW | MAE={results['mae_mean']:.2f} MW")

    lf = results["last_fold"]
    if lf:
        pred_path = outdir / "predicted_vs_actual.png"
        loss_path = outdir / "loss_curve.png"
        if _get_plt(outdir) is not None:
            plot_pred_vs_actual(lf["y_true"], lf["y_pred"], pred_path, outdir)
            plot_loss_curve(lf["history"], loss_path, outdir)
            print(f"\nSaved plots:")
            print(f"- Predicted vs Actual: {pred_path}")
            print(f"- Loss curve:         {loss_path}")
        else:
            print("\nmatplotlib not available; skipping plot file generation.")
            print("To enable plots: pip install matplotlib")

if __name__ == "__main__":
    main()

