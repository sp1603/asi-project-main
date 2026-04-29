from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_PATH = Path("data/hangzhou_1x1_bc-tyc_18041607_1h_lstm_data.csv")
MODEL_PATH = Path("models/hangzhou_1x1_lstm.pt")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    history: int = 12
    horizon: int = 3
    batch_size: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 30
    input_dim: int = 5
    output_dim: int = 4


class TrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TrafficLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def make_windows(df: pd.DataFrame, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = ["dir_0", "dir_1", "dir_2", "dir_3", "phase"]
    target_cols = ["dir_0", "dir_1", "dir_2", "dir_3"]

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for episode_id, g in df.groupby("episode"):
        g = g.sort_values("step")
        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_cols].to_numpy(dtype=np.float32)

        for t in range(len(g) - history - horizon + 1):
            X_list.append(features[t : t + history])
            y_list.append(targets[t + history + horizon - 1])

    if not X_list:
        raise ValueError("No sliding windows were created. Increase data or reduce history/horizon.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def normalize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    mean = X_train[:, :, :4].mean(axis=(0, 1), keepdims=True)
    std = X_train[:, :, :4].std(axis=(0, 1), keepdims=True) + 1e-6

    X_train = X_train.copy()
    X_val = X_val.copy()
    y_train = y_train.copy()
    y_val = y_val.copy()

    X_train[:, :, :4] = (X_train[:, :, :4] - mean) / std
    X_val[:, :, :4] = (X_val[:, :, :4] - mean) / std
    y_train = (y_train - mean.reshape(-1)) / std.reshape(-1)
    y_val = (y_val - mean.reshape(-1)) / std.reshape(-1)

    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    return X_train, X_val, y_train, y_val, stats


def persistence_baseline_rmse(X: np.ndarray, y: np.ndarray) -> float:
    pred = X[:, -1, :4]
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> None:
    cfg = Config()
    df = pd.read_csv(DATA_PATH)

    episode_ids = sorted(df["episode"].unique())
    if len(episode_ids) < 5:
        raise ValueError("Need more collected episodes before training the LSTM.")

    split_idx = int(0.8 * len(episode_ids))
    train_episode_ids = set(episode_ids[:split_idx])
    val_episode_ids = set(episode_ids[split_idx:])

    train_df = df[df["episode"].isin(train_episode_ids)].copy()
    val_df = df[df["episode"].isin(val_episode_ids)].copy()

    X_train, y_train = make_windows(train_df, cfg.history, cfg.horizon)
    X_val, y_val = make_windows(val_df, cfg.history, cfg.horizon)

    X_train, X_val, y_train, y_val, stats = normalize_train_val(X_train, X_val, y_train, y_val)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    print("Persistence baseline RMSE:", persistence_baseline_rmse(X_val, y_val))

    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TrafficDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficLSTM(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_checkpoint: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "stats": stats,
            }

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint saved.")

    torch.save(best_checkpoint, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()