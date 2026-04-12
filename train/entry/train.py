# train/entry/train.py
"""
File: train/entry/train.py

ソースコードの役割:
本モジュールは、特徴量Parquetを読み込み、Transformerモデルの学習・検証・テストを実行します。
過学習を防ぐためのEarly Stopping、学習率スケジューラ（ReduceLROnPlateau）、および
最適なモデル重みの保存（チェックポイント）機能を統合管理する学習パイプラインのエントリーポイントです。
"""

import argparse
import glob
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .dataset import create_dataloaders
from .model import TimeSeriesTransformer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """1エポック分の学習を実行します。

    Args:
        model (nn.Module): 学習対象のPyTorchモデル。
        dataloader (DataLoader): 学習用データのDataLoader。
        criterion (nn.Module): 損失関数。
        optimizer (optim.Optimizer): 最適化アルゴリズム。
        device (torch.device): 演算に使用するデバイス (CPU/GPU)。

    Returns:
        float: 1エポックの平均学習損失 (Train Loss)。
    """
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(batch_x)
        
        # 損失計算 (Batch, 1) と (Batch, 1) の誤差
        loss = criterion(outputs, batch_y)
        
        # 逆伝播と重み更新
        loss.backward()
        
        # 勾配爆発を防ぐためのクリッピング（Transformerの学習安定化に必須）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        
    return total_loss / len(dataloader.dataset)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """1エポック分の検証（評価）を実行します。

    Args:
        model (nn.Module): 評価対象のPyTorchモデル。
        dataloader (DataLoader): 検証またはテスト用データのDataLoader。
        criterion (nn.Module): 損失関数。
        device (torch.device): 演算に使用するデバイス (CPU/GPU)。

    Returns:
        float: 1エポックの平均検証損失 (Validation/Test Loss)。
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            
    return total_loss / len(dataloader.dataset)


def main():
    """コマンドライン引数を解析し、モデルの学習パイプライン全体を実行します。"""
    parser = argparse.ArgumentParser(description="Transformerモデルの学習を実行します。")
    parser.add_argument("--feature-dir", type=str, default="data/features/entry/*/*/*.parquet", help="特徴量Parquetのパスパターン")
    parser.add_argument("--epochs", type=int, default=50, help="最大学習エポック数")
    parser.add_argument("--batch-size", type=int, default=256, help="バッチサイズ")
    parser.add_argument("--seq-len", type=int, default=60, help="シーケンス長（過去の足の本数）")
    parser.add_argument("--lr", type=float, default=1e-4, help="初期学習率")
    parser.add_argument("--patience", type=int, default=7, help="Early Stoppingが発動するまでのエポック数")
    parser.add_argument("--out-model-path", type=str, default="data/models/transformer_entry.pth", help="モデル重みの保存先")
    
    # データ分割の設定（例: 学習60日、検証20日、テスト5日）
    parser.add_argument("--train-days", type=int, default=60, help="学習データの日数")
    parser.add_argument("--valid-days", type=int, default=20, help="検証データの日数")
    parser.add_argument("--test-days", type=int, default=5, help="テストデータの日数")
    
    args = parser.parse_args()

    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データの取得とDataLoaderの作成
    all_files = glob.glob(args.feature_dir)
    if not all_files:
        print(f"エラー: {args.feature_dir} に特徴量ファイルが見つかりません。")
        return

    print(f"Found {len(all_files)} feature files. Constructing dataloaders...")
    train_loader, valid_loader, test_loader, num_features = create_dataloaders(
        file_paths=all_files,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_days=args.train_days,
        valid_days=args.valid_days,
        test_days=args.test_days
    )
    print(f"Input features dimension: {num_features}")

    # 3. モデルの初期化
    model = TimeSeriesTransformer(
        num_features=num_features,
        d_model=128,          # モデルの表現力。大きすぎると過学習しやすくなります
        nhead=8,              # Multi-Head Attentionの数
        num_layers=3,         # Encoder層の深さ
        dim_feedforward=256, 
        dropout=0.2           # 金融データはノイズが多いのでDropoutは強め(0.2~0.3)に設定
    ).to(device)

    # 4. 損失関数とオプティマイザ
    # 効率比の予測（回帰）なので Huber Loss (SmoothL1Loss) を使用し、外れ値への耐性を高めます
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 検証Lossが改善しなくなった場合に学習率を1/2に下げる
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 5. 学習ループ
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(args.out_model_path), exist_ok=True)

    print("--- Training Started ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate_epoch(model, valid_loader, criterion, device)
        
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | "
              f"LR: {current_lr:.8f} | "
              f"Time: {epoch_duration:.1f}s")

        if current_lr != prev_lr:
            print(
                f"  -> Learning rate changed: {prev_lr:.8f} -> {current_lr:.8f}"
            )
        
        # Early Stopping とモデル保存の判定
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            # ベストな重みを保存
            torch.save(model.state_dict(), args.out_model_path)
            print(f"  -> Best model saved! (Valid Loss: {best_valid_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered. No improvement for {args.patience} epochs.")
                break

    total_time = time.time() - start_time
    print(f"--- Training Completed in {total_time / 60:.2f} mins ---")

    # 6. テストデータでの最終評価
    print("\nLoading best model for Test evaluation...")
    state_dict = torch.load(
        args.out_model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
