# train/entry/train.py
"""
File: train/entry/train.py

ソースコードの役割:
本モジュールは、特徴量Parquetを読み込み、Transformerモデルの学習・検証・テストを実行します。
過学習を防ぐためのEarly Stopping、学習率スケジューラ（ReduceLROnPlateau）、および
最適なモデル重みの保存（チェックポイント）機能を統合管理する学習パイプラインのエントリーポイントです。

Walk-Forward Validation（ウォークフォワード検証）に対応し、時系列のスライディングウィンドウごとに
モデルの学習と評価を繰り返し実行します。
"""

import argparse
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path
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
    """
    1エポック分の学習を実行します。

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
    """
    1エポック分の検証（評価）を実行します。

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


def build_output_model_path(base_dir: str, target_file_path: str) -> str:
    """
    出力先モデルパスを、対象 parquet の日付・セッション種別から構築します。

    出力形式:
        data/entry/DAY/20YY/20YY-MM-DD.pth
        data/entry/NIGHT/20YY/20YY-MM-DD.pth

    Args:
        base_dir (str): ベースディレクトリ (例: "data")
        target_file_path (str): 対象ファイルパス (例: data/features/entry/DAY/2018/2018-01-04.parquet)

    Returns:
        str: 保存先の .pth パス
    """
    normalized = Path(target_file_path)

    # .../entry/<SESSION>/<YEAR>/<YYYY-MM-DD>.parquet を想定
    trade_date_str = normalized.stem
    year_str = normalized.parent.name
    session_type = normalized.parent.parent.name

    out_dir = os.path.join(
        base_dir,
        "entry",
        session_type,
        year_str,
    )
    os.makedirs(out_dir, exist_ok=True)

    return os.path.join(
        out_dir,
        f"{trade_date_str}.pth",
    )


def main():
    """コマンドライン引数を解析し、モデルの学習パイプライン全体を実行します。"""
    
    # ログ出力の設定
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M") + ".log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Transformerモデルの学習を実行します。")
    parser.add_argument("--feature-dir", type=str, default="data/features/entry/*/*/*.parquet", help="特徴量Parquetのパスパターン")
    parser.add_argument("--epochs", type=int, default=50, help="最大学習エポック数")
    parser.add_argument("--batch-size", type=int, default=256, help="バッチサイズ")
    parser.add_argument("--seq-len", type=int, default=60, help="シーケンス長（過去の足の本数）")
    parser.add_argument("--lr", type=float, default=1e-4, help="初期学習率")
    parser.add_argument("--patience", type=int, default=7, help="Early Stoppingが発動するまでのエポック数")
    parser.add_argument(
        "--out-base-dir", type=str, default="data",
        help="モデル出力ベースディレクトリ。実際の保存先は data/entry/<DAY|NIGHT>/<YEAR>/<DATE>.pth"
    )
    
    # データ分割の設定（例: 学習60日、検証20日、テスト5日）
    parser.add_argument("--train-days", type=int, default=60, help="学習データの日数")
    parser.add_argument("--valid-days", type=int, default=20, help="検証データの日数")
    parser.add_argument("--test-days", type=int, default=5, help="テストデータの日数")
    parser.add_argument("--start", type=str, default=None, help="処理を開始するテスト対象日 (YYYY-MM-DD)")
    
    args = parser.parse_args()

    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. データの取得
    all_files = glob.glob(args.feature_dir)
    if not all_files:
        logging.error(f"エラー: {args.feature_dir} に特徴量ファイルが見つかりません。")
        return

    logging.info(f"Found {len(all_files)} feature files. Starting Walk-Forward Validation...")
    
    # 時系列を 1 回だけ分割するのではなく、
    # test 窓を 1 日ずつ前へ進めながらループ処理します。
    total_required = args.train_days + args.valid_days + args.test_days
    sorted_feature_files = sorted(all_files)
    
    if len(sorted_feature_files) < total_required:
        raise ValueError(
            f"Not enough feature files. required={total_required}, actual={len(sorted_feature_files)}"
        )

    for end_idx in range(total_required, len(sorted_feature_files) + 1):
        window_files = sorted_feature_files[end_idx - total_required : end_idx]
        
        # 現在のウィンドウの末尾（test_days分）がテスト対象のファイル
        test_files = window_files[-args.test_days:]
        if not test_files:
            continue
            
        # 開始日が指定されている場合、テスト対象日がそれより前なら処理コストを省くためスキップ
        test_target_date = Path(test_files[0]).stem
        if args.start and test_target_date < args.start:
            continue
         
        # SeqLen未満のファイルに起因する例外などをキャッチし、該当ウィンドウをスキップする
        try:
            train_loader, valid_loader, test_loader, num_features = create_dataloaders(
                file_paths=window_files,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                train_days=args.train_days,
                valid_days=args.valid_days,
                test_days=args.test_days,
            )
        except ValueError as e:
            logging.warning(f"Skipping window ending at {window_files[-1]}: {e}")
            continue

        out_model_path = build_output_model_path(
            base_dir=args.out_base_dir,
            target_file_path=test_files[0],
        )

        logging.info(f"Current test target: {test_files[0]}")
        logging.info(f"Model output path: {out_model_path}")

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
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        # 5. 学習ループ
        best_valid_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            valid_loss = validate_epoch(model, valid_loader, criterion, device)
            
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start
            
            logging.info(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | "
                f"LR: {current_lr:.8f} | Time: {epoch_duration:.1f}s"
            )

            if current_lr != prev_lr:
                logging.info(f"  -> Learning rate changed: {prev_lr:.8f} -> {current_lr:.8f}")
            
            # Early Stopping とモデル保存の判定
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), out_model_path)
                logging.info(f"  -> Best model saved to: {out_model_path}")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= args.patience:
                logging.info("Early stopping triggered.")
                break

        # 6. テストデータでの最終評価
        logging.info("Loading best model for Test evaluation...")
        state_dict = torch.load(
            out_model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        test_loss = validate_epoch(model, test_loader, criterion, device)
        logging.info(f"Final Test Loss: {test_loss:.6f}")
        logging.info("-" * 80)


if __name__ == "__main__":
    main()
