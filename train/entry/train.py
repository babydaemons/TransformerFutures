# train/entry/train.py
"""
File: train/entry/train.py

ソースコードの役割:
本モジュールは、特徴量Parquetを読み込み、Transformerモデルの学習・検証・テストを実行します。
過学習を防ぐためのEarly Stopping、学習率スケジューラ（ReduceLROnPlateau）、および
最適なモデル重みの保存（チェックポイント）機能を統合管理する学習パイプラインのエントリーポイントです。

本戦略は回帰ではなく、優位性の高いトレンドが発生するかどうかの「分類タスク（Classification）」として
学習を行います。Walk-Forward Validation（ウォークフォワード検証）に対応し、
時系列のスライディングウィンドウごとにモデルの学習と評価を繰り返し実行します。
テスト時には、ROC AUCスコアや上位パーセンタイルにおけるトレンド発生確率といった
実運用における優位性（エッジ）を評価します。

入力想定:
- data/features/entry/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.parquet

出力想定:
- data/entry/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.pth
"""

import argparse
import glob
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score
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
        criterion (nn.Module): 損失関数（分類タスクのためBCEWithLogitsLossを想定）。
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

        # 損失計算: outputsは (Batch, 1), batch_yは (Batch,) なので次元を合わせる
        target = batch_y.unsqueeze(-1) if batch_y.dim() == 1 else batch_y
        loss = criterion(outputs, target)

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

            # 損失計算: 次元を合わせて計算
            target = batch_y.unsqueeze(-1) if batch_y.dim() == 1 else batch_y
            loss = criterion(outputs, target)

            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


def build_output_model_path(
    base_dir: str,
    year_str: str,
    date_str: str,
    session_type: str,
) -> str:
    """
    出力先モデルパスを、年・日付・セッション種別から構築します。

    出力形式:
        <base_dir>/entry/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.pth

    Args:
        base_dir (str): ベースディレクトリ。
        year_str (str): 年文字列。
        date_str (str): 取引日 (YYYY-MM-DD)。
        session_type (str): セッション種別 (DAY または NIGHT)。

    Returns:
        str: 保存先の .pth パス。
    """
    out_dir = Path(base_dir) / "entry" / year_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{date_str}-{session_type}.pth")


def main():
    """コマンドライン引数を解析し、モデルの学習・テストパイプライン全体を実行します。"""

    # ==========================================
    # 1. ログと引数の設定
    # ==========================================
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M") + ".log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(description="Transformerモデルの学習を実行します。")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="data/features/entry/*/*.parquet",
        help="特徴量Parquetのパスパターン",
    )
    parser.add_argument("--epochs", type=int, default=50, help="最大学習エポック数")
    parser.add_argument("--batch-size", type=int, default=4096, help="バッチサイズ")
    parser.add_argument("--seq-len", type=int, default=60, help="シーケンス長（過去の足の本数）")
    parser.add_argument("--lr", type=float, default=1e-4, help="初期学習率")
    parser.add_argument("--patience", type=int, default=7, help="Early Stoppingが発動するまでのエポック数")
    parser.add_argument(
        "--out-base-dir",
        type=str,
        default="data",
        help="モデル出力ベースディレクトリ。実際の保存先は data/entry/<YEAR>/<DATE>-<DAY|NIGHT>.pth",
    )

    # データ分割の設定（例: 学習60日、検証20日、テスト5日）
    parser.add_argument("--train-days", type=int, default=60, help="学習データの日数")
    parser.add_argument("--valid-days", type=int, default=20, help="検証データの日数")
    parser.add_argument("--test-days", type=int, default=5, help="テストデータの日数")
    parser.add_argument("--start", type=str, default=None, help="処理を開始するテスト対象日 (YYYY-MM-DD)")

    args = parser.parse_args()

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ==========================================
    # 2. データの取得とWalk-Forward Validationの準備
    # ==========================================
    all_feature_files = sorted(glob.glob(args.feature_dir))
    if not all_feature_files:
        logging.error(f"エラー: {args.feature_dir} に特徴量ファイルが見つかりません。")
        return

    total_required = args.train_days + args.valid_days + args.test_days
    logging.info(f"Found {len(all_feature_files)} feature files. Starting Walk-Forward Validation...")

    # DAY と NIGHT でループを分けることで、学習期間内に他方のセッションが混ざらないようにする
    for current_session in ["DAY", "NIGHT"]:
        # 指定されたセッションのファイルのみを抽出
        session_files = [
            file_path
            for file_path in all_feature_files
            if file_path.endswith(f"-{current_session}.parquet")
        ]

        if len(session_files) < total_required:
            logging.info(
                f"Skipping {current_session} session: "
                f"Not enough files (Found {len(session_files)}, need {total_required})."
            )
            continue

        logging.info(f"=== Starting Walk-Forward Validation for {current_session} session ===")

        for end_idx in range(total_required, len(session_files) + 1):
            window_files = session_files[end_idx - total_required: end_idx]

            # 現在のウィンドウの末尾がテスト対象
            test_files = window_files[-args.test_days:]
            if not test_files:
                continue

            # ファイル名から日付とセッションを再確認（rsplitを使用）
            test_file_stem = Path(test_files[0]).stem
            test_target_date, session_type = test_file_stem.rsplit("-", 1)

            # 開始日が指定されている場合、テスト対象日がそれより前ならスキップ
            if args.start and test_target_date < args.start:
                continue

            try:
                # 同一セッションのファイル群からデータローダーを作成
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

            year_str = Path(test_files[0]).parts[-2]
            out_model_path = build_output_model_path(
                args.out_base_dir,
                year_str,
                test_target_date,
                session_type,
            )

            logging.info(f"[{session_type}] Current test target: {test_files[0]}")
            logging.info(f"Model output path: {out_model_path}")

            # ==========================================
            # 3. モデルの初期化
            # ==========================================
            model = TimeSeriesTransformer(
                num_features=num_features,
                d_model=128,          # モデルの表現力。大きすぎると過学習しやすくなります
                nhead=8,              # Multi-Head Attentionの数
                num_layers=3,         # Encoder層の深さ
                dim_feedforward=256,
                dropout=0.2           # 金融データはノイズが多いのでDropoutは強めに設定
            ).to(device)

            # ==========================================
            # 4. 損失関数とオプティマイザ
            # ==========================================
            # 分類タスク（強いトレンドが発生するかどうか）なので BCEWithLogitsLoss を使用します
            # 不均衡データ（トレンド発生確率が低い）対策としてポジティブサンプルに重みを付けます
            pos_weight = torch.tensor([4.0]).to(device)  # 発生確率が約20%なら4.0程度が目安
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

            # 検証Lossが改善しなくなった場合に学習率を1/2に下げる
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            # ==========================================
            # 5. 学習ループ
            # ==========================================
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

            # ==========================================
            # 6. テストデータでの最終評価
            # ==========================================
            logging.info("Loading best model for Test evaluation...")
            state_dict = torch.load(
                out_model_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

            # ====== トレード向けのビジネス指標評価 ======
            model.eval()
            test_loss = 0.0
            all_preds = []
            all_trues = []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)

                    target = batch_y.unsqueeze(-1) if batch_y.dim() == 1 else batch_y
                    loss = criterion(outputs, target)
                    test_loss += loss.item() * batch_x.size(0)

                    # ロジットをSigmoidに通して確率(0.0~1.0)に変換して保存
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    all_preds.extend(probs)
                    all_trues.extend(batch_y.cpu().numpy().flatten())

            test_loss /= len(test_loader.dataset)
            logging.info(f"Final Test Loss: {test_loss:.6f}")

            preds_np = np.array(all_preds)
            trues_np = np.array(all_trues)

            # 予測のエッジ（優位性）の定量的評価
            if len(preds_np) > 1:
                # ROC AUCの計算 (正例と負例が両方存在する場合のみ)
                if len(np.unique(trues_np)) > 1:
                    auc = roc_auc_score(trues_np, preds_np)
                    logging.info(f"# ROC AUC Score: {auc:.4f}")

                threshold = np.percentile(preds_np, 80)  # 上位20%の予測スコア閾値
                high_conf_idx = preds_np >= threshold

                baseline_prob = trues_np.mean()
                top20_prob = trues_np[high_conf_idx].mean() if high_conf_idx.sum() > 0 else 0.0

                logging.info(f"# Baseline Trend Prob: {baseline_prob * 100:.2f}%")
                logging.info(f"# Top 20% Trend Prob:  {top20_prob * 100:.2f}%")
                logging.info(f"# Edge (Improvement):  {(top20_prob - baseline_prob) * 100:.2f}%")

            logging.info("-" * 80)


if __name__ == "__main__":
    main()
