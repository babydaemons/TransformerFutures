# train/entry/backtest.py
"""
File: train/entry/backtest.py

ソースコードの役割:
学習済みのTransformerモデルと価格モメンタムを組み合わせたハイブリッド戦略のバックテストを実行します。
モデルが「トレンド発生」を予測した時のみ、直近5分間（10本）の価格変化（モメンタム）の方向に従ってエントリーし、
往復コスト(15円)を考慮した実際の損益（エクイティカーブ）を計算・出力します。
新たにSL(Stop Loss)およびTP(Take Profit)のティックベースでのエグジットロジック、
および決済直後の連続エントリーを防止するクールダウン（待機期間）機能が組み込まれています。
また、特徴量生成前に保存しておいた絶対価格列を用いて、バックテスト用の価格を正確に復元します。
さらに、学習時に出力した edge 情報のJSONを参照し、指定したエッジ(%)を上回るモデルのみを
バックテスト対象に絞り込むフィルタリング機能を提供します。
加えて、1取引日ファイル内に DAY / NIGHT の両セッションが同居する構成に対応するため、
セッション別モデルを同時に読み込み、各バーの session_type に応じて対応モデルを切り替えて推論します。
バックテスト終了後には、matplotlibを用いてエクイティカーブ（資産曲線）を描画・保存します。
"""

import argparse
from datetime import datetime
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

# 既存モジュールのインポート
from train.entry.dataset import SessionSequenceDataset, compute_train_statistics
from train.entry.model import TimeSeriesTransformer


def simulate_trades(
    df: pl.DataFrame,
    prob_threshold: float,
    hold_horizon: int = 720,
    sl_ticks: int = 100,  # 損切り幅 (日経ミニなら 100 tick = 100円幅 = -10000円)
    tp_ticks: int = 250,  # 利食い幅 (日経ミニなら 250 tick = 250円幅 = +25000円)
) -> Tuple[float, List[dict]]:
    """
    1セッション分のデータフレームでトレードをシミュレーションします。

    Args:
        df: 予測確率('prob')と価格が含まれたDataFrame。
        prob_threshold: エントリーを許可するAIの予測確率の閾値。
        hold_horizon: 最大ホールド期間（足の本数）。
        sl_ticks: 損切りのティック数。
        tp_ticks: 利食いのティック数。

    Returns:
        セッションの合計損益(円)と、トレード履歴のリスト。
    """
    multiplier = 100  # 日経225ミニの乗数
    cost_yen = 80     # 往復コスト（円）

    # 状態管理
    position = 0      # 1: Long, -1: Short, 0: None
    entry_price = 0.0
    bars_held = 0
    cooldown_bars = 0  # 決済後の待機時間（足の本数）

    session_pnl = 0.0
    trades: List[dict] = []

    # PolarsからPythonのリストに変換して高速にイテレーション
    closes = df["raw_close"].to_list()
    opens = df["raw_open"].to_list()
    probs = df["prob"].to_list()
    thresholds = df["threshold"].to_list()
    momentum = df["momentum"].to_list()
    highs = df["raw_high"].to_list()
    lows = df["raw_low"].to_list()
    timestamps = df["bar_start_jst"].to_list()

    n = len(df)

    for i in range(n):
        # クールダウンのカウントダウン
        if cooldown_bars > 0:
            cooldown_bars -= 1

        # --- 1. エグジット判定 ---
        if position != 0:
            bars_held += 1

            # 現在の足の高値・安値で SL / TP に到達したかチェック
            hit_sl = False
            hit_tp = False
            exit_price = 0.0

            if position == 1:  # Long
                if lows[i] <= entry_price - (sl_ticks * 5.0):
                    hit_sl, exit_price = True, entry_price - (sl_ticks * 5.0)
                elif highs[i] >= entry_price + (tp_ticks * 5.0):
                    hit_tp, exit_price = True, entry_price + (tp_ticks * 5.0)
            elif position == -1:  # Short
                if highs[i] >= entry_price + (sl_ticks * 5.0):
                    hit_sl, exit_price = True, entry_price + (sl_ticks * 5.0)
                elif lows[i] <= entry_price - (tp_ticks * 5.0):
                    hit_tp, exit_price = True, entry_price - (tp_ticks * 5.0)

            # 時間切れ（ホールド限界）またはセッション終了
            time_exit = bars_held >= hold_horizon or i == n - 1

            if hit_sl or hit_tp or time_exit:
                if not (hit_sl or hit_tp):
                    # 時間切れの場合は次の足の始値（または現在の終値）で決済
                    exit_price = opens[i + 1] if i < n - 1 else closes[i]
                    reason = "TIME"
                else:
                    reason = "TP" if hit_tp else "SL"

                # 損益計算（価格差 × 建玉方向 × 乗数 - 往復コスト）
                trade_pnl = ((exit_price - entry_price) * position * multiplier) - cost_yen
                session_pnl += trade_pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamps[i],
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": trade_pnl,
                        "reason": reason,
                        "prob": entry_prob,
                        "direction_source": "momentum_10bars_sign",
                    }
                )

                position = 0
                bars_held = 0
                cooldown_bars = 60  # 決済後、30秒足60本(30分間)は再エントリーを禁止する

        # --- 2. エントリー判定 ---
        if position == 0 and cooldown_bars == 0 and i < n - 1:
            prob = probs[i]
            # JSONから取得した動的閾値がある場合はそれを使用、なければ固定値
            thr = thresholds[i] if thresholds[i] > 0.0 else prob_threshold

            if prob is not None and prob >= thr and momentum[i] is not None:
                # 次の足の始値で成行エントリー
                entry_price = opens[i + 1]
                entry_time = timestamps[i + 1]
                entry_prob = prob

                # 現在の方向判定はモデル出力ではなく、直近10本モメンタムの符号を使います。
                # 方向ソースを JSON にも保存し、モデルのエントリー判定と混同しないようにします。
                if momentum[i] > 0:
                    position = 1  # Long
                elif momentum[i] < 0:
                    position = -1  # Short

    return session_pnl, trades


def evaluate_window(
    train_files: List[str],
    test_files: List[str],
    models_to_run: Dict[str, dict],
    seq_len: int,
    device: torch.device,
    prob_threshold: float,
    sl_ticks: int,
    tp_ticks: int,
) -> List[dict]:
    """
    1つのWalk-Forwardウィンドウで必要なモデル群をロードし、テストデータでバックテストを行います。

    Args:
        train_files: スケーラー復元用の学習データファイルパスのリスト。
        test_files: 評価対象となるテストデータファイルパスのリスト。
        models_to_run: セッション種別をキー、モデル情報を値とする辞書。
        seq_len: モデル入力のシーケンス長。
        device: 推論に使用するデバイス。
        prob_threshold: エントリーを許可するAIの予測確率の閾値。
        sl_ticks: 損切りのティック数。
        tp_ticks: 利食いのティック数。

    Returns:
        指定されたウィンドウ内での全トレード履歴。
    """
    # スケーラー（平均・標準偏差）をTrainから復元
    means, stds, _ = compute_train_statistics(train_files, label_col="label_efficiency_240")

    # 正しい特徴量次元数を動的に取得するため、学習データ先頭ファイルからダミーの
    # データセットを構築して num_features を確定します。
    dummy_ds = SessionSequenceDataset(
        train_files[0],
        seq_len=seq_len,
        feature_means=means,
        feature_stds=stds,
    )
    num_features = dummy_ds.X_data.shape[1]

    # 存在するセッションのモデルをロードする。
    loaded_models: Dict[str, dict] = {}
    for session_type, meta in models_to_run.items():
        try:
            model = TimeSeriesTransformer(
                num_features=num_features,
                d_model=128,
                nhead=8,
                num_layers=3,
                dim_feedforward=256,
                dropout=0.2,
            )
            model.load_state_dict(torch.load(meta["path"], map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            loaded_models[session_type] = {"model": model, "threshold": meta["threshold"]}
        except Exception as e:
            logging.error(f"Failed to load {session_type} model: {e}")

    if not loaded_models:
        return []

    all_trades: List[dict] = []

    for test_file in test_files:
        df = pl.read_parquet(test_file)
        if len(df) <= seq_len:
            continue

        # データセットの作成と推論。
        # 1日ファイル内に DAY / NIGHT が混在し得るため、target_session="ALL" で全行を対象にする。
        dataset = SessionSequenceDataset(
            test_file,
            seq_len=seq_len,
            feature_means=means,
            feature_stds=stds,
            target_session="ALL",
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        # 各有効インデックス位置へセッション別モデルの予測値を直接書き戻す。
        # 推論不能な先頭部や対象外セッションの位置は 0.0 のまま残る。
        full_preds = np.zeros(len(df), dtype=np.float32)
        full_thresholds = np.zeros(len(df), dtype=np.float32)
        session_types = df["session_type"].to_list()

        with torch.no_grad():
            for batch_idx, (batch_x, _) in enumerate(loader):
                batch_x = batch_x.to(device)

                start_i = batch_idx * 256
                end_i = start_i + len(batch_x)
                batch_indices = dataset.valid_indices[start_i:end_i]

                for s_type, m_data in loaded_models.items():
                    model = m_data["model"]
                    thr = m_data["threshold"]
                    mask = [session_types[idx] == s_type for idx in batch_indices]
                    if not any(mask):
                        continue

                    probs = torch.sigmoid(model(batch_x)).cpu().numpy().flatten()
                    for i, idx in enumerate(batch_indices):
                        if session_types[idx] == s_type:
                            full_preds[idx] = probs[i]
                            full_thresholds[idx] = thr

        df = df.with_columns(
            [
                pl.Series("prob", full_preds, dtype=pl.Float32),
                pl.Series("threshold", full_thresholds, dtype=pl.Float32),
                # 事前に保存しておいた絶対価格を復元用に使用する。
                # 累積和ベースの復元で起こり得るオフバイワンやGAP起因の累積誤差を避ける。
                pl.col("raw_open_abs").alias("raw_open"),
                pl.col("raw_high_abs").alias("raw_high"),
                pl.col("raw_low_abs").alias("raw_low"),
                pl.col("raw_close_abs").alias("raw_close"),
            ]
        ).with_columns(
            [
                # トリガー用に直近5分間（10本）の価格変化（モメンタム）を計算する。
                (pl.col("raw_close") - pl.col("raw_close").shift(10)).alias("momentum"),
            ]
        )

        # シミュレーション実行
        _, trades = simulate_trades(
            df,
            prob_threshold=prob_threshold,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
        )
        all_trades.extend(trades)

    return all_trades


def main() -> None:
    """バックテスト実行のエントリーポイントです。"""
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

    parser = argparse.ArgumentParser(description="Transformerハイブリッド戦略のバックテスト")
    parser.add_argument("--feature-dir", type=str, default="data/features/entry/*/*.parquet")
    parser.add_argument("--model-dir", type=str, default="data/entry")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--valid-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--start", type=str, default=None, help="バックテスト開始日 YYYY-MM-DD")
    parser.add_argument("--prob-threshold", type=float, default=0.6, help="エントリーを許可するAIの確率閾値")
    parser.add_argument("--sl-ticks", type=int, default=10, help="損切り幅(ティック数: 1tick=5円)")
    parser.add_argument("--tp-ticks", type=int, default=20, help="利食い幅(ティック数: 1tick=5円)")
    parser.add_argument("--edge", type=float, default=None, help="指定したエッジ(%)を超える場合のみバックテストを実行する")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sorted_files = sorted(glob.glob(args.feature_dir))
    total_required = args.train_days + args.valid_days + args.test_days

    all_trades: List[dict] = []

    logging.info("Starting Backtest...")

    for end_idx in range(total_required, len(sorted_files) + 1):
        window_files = sorted_files[end_idx - total_required: end_idx]
        test_files = window_files[-args.test_days:]
        train_files = window_files[:args.train_days]

        test_date = Path(test_files[0]).stem
        if args.start and test_date < args.start:
            continue

        year_str = os.path.basename(os.path.dirname(test_files[0]))

        # 1取引日内の DAY / NIGHT セッションに対応するモデルを収集する。
        models_to_run: Dict[str, dict] = {}
        for session_type in ["DAY", "NIGHT"]:
            model_path = os.path.join(args.model_dir, year_str, f"{test_date}-{session_type}.pth")
            json_path = os.path.join(args.model_dir, year_str, f"{test_date}-{session_type}.json")

            if not os.path.exists(model_path):
                continue

            prob_thr = args.prob_threshold
            # --- エッジによるフィルタリング ---
            if args.edge is not None:
                if not os.path.exists(json_path):
                    continue

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        record = json.loads(f.read().strip())
                        edge_val = record.get("edge", 0.0)
                        prob_thr = record.get("prob_threshold", args.prob_threshold)

                    if edge_val <= args.edge:
                        logging.info(
                            f"Skipping {test_date} ({session_type}): "
                            f"Edge ({edge_val}%) <= Threshold"
                        )
                        continue
                except Exception:
                    continue

            models_to_run[session_type] = {"path": model_path, "threshold": prob_thr}

        if not models_to_run:
            continue

        logging.info(f"Evaluating Date: {test_date} ...")
        trades = evaluate_window(
            train_files,
            test_files,
            models_to_run,
            args.seq_len,
            device,
            args.prob_threshold,
            args.sl_ticks,
            args.tp_ticks,
        )
        all_trades.extend(trades)

    # --- バックテスト結果の集計 ---
    if not all_trades:
        logging.info("No trades executed.")
        return

    total_trades = len(all_trades)
    winning_trades = [trade for trade in all_trades if trade["pnl"] > 0]
    losing_trades = [trade for trade in all_trades if trade["pnl"] <= 0]

    gross_profit = sum(trade["pnl"] for trade in winning_trades)
    gross_loss = abs(sum(trade["pnl"] for trade in losing_trades))
    net_profit = gross_profit - gross_loss

    win_rate = len(winning_trades) / total_trades * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 最大ドローダウンの計算
    cumulative_pnl = np.cumsum([trade["pnl"] for trade in all_trades])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns)

    logging.info("=" * 50)
    logging.info("          BACKTEST RESULTS          ")
    logging.info("=" * 50)
    logging.info(f"Total Trades : {total_trades}")
    logging.info(f"Win Rate     : {win_rate:.2f}% ({len(winning_trades)}W / {len(losing_trades)}L)")
    logging.info(f"Profit Factor: {profit_factor:.3f}")
    logging.info(f"Gross Profit : +{gross_profit:,.0f} JPY")
    logging.info(f"Gross Loss   : -{gross_loss:,.0f} JPY")
    logging.info(f"Net Profit   : {net_profit:+,.0f} JPY")
    logging.info(f"Max Drawdown : {max_drawdown:,.0f} JPY")
    logging.info("=" * 50)

    # --- エクイティカーブ（資産曲線）の描画と保存 ---
    try:
        import matplotlib.pyplot as plt

        times = [trade["exit_time"] for trade in all_trades]
        plt.figure(figsize=(12, 6))
        plt.plot(times, cumulative_pnl, label="Equity Curve", color="blue", linewidth=1.5)
        plt.title(f"Transformer + Momentum Strategy (PF: {profit_factor:.2f} / Net: {net_profit:,.0f} JPY)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL (JPY)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(args.model_dir, "equity_curve.png")
        plt.savefig(plot_path)
        logging.info(f"Equity curve saved to: {plot_path}")
    except ImportError:
        logging.warning("matplotlib is not installed. Skipping equity curve plot.")


if __name__ == "__main__":
    main()
