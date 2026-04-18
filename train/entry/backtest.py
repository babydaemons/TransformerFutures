# train/entry/backtest.py
"""
File: train/entry/backtest.py

ソースコードの役割:
学習済みのTransformerモデルと価格モメンタムを組み合わせたハイブリッド戦略のバックテストを実行します。
モデルが「トレンド発生」を予測した時のみ、直近5分間（10本）の価格変化（モメンタム）の方向に従ってエントリーし、
往復コスト(15円)を考慮した実際の損益（エクイティカーブ）を計算・出力します。
新たにSL(Stop Loss)およびTP(Take Profit)のティックベースでのエグジットロジック、
および決済直後の連続エントリーを防止するクールダウン（待機期間）機能が組み込まれています。
また、推論用に階差（差分）に変換された価格データを累積和で復元し、正確な価格計算を行います。
バックテスト終了後には、matplotlibを用いてエクイティカーブ（資産曲線）を描画・保存します。
"""

import argparse
from datetime import datetime
import glob
import os
import logging
from typing import List, Dict, Tuple
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

# 既存モジュールのインポート
from train.entry.model import TimeSeriesTransformer
from train.entry.dataset import SessionSequenceDataset, compute_train_statistics


def simulate_trades(
    df: pl.DataFrame, 
    prob_threshold: float, 
    hold_horizon: int = 240,
    sl_ticks: int = 10,  # 損切り幅 (日経ミニなら 10 tick = 50円幅 = -5000円)
    tp_ticks: int = 20   # 利食い幅 (日経ミニなら 20 tick = 100円幅 = +10000円)
) -> Tuple[float, List[dict]]:
    """
    1セッション分のデータフレームでトレードをシミュレーションします。
    
    Args:
        df (pl.DataFrame): 予測確率('prob')と価格が含まれたDataFrame
        prob_threshold (float): エントリーを許可するAIの予測確率の閾値
        hold_horizon (int, optional): 最大ホールド期間（足の本数）. Defaults to 240.
        sl_ticks (int, optional): 損切りのティック数. Defaults to 10.
        tp_ticks (int, optional): 利食いのティック数. Defaults to 20.
        
    Returns:
        Tuple[float, List[dict]]: セッションの合計損益(円)と、トレード履歴のリスト
    """
    MULTIPLIER = 100  # 日経225ミニの乗数
    COST_YEN = 15     # 往復コスト（円）
    
    # 状態管理
    position = 0      # 1: Long, -1: Short, 0: None
    entry_price = 0.0
    bars_held = 0
    cooldown_bars = 0 # 決済後の待機時間（足の本数）
    
    session_pnl = 0.0
    trades = []
    
    # PolarsからPythonのリストに変換して高速にイテレーション
    closes = df["raw_close"].to_list()
    opens = df["raw_open"].to_list()
    probs = df["prob"].to_list()
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
            
            if position == 1: # Long
                if lows[i] <= entry_price - (sl_ticks * 5.0):
                    hit_sl, exit_price = True, entry_price - (sl_ticks * 5.0)
                elif highs[i] >= entry_price + (tp_ticks * 5.0):
                    hit_tp, exit_price = True, entry_price + (tp_ticks * 5.0)
            elif position == -1: # Short
                if highs[i] >= entry_price + (sl_ticks * 5.0):
                    hit_sl, exit_price = True, entry_price + (sl_ticks * 5.0)
                elif lows[i] <= entry_price - (tp_ticks * 5.0):
                    hit_tp, exit_price = True, entry_price - (tp_ticks * 5.0)
            
            # 時間切れ（ホールド限界）またはセッション終了
            time_exit = bars_held >= hold_horizon or i == n - 1
            
            if hit_sl or hit_tp or time_exit:
                if not (hit_sl or hit_tp):
                    # 時間切れの場合は次の足の始値（または現在の終値）で決済
                    exit_price = opens[i+1] if i < n - 1 else closes[i]
                    reason = "TIME"
                else:
                    reason = "TP" if hit_tp else "SL"
                
                # 損益計算（ティック差分 × 乗数 - コスト）
                trade_pnl = ((exit_price - entry_price) * position * MULTIPLIER) - COST_YEN
                session_pnl += trade_pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": trade_pnl,
                    "reason": reason,
                    "prob": entry_prob
                })
                
                position = 0
                bars_held = 0
                cooldown_bars = 60 # 決済後、30秒足60本(30分間)は再エントリーを禁止する
                
        # --- 2. エントリー判定 ---
        if position == 0 and cooldown_bars == 0 and i < n - 1:
            prob = probs[i]
            # 確率が閾値を超えており、かつモメンタムが計算できている場合
            if prob is not None and prob >= prob_threshold and momentum[i] is not None:
                # 次の足の始値で成行エントリー
                entry_price = opens[i+1]
                entry_time = timestamps[i+1]
                entry_prob = prob
                
                if momentum[i] > 0:
                    position = 1  # Long
                elif momentum[i] < 0:
                    position = -1 # Short

    return session_pnl, trades


def evaluate_window(
    train_files: List[str],
    test_files: List[str],
    model_path: str,
    seq_len: int,
    device: torch.device,
    prob_threshold: float,
    sl_ticks: int,
    tp_ticks: int
) -> List[dict]:
    """
    1つのWalk-Forwardウィンドウでモデルをロードし、テストデータでバックテストを行います。
    
    Args:
        train_files (List[str]): スケーラー復元用の学習データファイルパスのリスト
        test_files (List[str]): 評価対象となるテストデータファイルパスのリスト
        model_path (str): ロードする学習済みPyTorchモデルのパス
        seq_len (int): モデル入力のシーケンス長
        device (torch.device): 推論に使用するデバイス
        prob_threshold (float): エントリーを許可するAIの予測確率の閾値
        sl_ticks (int): 損切りのティック数
        tp_ticks (int): 利食いのティック数
        
    Returns:
        List[dict]: 指定されたウィンドウ内での全トレード履歴
    """
    
    # スケーラー（平均・標準偏差）をTrainから復元
    means, stds, _ = compute_train_statistics(train_files, label_col="label_efficiency_240")
    
    # モデルのロード
    dummy_ds = SessionSequenceDataset(train_files[0], seq_len=seq_len, feature_means=means, feature_stds=stds)
    num_features = dummy_ds.X_data.shape[1]
    
    model = TimeSeriesTransformer(num_features=num_features, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2)
    
    # 破損したモデルファイルのロードに対する例外ハンドリングを追加
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        logging.error(f"Failed to load model {model_path}. File might be corrupted: {e}")
        return []  # 破損している場合はトレードなしとしてスキップ
        
    model.to(device)
    model.eval()

    all_trades = []
    
    for test_file in test_files:
        df = pl.read_parquet(test_file)
        if len(df) <= seq_len:
            continue
            
        # データセットの作成と推論
        dataset = SessionSequenceDataset(test_file, seq_len=seq_len, feature_means=means, feature_stds=stds)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(probs)
                
        # 予測値をDataFrameに結合（最初の seq_len - 1 行は予測不能なので None で埋める）
        pad = [None] * (seq_len - 1)
        full_preds = pad + preds
        
        df = df.with_columns([
            pl.Series("prob", full_preds, dtype=pl.Float32),
            # 特徴量生成時に差分(階差)に変換されているため、累積和(cum_sum)で相対的な元の価格推移に復元する
            pl.col("open").cum_sum().alias("raw_open"),
            pl.col("high").cum_sum().alias("raw_high"),
            pl.col("low").cum_sum().alias("raw_low"),
            pl.col("close").cum_sum().alias("raw_close"),
        ]).with_columns([
            # トリガー用に直近5分間（10本）の価格変化（モメンタム）を計算
            (pl.col("raw_close") - pl.col("raw_close").shift(10)).alias("momentum")
        ])
        
        # シミュレーション実行
        _, trades = simulate_trades(df, prob_threshold=prob_threshold, sl_ticks=sl_ticks, tp_ticks=tp_ticks)
        all_trades.extend(trades)
        
    return all_trades


def main():
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

    """バックテスト実行のエントリーポイント。"""
    parser = argparse.ArgumentParser(description="Transformerハイブリッド戦略のバックテスト")
    parser.add_argument("--feature-dir", type=str, default="data/features/entry/*/*/*.parquet")
    parser.add_argument("--model-dir", type=str, default="data/entry")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--valid-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--start", type=str, default=None, help="バックテスト開始日 YYYY-MM-DD")
    parser.add_argument("--prob-threshold", type=float, default=0.6, help="エントリーを許可するAIの確率閾値")
    parser.add_argument("--sl-ticks", type=int, default=10, help="損切り幅(ティック数: 1tick=5円)")
    parser.add_argument("--tp-ticks", type=int, default=20, help="利食い幅(ティック数: 1tick=5円)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sorted_files = sorted(glob.glob(args.feature_dir))
    total_required = args.train_days + args.valid_days + args.test_days
    
    all_trades = []

    logging.info("Starting Backtest...")
    
    for end_idx in range(total_required, len(sorted_files) + 1):
        window_files = sorted_files[end_idx - total_required : end_idx]
        test_files = window_files[-args.test_days:]
        train_files = window_files[:args.train_days]
        
        test_date = os.path.splitext(os.path.basename(test_files[0]))[0]
        if args.start and test_date < args.start:
            continue
            
        # 対応する学習済みモデルのパスを推測（train.pyと同じロジック）
        session_type = os.path.basename(os.path.dirname(os.path.dirname(test_files[0])))
        year_str = os.path.basename(os.path.dirname(test_files[0]))
        model_path = os.path.join(args.model_dir, session_type, year_str, f"{test_date}.pth")
        
        if not os.path.exists(model_path):
            logging.warning(f"Model not found for {test_date}: {model_path}")
            continue
            
        logging.info(f"Evaluating Date: {test_date} ...")
        trades = evaluate_window(train_files, test_files, model_path, args.seq_len, device, args.prob_threshold, args.sl_ticks, args.tp_ticks)
        all_trades.extend(trades)

    # --- バックテスト結果の集計 ---
    if not all_trades:
        logging.info("No trades executed.")
        return

    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t["pnl"] > 0]
    losing_trades = [t for t in all_trades if t["pnl"] <= 0]
    
    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    net_profit = gross_profit - gross_loss
    
    win_rate = len(winning_trades) / total_trades * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # 最大ドローダウンの計算
    cumulative_pnl = np.cumsum([t["pnl"] for t in all_trades])
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
        
        times = [t["exit_time"] for t in all_trades]
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
