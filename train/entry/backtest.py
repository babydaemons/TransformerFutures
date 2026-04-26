# train/entry/backtest.py
"""
File: train/entry/backtest.py

ソースコードの役割:
学習済みのTransformerモデルと価格モメンタムを組み合わせたハイブリッド戦略のバックテストを実行します。
モデルが「トレンド発生」を予測した時のみ、指定本数の価格変化（モメンタム）の方向に従ってエントリーし、
予測確率の上昇幅、最小モメンタム幅、最大モメンタム幅、日次トレード数上限、確率上限で横ばい・ピークアウト・飛び乗り局面のノイズを抑制しながら、
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
from typing import Dict, List, Optional, Tuple

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
    sl_ticks: int = 100,
    tp_ticks: int = 250,
    min_prob_rise: float = 0.0,
    min_momentum_abs: float = 0.0,
    max_momentum_abs: float = 0.0,
    max_daily_trades: int = 0,
    max_prob: float = 1.0,
) -> Tuple[float, List[dict]]:
    """
    1日分のデータフレームでトレードをシミュレーションします。

    Args:
        df: 予測確率('prob')と価格が含まれたDataFrame。
        prob_threshold: エントリーを許可するAIの確率下限閾値。
        hold_horizon: 最大ホールド期間（足の本数）。
        sl_ticks: 損切りのティック数。
        tp_ticks: 利食いのティック数。
        min_prob_rise: 直近平均との差分で要求する最小の確率上昇幅。0.0以下なら無効。
        min_momentum_abs: 方向判定に使う最小モメンタム幅。0.0以下なら無効。
        max_momentum_abs: 飛び乗り抑制用の最大モメンタム幅。0.0以下なら無効。
        max_daily_trades: 1日(session_date × session_type)あたりの最大エントリー数。
            0 の場合は無制限。過剰エントリー抑制に使用します。
        max_prob: エントリーを許可するAIの確率上限閾値（デフォルト 1.0 = 上限なし）。
            高確率ゾーンでの逆張り的過適合を防ぎます。

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
    entry_session_key: Optional[str] = None  # セッション境界検出用

    session_pnl = 0.0
    trades: List[dict] = []
    daily_trade_count: dict = {}  # session_key -> count (max_daily_trades 制御用)

    # PolarsからPythonのリストに変換して高速にイテレーション
    closes = df["raw_close"].to_list()
    opens = df["raw_open"].to_list()
    probs = df["prob"].to_list()
    thresholds = df["threshold"].to_list()
    momentum = df["momentum"].to_list()
    prob_rise = df["prob_rise"].to_list()
    highs = df["raw_high"].to_list()
    lows = df["raw_low"].to_list()
    timestamps = df["bar_start_jst"].to_list()
    session_types = df["session_type"].to_list()
    session_dates = df["session_date_jst"].to_list()

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

            # セッション境界チェック: DAY→NIGHT またはファイル間をまたいだ場合に強制決済する。
            # 1日ファイルは [DAY 08:45-15:15] → [NIGHT 16:30-翌05:55] の順で格納されているが、
            # NIGHT バーの bar_start_jst は前日〜当日早朝のため、
            # そのままホールドすると exit_time < entry_time の時刻逆転が発生する。
            current_session_key = f"{session_dates[i]}_{session_types[i]}"
            session_changed = (
                entry_session_key is not None
                and current_session_key != entry_session_key
            )
            time_exit = bars_held >= hold_horizon or i == n - 1 or session_changed

            if hit_sl or hit_tp or time_exit:
                if not (hit_sl or hit_tp):
                    # 時間切れまたはセッション境界到達の場合は次の足の始値（または現在の終値）で決済
                    exit_price = opens[i + 1] if i < n - 1 else closes[i]
                    reason = "SESSION" if session_changed else "TIME"
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
                logging.info(
                    f"[TRADE] {entry_time} -> {timestamps[i]}"
                    f" | {'LONG ' if position == 1 else 'SHORT'}"
                    f" | entry={entry_price:,.0f}  exit={exit_price:,.0f}"
                    f" | {reason:<7}"
                    f" | {trade_pnl:+,.0f} JPY"
                    f" | prob={entry_prob:.3f}"
                    f" | bars={bars_held}"
                    f" | session={entry_session_key}"
                )

                position = 0
                bars_held = 0
                entry_session_key = None
                cooldown_bars = 60  # 決済後、30秒足60本(30分間)は再エントリーを禁止する

        # --- 2. エントリー判定 ---
        if position == 0 and cooldown_bars == 0 and i < n - 1:
            prob = probs[i]
            # JSON閾値が存在する場合でも、CLIで指定した固定閾値より緩くはしません。
            # --prob-threshold 0.80 を指定したのに、古いJSON閾値で過剰売買になる事故を防ぎます。
            dynamic_thr = thresholds[i] if thresholds[i] > 0.0 else 0.0
            thr = max(prob_threshold, dynamic_thr)

            prob_rise_ok = (
                min_prob_rise <= 0.0
                or (prob_rise[i] is not None and prob_rise[i] >= min_prob_rise)
            )
            momentum_abs = abs(momentum[i]) if momentum[i] is not None else None
            min_momentum_ok = (
                min_momentum_abs <= 0.0
                or (momentum_abs is not None and momentum_abs >= min_momentum_abs)
            )
            max_momentum_ok = (
                max_momentum_abs <= 0.0
                or (momentum_abs is not None and momentum_abs <= max_momentum_abs)
            )

            # セッションキー: 日次トレード数カウントの単位 (session_date × session_type)
            current_key = f"{session_dates[i]}_{session_types[i]}"
            under_daily_limit = (
                max_daily_trades == 0
                or daily_trade_count.get(current_key, 0) < max_daily_trades
            )

            if (
                prob is not None
                and prob >= thr
                and prob <= max_prob
                and momentum[i] is not None
                and prob_rise_ok
                and min_momentum_ok
                and max_momentum_ok
                and under_daily_limit
            ):
                # 次の足の始値で成行エントリー
                entry_price = opens[i + 1]
                entry_time = timestamps[i + 1]
                entry_prob = prob
                entry_session_key = f"{session_dates[i + 1]}_{session_types[i + 1]}"
                daily_trade_count[current_key] = daily_trade_count.get(current_key, 0) + 1

                # 現在の方向判定はモデル出力ではなく、直近モメンタムの符号を使います。
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
    hold_horizon: int,
    sl_ticks: int,
    tp_ticks: int,
    direction_lookback_bars: int,
    prob_rise_bars: int,
    min_prob_rise: float,
    min_momentum_abs: float,
    max_momentum_abs: float,
    max_daily_trades: int = 0,
    max_prob: float = 1.0,
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
        hold_horizon: 最大ホールド期間（足の本数）。ラベルの horizon と合わせること。
        sl_ticks: 損切りのティック数。
        tp_ticks: 利食いのティック数。
        direction_lookback_bars: 方向判定に使う価格モメンタムの参照本数。
        prob_rise_bars: 予測確率の上昇幅を計算する比較本数。
        min_prob_rise: エントリーに必要な予測確率の最小上昇幅。0.0以下なら無効。
        min_momentum_abs: エントリーに必要な最小モメンタム幅。0.0以下なら無効。
        max_momentum_abs: 飛び乗り抑制用の最大モメンタム幅。0.0以下なら無効。
        max_daily_trades: session_date × session_type あたりの最大エントリー数（0 = 無制限）。
        max_prob: エントリーを許可するAIの確率上限閾値（デフォルト 1.0 = 上限なし）。

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
                # トリガー用に直近の価格変化（モメンタム）を計算する。
                # .over() でセッション境界をまたがないよう制限する。
                (
                    pl.col("raw_close")
                    - pl.col("raw_close").shift(direction_lookback_bars).over(
                        ["session_date_jst", "session_type"]
                    )
                ).alias("momentum"),
                # 確率の絶対値だけでなく「予測確率が上昇している局面」に限定します。
                # これにより、ピークアウト後の遅い順張りエントリーを減らします。
                (
                    pl.col("prob")
                    - pl.col("prob").shift(prob_rise_bars).over(["session_date_jst", "session_type"])
                )
                .fill_null(0.0)
                .alias("prob_rise"),
            ]
        )

        # シミュレーション実行
        _, trades = simulate_trades(
            df,
            prob_threshold=prob_threshold,
            hold_horizon=hold_horizon,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            min_prob_rise=min_prob_rise,
            min_momentum_abs=min_momentum_abs,
            max_momentum_abs=max_momentum_abs,
            max_daily_trades=max_daily_trades,
            max_prob=max_prob,
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
    parser.add_argument("--prob-threshold", type=float, default=0.82, help="エントリーを許可するAIの確率閾値")
    parser.add_argument("--sl-ticks", type=int, default=20, help="損切り幅(ティック数: 1tick=5円)")
    parser.add_argument("--tp-ticks", type=int, default=50, help="利食い幅(ティック数: 1tick=5円)")
    parser.add_argument("--hold-horizon", type=int, default=240, help="最大ホールド期間（足の本数）。ラベル horizon=240 に合わせる")
    parser.add_argument("--edge", type=float, default=None, help="指定したエッジ(%)を超える場合のみバックテストを実行する")
    parser.add_argument("--direction-lookback-bars", type=int, default=20, help="方向判定に使うモメンタム本数。30秒足なら20本=10分")
    parser.add_argument("--prob-rise-bars", type=int, default=5, help="確率上昇を判定する比較本数。30秒足なら5本=2.5分")
    parser.add_argument("--min-prob-rise", type=float, default=0.0, help="エントリーに必要な予測確率の最小上昇幅。0以下なら無効")
    parser.add_argument("--min-momentum-abs", type=float, default=0.0, help="エントリーに必要な最小モメンタム幅。0以下なら無効")
    parser.add_argument("--max-momentum-abs", type=float, default=80.0, help="飛び乗り抑制用の最大モメンタム幅。0以下なら無効")
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=0,
        help=(
            "session_date × session_type あたりの最大エントリー数（デフォルト 0 = 無制限）。"
            "例: 2 を指定すると同一セッション内で2件まで。過剰エントリーを防ぐ。"
        ),
    )
    parser.add_argument(
        "--max-prob",
        type=float,
        default=1.0,
        help=(
            "エントリーを許可するAIの確率上限（デフォルト 1.0 = 上限なし）。"
            "高確率ゾーンの過適合抑制に使用。例: 0.87 を指定すると prob>=0.87 はスキップ。"
        ),
    )
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
                        # JSON閾値は参考値。CLI閾値より緩い場合は採用しません。
                        prob_thr = max(args.prob_threshold, record.get("prob_threshold", args.prob_threshold))

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
            args.hold_horizon,
            args.sl_ticks,
            args.tp_ticks,
            args.direction_lookback_bars,
            args.prob_rise_bars,
            args.min_prob_rise,
            args.min_momentum_abs,
            args.max_momentum_abs,
            max_daily_trades=args.max_daily_trades,
            max_prob=args.max_prob,
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
