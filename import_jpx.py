# importer/import_jpx.py
"""
File: importer/import_jpx.py

ソースコードの役割:
JPXデータクラウド形式の日経平均先物（NK225）歩み値データをインポートします。
期近限月の選別、Tick Testによる売買方向推定を行い、1分足集計を行ったのち、
Volume ProfileからPoint of Control (POC) 特徴量を算出し、階層化されたParquet形式で保存します。
"""

import glob
import os
import re
import sys
from typing import List, Optional

import polars as pl

# 親ディレクトリの config.py を読み込むためのパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg, BAR_SECONDS


def load_jpx_ticks_from_tsv(file_path: str) -> pl.DataFrame:
    """
    JPX形式のTSVから歩み値を読み込み、売買方向を推定して返します。

    Args:
        file_path (str): 読み込むTSVファイルのパス

    Returns:
        pl.DataFrame: 処理済みの歩み値データフレーム。ファイルが存在しない・エラーの場合は空のDataFrameを返します。

    Notes:
        後段の1分足集計で `max_trade_size` や `avg_trade_size` を計算できるよう、
        元の約定数量 `trade_volume` はここで捨てずに保持します。
        JPX歩み値には約定数量 `Trade_Volume` が含まれるため、
        インポート時点で保持しておくのが自然です。
    """
    if not os.path.exists(file_path):
        return pl.DataFrame()

    try:
        ldf = pl.read_csv(
            file_path,
            separator="\t",
            has_header=True,
            dtypes={
                "trade_date": pl.Utf8,
                "time": pl.Utf8,
                "trade_price": pl.Float64,
                "trade_volume": pl.Int64,
                "contract_month": pl.Int64,
                "price_type": pl.Utf8,
            },
        )

        ldf = ldf.with_columns(
            (pl.col("trade_date") + pl.col("time").str.zfill(9))
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S%3f", strict=False)
            .alias("trade_ts")
        ).drop_nulls(subset=["trade_ts"])

        # 期近フィルタリング
        near_cm_df = ldf.group_by("trade_date").agg(
            pl.col("contract_month").min().alias("min_cm")
        )
        ldf = ldf.join(near_cm_df, on="trade_date").filter(
            pl.col("contract_month") == pl.col("min_cm")
        )

        # Tick Testによる売買方向推定
        # `trade_date` は JPX の「1取引日」単位（夜間開始〜翌日中終了）なので、
        # 差分計算・forward fill もこの単位で閉じる。
        # これにより、前取引日の終値から当日の寄付きへ跨るギャップで
        # 売買方向が汚染されることを防ぐ。
        ldf = ldf.filter(pl.col("price_type") == "N").sort(["trade_date", "trade_ts"])
        ldf = ldf.with_columns(
            pl.col("trade_price").diff().over("trade_date").alias("price_diff")
        )
        ldf = (
            ldf.with_columns(
                pl.when(pl.col("price_diff") > 0)
                .then(1)
                .when(pl.col("price_diff") < 0)
                .then(-1)
                .otherwise(None)
                .alias("direction_raw")
            )
            .with_columns(
                pl.col("direction_raw")
                .forward_fill()
                .over("trade_date")
                .fill_null(0)
                .cast(pl.Int8)
                .alias("direction")
            )
            .drop("direction_raw")
        )

        ldf = ldf.with_columns(
            [
                pl.when(pl.col("direction") == 1)
                .then(pl.col("trade_volume"))
                .otherwise(0)
                .alias("buy_vol"),
                pl.when(pl.col("direction") == -1)
                .then(pl.col("trade_volume"))
                .otherwise(0)
                .alias("sell_vol"),
            ]
        )

        return ldf.select(
            [
                "trade_ts",
                pl.col("trade_price").alias("price"),
                "trade_volume",
                "buy_vol",
                "sell_vol",
            ]
        )

    except Exception as e:
        print(f"  [Error] {os.path.basename(file_path)} の処理中にエラー: {e}")
        return pl.DataFrame()


def resample_to_bars(tick_df: pl.DataFrame, interval_sec: int) -> pl.DataFrame:
    """
    歩み値データを指定秒足に集約します。

    Args:
        tick_df (pl.DataFrame): 歩み値のデータフレーム
        interval_sec (int): 集約する秒数 (例: 60なら1分足)

    Returns:
        pl.DataFrame: OHLCVや需給系特徴量を含むバーデータ

    Notes:
        学習設定 `FeatureConfig.continuous_cols` で要求される需給系特徴量のうち、
        1本のバーだけで確定できるものはここで生成しておきます。
    """
    interval_str = f"{interval_sec}s"
    bars_df = (
        tick_df.with_columns(
            pl.col("trade_ts").dt.truncate(interval_str).alias("trade_ts")
        )
        .group_by("trade_ts")
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("trade_volume").mean().alias("avg_trade_size"),
                pl.col("trade_volume").max().alias("max_trade_size"),
                pl.col("buy_vol").sum().alias("buy_volume"),
                pl.col("sell_vol").sum().alias("sell_volume"),
                (pl.col("buy_vol").sum() + pl.col("sell_vol").sum()).alias("volume"),
                pl.len().alias("tick_count"),
            ]
        )
        .sort("trade_ts")
    )

    return bars_df.with_columns(
        [
            pl.when(pl.col("volume") > 0)
            .then((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("volume"))
            .otherwise(0.0)
            .alias("size_imb_1bar")
        ]
    )


def compute_poc_features(
    bars_df: pl.DataFrame, price_bin_size: int = 10
) -> pl.DataFrame:
    """
    全期間のバーデータから日次のVolume Profileを計算し、
    過去のPOC (Point of Control) 特徴量を算出・結合します。

    Args:
        bars_df (pl.DataFrame): ベースとなるバーデータ (1分足など)
        price_bin_size (int, optional): POC計算時の価格ビン幅. Defaults to 10.

    Returns:
        pl.DataFrame: `prev_poc_1d`, `prev_poc_1w`, `prev_poc_4w` が追加されたデータフレーム
    """
    # 1. 処理用に日付と価格ビンを追加
    df = bars_df.with_columns(
        [
            pl.col("trade_ts").dt.date().alias("date"),
            (pl.col("close") // price_bin_size * price_bin_size)
            .cast(pl.Int32)
            .alias("price_bin"),
        ]
    )

    # 2. 日ごとの各価格ビンの出来高を集計
    daily_profile = (
        df.group_by(["date", "price_bin"])
        .agg(pl.col("volume").sum().alias("volume"))
        .sort("date")
    )

    dates = df.select("date").unique().sort("date").get_column("date").to_list()

    # 3. 日付ごとのDataFrameを辞書化 (高速化のため)
    profile_dict = {}
    for date_tuple, group in daily_profile.partition_by("date", as_dict=True).items():
        # partition_byのキー仕様の違い（タプル返却）を吸収
        date_val = date_tuple[0] if isinstance(date_tuple, tuple) else date_tuple
        profile_dict[date_val] = group

    # 4. ローリングで過去N日間のPOCを計算
    results = []
    for i, current_date in enumerate(dates):

        def get_poc(target_dates: list) -> Optional[int]:
            """指定された日付リストの出来高を合算し、最大の価格ビン（POC）を取得する内部関数"""
            if not target_dates:
                return None
            dfs = [profile_dict[d] for d in target_dates]
            concat_df = pl.concat(dfs).group_by("price_bin").agg(pl.col("volume").sum())

            # 最大出来高の価格ビンを取得
            max_vol_df = concat_df.filter(pl.col("volume") == pl.col("volume").max())
            if len(max_vol_df) > 0:
                return max_vol_df["price_bin"][0]
            return None

        # インデックス範囲を考慮して過去のPOCを取得 (1w=5日, 4w=20日)
        poc_1d = get_poc([dates[i - 1]]) if i >= 1 else None
        poc_1w = get_poc(dates[i - 5 : i]) if i >= 5 else None
        poc_4w = get_poc(dates[i - 20 : i]) if i >= 20 else None

        results.append(
            {
                "date": current_date,
                "prev_poc_1d": poc_1d,
                "prev_poc_1w": poc_1w,
                "prev_poc_4w": poc_4w,
            }
        )

    poc_df = pl.DataFrame(
        results,
        schema={
            "date": pl.Date,
            "prev_poc_1d": pl.Int32,
            "prev_poc_1w": pl.Int32,
            "prev_poc_4w": pl.Int32,
        },
    )

    # 5. 元の1分足データフレームに結合
    bars_df = df.join(poc_df, on="date", how="left")

    # 作業用カラムを削除して返す
    return bars_df.drop(["date", "price_bin"])


def main():
    """
    TSVファイルから歩み値を読み込み、秒足集計および全期間のPOC計算を行った後、
    月単位でParquetファイルとして保存します。
    """
    root_tsv_dir = "C:/transformer_futures_data/tsv"
    output_base_dir = "C:/transformer_futures_data/parquet"
    symbol = "NK225"

    print(f"NK225 インポート開始: Root={root_tsv_dir}, Output={output_base_dir}")

    files = sorted(
        glob.glob(
            os.path.join(root_tsv_dir, symbol, "**", "future_tick_*.tsv.gz"),
            recursive=True,
        )
    )

    all_bars = []

    # 1. 各TSVファイルの読み込みとバー集計
    for file_path in files:
        # ファイル名から YYYYMM を抽出 (例: future_tick_19_201801.tsv.gz -> 201801)
        match = re.search(r"_(\d{6,8})\.tsv", os.path.basename(file_path))
        target_period = match.group(1) if match else "000000"

        raw_ticks = load_jpx_ticks_from_tsv(file_path)
        if raw_ticks.is_empty():
            continue

        bars_df = resample_to_bars(raw_ticks, BAR_SECONDS)
        all_bars.append(bars_df)
        print(f"  => Processed {target_period} ({len(bars_df)} rows)")

    if not all_bars:
        print("処理するデータがありませんでした。")
        return

    # 2. 全期間データの結合とPOCの計算
    print("全期間のデータを結合してPOCを計算します...")
    full_bars_df = pl.concat(all_bars).sort("trade_ts")
    full_bars_df = compute_poc_features(full_bars_df)

    # 3. 月ごとに分割して保存
    print("月ごとに分割して保存します...")
    full_bars_df = full_bars_df.with_columns(
        pl.col("trade_ts").dt.strftime("%Y%m").alias("period")
    )

    for period_tuple, group in full_bars_df.partition_by(
        "period", as_dict=True
    ).items():
        # partition_byのキー仕様の違い（タプル返却）を吸収
        period = period_tuple[0] if isinstance(period_tuple, tuple) else period_tuple
        year_str = period[:4]
        out_dir = os.path.join(output_base_dir, symbol, year_str)
        out_filename = f"{symbol}-{BAR_SECONDS}-{period}.parquet"
        out_path = os.path.join(out_dir, out_filename)

        try:
            os.makedirs(out_dir, exist_ok=True)
            # 保存時は作業用カラム(period)を除去
            group.drop("period").write_parquet(out_path)
            print(f"  => Saved: {out_path} ({len(group)} rows)")
        except Exception as e:
            print(f"  [Save Error] {out_filename}: {e}")

    print("\nすべてのNK225データ処理が完了しました。")


if __name__ == "__main__":
    main()
