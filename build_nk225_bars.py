# build_nk225_bars.py
"""
File: build_nk225_bars.py

ソースコードの役割:
raw parquet 形式の日経平均先物（NK225）歩み値データを読み込み、
30秒足の bar parquet を生成します。

本モジュールは bar 集約だけに責務を限定します。
POC 算出や学習用特徴量生成は行いません。

処理内容:
1. raw parquet を 1取引日単位で読み込む
2. DAY / NIGHT セッションごとに 30秒足へ集約する
3. OHLCV、buy_volume、sell_volume、signed_volume、VWAP を算出する
4. session_type ごとに bar parquet を 1日1ファイルで保存する
"""

import argparse
import glob
import os
from datetime import time
from typing import Iterable

import polars as pl


# 集約する足の秒数
BAR_SECONDS = 30


def iter_raw_parquet_files(raw_base_dir: str) -> Iterable[str]:
    """
    raw parquet ファイル群を列挙します。

    Args:
        raw_base_dir (str): raw parquet のベースディレクトリ

    Returns:
        Iterable[str]: 見つかった raw parquet ファイルのパス一覧
    """
    pattern = os.path.join(raw_base_dir, "raw", "jpx_nk225_tick", "*", "*.parquet")
    return sorted(glob.glob(pattern))


def load_raw_parquet(file_path: str) -> pl.DataFrame:
    """
    raw parquet を読み込みます。

    Args:
        file_path (str): raw parquet のパス

    Returns:
        pl.DataFrame: 読み込まれた raw データ
    """
    return pl.read_parquet(file_path)


def classify_session_type_from_bar_start(bar_start_col: pl.Expr) -> pl.Expr:
    """
    bar_start_jst から DAY / NIGHT を再判定します。

    Args:
        bar_start_col (pl.Expr): 30秒バー開始時刻のカラム表現

    Returns:
        pl.Expr: DAY / NIGHT の文字列を返す条件式

    Notes:
        raw parquet 側の session_type をそのまま信用せず、
        30秒バー化した時点の bar_start_jst から再計算します。
        これにより、raw 側の session_type 誤判定があっても
        bars 側では正しいセッションへ出力できます。
    """
    # 0時からの経過秒数を計算 (Int8のオーバーフローを防ぐためInt64にキャスト)
    seconds = (
        bar_start_col.dt.hour().cast(pl.Int64) * 3600
        + bar_start_col.dt.minute().cast(pl.Int64) * 60
        + bar_start_col.dt.second().cast(pl.Int64)
    )
    # 08:45:00 ～ 15:45:00 をDAYセッション、それ以外をNIGHTセッションとする
    return pl.when(
        (seconds >= 8 * 3600 + 45 * 60) & (seconds < 15 * 3600 + 45 * 60)
    ).then(pl.lit("DAY")).otherwise(pl.lit("NIGHT"))


def build_30s_bars_from_raw(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    raw データを 30秒足へ集約します。

    Args:
        raw_df (pl.DataFrame): raw parquet 由来の DataFrame

    Returns:
        pl.DataFrame: 30秒足に集約された DataFrame

    Notes:
        raw parquet には板情報が無いため、bid/ask 系列は NULL で埋めます。
        今後 kabu の板付き raw へ拡張した場合は、この関数内で集約ロジックを
        差し替えられるようにプレースホルダーとして定義しています。
    """
    if raw_df.is_empty():
        return pl.DataFrame()

    # bar 開始時刻を 30秒単位で切り下げて集約ベースを作成
    bar_df = (
        raw_df.with_columns(
            [
                pl.col("trade_ts").dt.truncate(f"{BAR_SECONDS}s").alias("bar_start_jst"),
                (pl.col("price") * pl.col("trade_volume")).alias("turnover"),
            ]
        )
        .group_by(
            [
                "session_date_jst",
                "symbol",
                "contract_code",
                "bar_start_jst",
            ]
        )
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("trade_volume").sum().cast(pl.Float64).alias("volume"),
                pl.len().cast(pl.Int32).alias("tick_count"),
                pl.col("buy_vol").sum().cast(pl.Float64).alias("buy_volume"),
                pl.col("sell_vol").sum().cast(pl.Float64).alias("sell_volume"),
                pl.col("turnover").sum().alias("turnover_sum"),
                pl.col("trade_ts").min().alias("first_trade_ts"),
                pl.col("trade_ts").max().alias("last_trade_ts"),
            ]
        )
        .sort(["session_date_jst", "bar_start_jst"])
    )

    # 派生指標 (VWAP, signed_volume 等) の計算とプレースホルダー列の追加
    bar_df = bar_df.with_columns(
        [
            (pl.col("bar_start_jst") + pl.duration(seconds=BAR_SECONDS)).alias("bar_end_jst"),
            classify_session_type_from_bar_start(pl.col("bar_start_jst")).alias("session_type"),
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("signed_volume"),
            pl.when(pl.col("volume") > 0)
            .then(pl.col("turnover_sum") / pl.col("volume"))
            .otherwise(pl.col("close"))
            .alias("vwap_30s"),
            pl.lit(None, dtype=pl.Float64).alias("bid_close"),
            pl.lit(None, dtype=pl.Float64).alias("ask_close"),
            pl.lit(None, dtype=pl.Float64).alias("spread_close"),
            pl.lit(None, dtype=pl.Float64).alias("bid_qty_close"),
            pl.lit(None, dtype=pl.Float64).alias("ask_qty_close"),
            pl.lit(None, dtype=pl.Float64).alias("order_imbalance"),
        ]
    )

    # is_complete:
    # 30秒バー内に少なくとも1約定があるものだけを出力するので基本は True。
    # 将来、空バー補完を入れた場合に False を使えるよう列は残しておきます。
    bar_df = (
        bar_df.with_columns(pl.lit(True).alias("is_complete"))
        .sort(["session_date_jst", "session_type", "bar_start_jst"])
    )

    return bar_df.select(
        [
            "bar_start_jst",
            "bar_end_jst",
            "session_date_jst",
            "session_type",
            "symbol",
            "contract_code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "bid_close",
            "ask_close",
            "spread_close",
            "bid_qty_close",
            "ask_qty_close",
            "order_imbalance",
            "buy_volume",
            "sell_volume",
            "signed_volume",
            "vwap_30s",
            "is_complete",
        ]
    )


def save_daily_bars(bar_df: pl.DataFrame, output_base_dir: str) -> None:
    """
    30秒足 DataFrame を session_type ごとに 1日1ファイルで保存します。

    Args:
        bar_df (pl.DataFrame): 30秒足 DataFrame
        output_base_dir (str): 出力ベースディレクトリ
    """
    if bar_df.is_empty():
        return

    # 日付とセッションタイプ単位でデータを分割
    grouped = bar_df.partition_by(["session_date_jst", "session_type"], as_dict=True)
    for key, group in grouped.items():
        if isinstance(key, tuple):
            trade_date = key[0]
            session_type = key[1]
        else:
            trade_date, session_type = key

        year_str = trade_date.strftime("%Y")
        trade_date_str = trade_date.strftime("%Y-%m-%d")

        out_dir = os.path.join(
            output_base_dir,
            "bars",
            "nk225_30s",
            session_type,
            year_str,
        )
        out_path = os.path.join(out_dir, f"{trade_date_str}.parquet")

        os.makedirs(out_dir, exist_ok=True)
        # 時系列順にソートして圧縮保存
        group.sort("bar_start_jst").write_parquet(
            out_path,
            compression="zstd",
        )
        print(f"Saved: {out_path} ({len(group)} rows)")


def process_one_file(file_path: str, output_base_dir: str) -> None:
    """
    raw parquet 1ファイルを読み込み、bar parquet へ変換して保存します。

    Args:
        file_path (str): raw parquet のパス
        output_base_dir (str): 出力ベースディレクトリ
    """
    print(f"Processing: {file_path}")
    raw_df = load_raw_parquet(file_path)
    if raw_df.is_empty():
        print("  -> empty")
        return

    bar_df = build_30s_bars_from_raw(raw_df)
    if bar_df.is_empty():
        print("  -> no bars")
        return

    save_daily_bars(bar_df, output_base_dir)


def main() -> None:
    """
    エントリポイントです。
    コマンドライン引数を解析し、指定された条件に合致するファイルを順次処理します。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-base-dir",
        default="data",
        help="raw parquet を含むベースディレクトリ",
    )
    parser.add_argument(
        "--output-base-dir",
        default="data",
        help="bars parquet の出力ベースディレクトリ",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD を指定すると、その日付ファイルだけ処理します。",
    )
    args = parser.parse_args()

    files = list(iter_raw_parquet_files(args.raw_base_dir))
    if not files:
        print("raw parquet ファイルが見つかりませんでした。")
        return

    # 日付フィルタの適用
    if args.date is not None:
        target_suffix = os.path.join(args.date[:4], f"{args.date}.parquet")
        files = [path for path in files if path.endswith(target_suffix)]

    if not files:
        print("対象日付の raw parquet ファイルが見つかりませんでした。")
        return

    # 見つかったファイルを順次処理
    for file_path in files:
        process_one_file(file_path, args.output_base_dir)

    print("30秒 bar parquet 生成が完了しました。")


if __name__ == "__main__":
    main()
