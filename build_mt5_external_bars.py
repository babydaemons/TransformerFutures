# build_mt5_external_bars.py
"""
File: build_mt5_external_bars.py

ソースコードの役割:
MT5 から保存した外部指標ティックTSV(.tsv / .tsv.gz)を読み込み、
30秒足の external bar parquet を生成します。

本モジュールは bar 集約だけに責務を限定します。
学習用特徴量生成は行いません。

入力フォーマット:
Saver.cs が出力する週次TSV/TSV.gz
  - ヘッダ: なし、または timestamp, bid, ask (自動判別)
  - timestamp 形式: yyyy-MM-dd HH:mm:ss.fff

出力フォーマット:
  data/bars/external_30s/<SYMBOL>/<YEAR>/<YYYY-MM-DD>.parquet
"""

import argparse
import glob
import os
from typing import Iterable

import polars as pl


BAR_SECONDS = 30


def classify_session_type_from_bar_start(bar_start_col: pl.Expr) -> pl.Expr:
    """
    bar_start_jst から DAY / NIGHT を判定します。

    Args:
        bar_start_col (pl.Expr): 30秒バー開始時刻

    Returns:
        pl.Expr: DAY / NIGHT を示す文字列の列
    """
    # Polarsのdt.hour()等はInt8などを返す場合があり、乗算時にオーバーフローする
    # 危険があるため、明示的にInt32へキャストしてから秒換算を行います。
    hour = bar_start_col.dt.hour().cast(pl.Int32)
    minute = bar_start_col.dt.minute().cast(pl.Int32)
    second = bar_start_col.dt.second().cast(pl.Int32)

    seconds = (
        hour * 3600
        + minute * 60
        + second
    )

    # 日中立会 (DAY): JST 08:45:00 <= t < 15:45:00
    return pl.when(
        (seconds >= 8 * 3600 + 45 * 60) & (seconds < 15 * 3600 + 45 * 60)
    ).then(pl.lit("DAY")).otherwise(pl.lit("NIGHT"))


def add_jpx_session_date(df: pl.DataFrame) -> pl.DataFrame:
    """
    bar_start_jst から JPX 基準の取引日 (session_date_jst) を推論します。
    16:00以降のデータは翌営業日（金曜なら翌週の月曜）のNIGHTセッションとして扱います。

    Args:
        df (pl.DataFrame): bar_start_jst を含む30秒足 DataFrame

    Returns:
        pl.DataFrame: session_date_jst 列を付与した DataFrame
    """
    # Polarsのweekday: 1=月曜 ... 5=金曜, 6=土曜, 7=日曜
    return df.with_columns(
        pl.when(pl.col("bar_start_jst").dt.hour() >= 16)
        .then(
            pl.when(pl.col("bar_start_jst").dt.weekday() == 5)  # 金曜
            .then(pl.col("bar_start_jst").dt.date() + pl.duration(days=3))
            .when(pl.col("bar_start_jst").dt.weekday() == 6)  # 土曜（念のため）
            .then(pl.col("bar_start_jst").dt.date() + pl.duration(days=2))
            .otherwise(pl.col("bar_start_jst").dt.date() + pl.duration(days=1))
        )
        .otherwise(pl.col("bar_start_jst").dt.date())
        .alias("session_date_jst")
    )


def iter_mt5_tsv_files(mt5_base_dir: str, symbol: str | None = None) -> Iterable[str]:
    """
    MT5 の TSV / TSV.gz ファイル群を列挙します。

    Args:
        mt5_base_dir (str): MT5 生データのベースディレクトリ
        symbol (str | None, optional): 対象シンボルを絞る場合のシンボル名

    Returns:
        Iterable[str]: 見つかったファイルパスのリスト（ソート済み）
    """
    if symbol:
        patterns = [
            os.path.join(mt5_base_dir, symbol, "*", f"{symbol}-*.tsv"),
            os.path.join(mt5_base_dir, symbol, "*", f"{symbol}-*.tsv.gz"),
        ]
    else:
        patterns = [
            os.path.join(mt5_base_dir, "*", "*", "*.tsv"),
            os.path.join(mt5_base_dir, "*", "*", "*.tsv.gz"),
        ]

    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(files)


def infer_symbol_from_path(file_path: str) -> str:
    """
    ディレクトリ階層の構造からシンボル名を推定します。

    Args:
        file_path (str): 入力ファイルパス (例: .../MT5/USDJPY/2026/USDJPY-20260302.tsv.gz)

    Returns:
        str: 推定されたシンボル名 (例: USDJPY)
    """
    return os.path.basename(os.path.dirname(os.path.dirname(file_path)))


def load_mt5_ticks(file_path: str, symbol: str | None = None) -> pl.DataFrame:
    """
    MT5 TSV / TSV.gz を読み込み、タイムゾーン変換と正規化を行ったティックDataFrameを返します。
    ヘッダー行の有無に関わらず、柔軟に列名と型を解決します。

    Args:
        file_path (str): 入力ファイルパス
        symbol (str | None, optional): シンボル名。None の場合はパスから推定します。

    Returns:
        pl.DataFrame: 正規化済みティック DataFrame (trade_ts, bid, ask, mid, spread等を含む)

    Raises:
        ValueError: TSVに必要な列（最低3列）が不足している場合
    """
    resolved_symbol = symbol or infer_symbol_from_path(file_path)

    # ヘッダー行の有無が不定なため、全てデータとして読み込む
    df = pl.read_csv(
        file_path,
        separator="\t",
        has_header=False,
        infer_schema_length=10000,
        null_values=["", "NULL", "null"],
        try_parse_dates=False,
    )

    if df.width < 3:
        raise ValueError(f"列数が不足しています（最低3列必要）: {df.width}列 | file={file_path}")

    # 先頭3列を強制的にリネーム
    cols = df.columns
    df = df.rename({
        cols[0]: "timestamp",
        cols[1]: "bid",
        cols[2]: "ask",
    })

    # ヘッダー行がデータ行として読み込まれている場合（1行目の timestamp 列が "timestamp" の場合）は除外する
    # ※ 数値型の列に "bid" 等の文字が混じると Polars が String 型として推論するため、
    #    この段階で除外し、後続の with_columns で明示的に Float64 にキャストします。
    df = df.filter(pl.col("timestamp") != "timestamp")

    df = (
        df.with_columns(
            [
                pl.col("timestamp")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.3f", strict=False)
                # TSV(MT5サーバー時間)を標準的なFXブローカーのタイムゾーン(EET)とみなし、JSTへ変換
                .dt.replace_time_zone("EET")
                .dt.convert_time_zone("Asia/Tokyo")
                .dt.replace_time_zone(None)  # タイムゾーン情報を落としてnaive datetimeに戻す
                .alias("trade_ts"),
                pl.col("bid").cast(pl.Float64, strict=False).alias("bid"),
                pl.col("ask").cast(pl.Float64, strict=False).alias("ask"),
            ]
        )
        .drop_nulls(subset=["trade_ts", "bid", "ask"])
        .with_columns(
            [
                ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
                (pl.col("ask") - pl.col("bid")).alias("spread"),
                pl.col("trade_ts").dt.date().alias("calendar_date_jst"),
                pl.lit(resolved_symbol).alias("symbol"),
            ]
        )
        .sort("trade_ts")
    )

    return df.select(
        [
            "trade_ts",
            "calendar_date_jst",
            "symbol",
            "bid",
            "ask",
            "mid",
            "spread",
        ]
    )


def build_30s_bars_from_ticks(tick_df: pl.DataFrame) -> pl.DataFrame:
    """
    MT5のティックデータを30秒足へ集約（リサンプリング）します。

    Args:
        tick_df (pl.DataFrame): 正規化済みティック DataFrame

    Returns:
        pl.DataFrame: 30秒足 DataFrame。外部指標用として mid 価格ベースの OHLC を算出。
    """
    if tick_df.is_empty():
        return pl.DataFrame()

    bar_df = (
        tick_df.with_columns(
            pl.col("trade_ts").dt.truncate(f"{BAR_SECONDS}s").alias("bar_start_jst")
        )
        .group_by(["symbol", "bar_start_jst"])
        .agg(
            [
                pl.col("bid").first().alias("bid_open"),
                pl.col("bid").max().alias("bid_high"),
                pl.col("bid").min().alias("bid_low"),
                pl.col("bid").last().alias("bid_close"),
                pl.col("ask").first().alias("ask_open"),
                pl.col("ask").max().alias("ask_high"),
                pl.col("ask").min().alias("ask_low"),
                pl.col("ask").last().alias("ask_close"),
                pl.col("mid").first().alias("mid_open"),
                pl.col("mid").max().alias("mid_high"),
                pl.col("mid").min().alias("mid_low"),
                pl.col("mid").last().alias("mid_close"),
                pl.col("spread").last().alias("spread_close"),
                pl.len().cast(pl.Int32).alias("tick_count"),
                pl.lit(0.0).cast(pl.Float64).alias("volume"),
            ]
        )
        .with_columns(
            [
                (pl.col("bar_start_jst") + pl.duration(seconds=BAR_SECONDS)).alias("bar_end_jst"),
                classify_session_type_from_bar_start(pl.col("bar_start_jst")).alias("session_type"),
                pl.lit(True).alias("is_complete"),
            ]
        )
        .sort(["symbol", "bar_start_jst"])
    )

    bar_df = add_jpx_session_date(bar_df)

    # 外部指標の OHLC は mid ベースを採用
    return bar_df.select(
        [
            "bar_start_jst",
            "bar_end_jst",
            "session_date_jst",
            "session_type",
            "symbol",
            pl.col("mid_open").alias("open"),
            pl.col("mid_high").alias("high"),
            pl.col("mid_low").alias("low"),
            pl.col("mid_close").alias("close"),
            "mid_open",
            "mid_high",
            "mid_low",
            "mid_close",
            "spread_close",
            "tick_count",
            "volume",
            "is_complete",
        ]
    )


def save_daily_bars(bar_df: pl.DataFrame, output_base_dir: str, symbol: str) -> None:
    """
    30秒足 DataFrame を JPX 基準の取引日・セッションタイプ (DAY/NIGHT) ごとに保存します。

    保存先フォーマット:
      data/bars/external_30s/<SYMBOL>/<YEAR>/<YYYY-MM-DD>.parquet

    Args:
        bar_df (pl.DataFrame): 30秒足 DataFrame
        output_base_dir (str): 出力ベースディレクトリ (Parquetファイルが保存されるルート)
        symbol (str): 出力対象シンボル
    """
    if bar_df.is_empty():
        return

    # 推論されたJPX Trade Date単位で保存
    for keys, group in bar_df.partition_by(["session_date_jst"], as_dict=True).items():
        session_date_jst = keys if not isinstance(keys, tuple) else keys[0]

        date_str = session_date_jst.strftime("%Y-%m-%d")
        year_str = date_str[:4]

        symbol_str = symbol if symbol else "UNKNOWN"
        out_dir = os.path.join(output_base_dir, "bars", "external_30s", symbol_str, year_str)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.parquet")

        group.write_parquet(out_path, compression="zstd")
        print(f"  -> Saved {out_path} ({len(group)} rows)")


def process_one_file(file_path: str, output_base_dir: str, symbol: str | None = None) -> None:
    """
    入力ファイル 1本を 30秒足 Parquet へ変換して保存する一連のパイプラインを実行します。

    Args:
        file_path (str): 入力ファイルパス
        output_base_dir (str): 出力ベースディレクトリ
        symbol (str | None, optional): シンボル名
    """
    print(f"Processing: {file_path}")
    tick_df = load_mt5_ticks(file_path, symbol=symbol)
    if tick_df.is_empty():
        print("  -> empty")
        return

    bar_df = build_30s_bars_from_ticks(tick_df)
    if bar_df.is_empty():
        print("  -> no bars")
        return

    resolved_symbol = symbol or infer_symbol_from_path(file_path)
    save_daily_bars(bar_df, output_base_dir, resolved_symbol)


def main() -> None:
    """
    スクリプトのエントリポイントです。
    コマンドライン引数をパースし、指定されたディレクトリ内のファイルを順次処理します。
    """
    parser = argparse.ArgumentParser(
        description="MT5のティックデータを30秒足のParquet形式に変換します。"
    )
    parser.add_argument(
        "--mt5-base-dir",
        default=r"C:\TransformerFutures\data\original\MT5",
        help="MT5 の生TSV/TSV.gz を含むベースディレクトリ",
    )
    parser.add_argument(
        "--output-base-dir",
        default="data",
        help="bars parquet の出力ベースディレクトリ",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="対象シンボルを絞る場合に指定します。例: USDJPY",
    )
    args = parser.parse_args()

    files = list(iter_mt5_tsv_files(args.mt5_base_dir, symbol=args.symbol))
    if not files:
        print("MT5 TSV / TSV.gz ファイルが見つかりませんでした。")
        return

    for file_path in files:
        process_one_file(file_path, args.output_base_dir, symbol=args.symbol)

    print("external 30秒 bar parquet 生成が完了しました。")


if __name__ == "__main__":
    main()
