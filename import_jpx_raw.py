# import_jpx_raw.py
"""
File: import_jpx_raw.py

ソースコードの役割:
JPXデータクラウド形式の日経平均先物（NK225）歩み値CSVを読み込み、
raw parquet を 1取引日 1ファイルで生成します。

本モジュールは raw データの正規化だけに責務を限定します。
秒足集計、POC算出、学習用特徴量生成は行いません。

処理内容:
1. CSV(.csv.gz) を直接読み込む
2. 列名差分（大文字/小文字、Make_Date/Execution_Date）を吸収する
3. trade_date 単位で期近限月を選別する
4. Tick Test により売買方向を推定する
5. buy_vol / sell_vol を付与する
6. trade_date ごとに raw parquet として保存する
"""

import argparse
import glob
import os
import gzip
from typing import Iterable

import polars as pl


def normalize_jpx_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    JPX CSV の列名揺れを吸収し、後段で扱いやすい列名へ正規化します。

    Args:
        df (pl.DataFrame): 正規化前のPolars DataFrame

    Returns:
        pl.DataFrame: 列名が正規化されたPolars DataFrame
    """
    rename_map = {}
    for col in df.columns:
        lower_col = col.lower()
        rename_map[col] = lower_col

    # 全列名を小文字に統一
    df = df.rename(rename_map)

    # 古い期間の列名差分を吸収
    if "make_date" in df.columns and "execution_date" not in df.columns:
        df = df.rename({"make_date": "execution_date"})

    # 2022/9以降だけ存在することがある列に対するフォールバック
    if "sco_category" not in df.columns:
        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("sco_category"))

    # JPX仕様差分によって execution_date が無いケースでは trade_date を代用
    if "execution_date" not in df.columns:
        df = df.with_columns(pl.col("trade_date").alias("execution_date"))

    # 古いCSVデータで 'no' 列が存在しない場合のフォールバック
    if "no" not in df.columns:
        # 行番号 (1〜) を生成して 'no' 列として代用する
        df = df.with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("no"))

    return df


def classify_session_type(trade_ts_col: pl.Expr) -> pl.Expr:
    """
    約定時刻から DAY / NIGHT を判定します。

    Args:
        trade_ts_col (pl.Expr): 約定タイムスタンプを保持するPolars式

    Returns:
        pl.Expr: DAY / NIGHT の文字列を返すPolars式
    """
    # 0時からの経過秒数を計算 (Int8のオーバーフローを防ぐためInt64にキャスト)
    seconds = (
        trade_ts_col.dt.hour().cast(pl.Int64) * 3600
        + trade_ts_col.dt.minute().cast(pl.Int64) * 60
        + trade_ts_col.dt.second().cast(pl.Int64)
    )
    # 08:45:00 から 15:45:00 未満をDAYセッションとする
    return pl.when(
        (seconds >= 8 * 3600 + 45 * 60) & (seconds < 15 * 3600 + 45 * 60)
    ).then(pl.lit("DAY")).otherwise(pl.lit("NIGHT"))


def detect_csv_separator(file_path: str) -> str:
    """
    入力ファイルの区切り文字を判定します。

    本来の入力は CSV を想定しますが、移行期間中の互換性維持のため
    TSV が渡された場合も自動判定で受け入れます。圧縮ファイル(.gz)にも対応します。

    Args:
        file_path (str): 読み込む CSV / TSV ファイルのパス

    Returns:
        str: Polars に渡す区切り文字 (',' または '\t')
    """
    # .gzファイルかプレーンテキストかで開き方を分岐
    open_func = gzip.open if file_path.endswith(".gz") else open
    
    with open_func(file_path, "rt", encoding="utf-8", newline="") as fp:
        first_line = fp.readline()

    comma_count = first_line.count(",")
    tab_count = first_line.count("\t")
    
    # カンマの数がタブ以上であればCSV、それ以外はTSVと判定
    return "," if comma_count >= tab_count else "\t"


def load_jpx_raw_from_csv(file_path: str) -> pl.DataFrame:
    """
    JPX CSV から raw parquet 向けの正規化済み歩み値 DataFrame を生成します。

    中間TSVは作成せず、JPX配布CSVを直接読み込みます。
    なお、既存資産との互換性維持のため、TSV が入力された場合も
    区切り文字を自動判定して読み込みます。

    Args:
        file_path (str): 読み込む CSV(.csv.gz) または TSV(.tsv.gz) のパス

    Returns:
        pl.DataFrame: 正規化・型変換・Tick Testが付与与されたPolars DataFrame
    
    Raises:
        ValueError: 必須列がデータ内に存在しなかった場合
    """
    separator = detect_csv_separator(file_path)

    df = pl.read_csv(
        file_path,
        separator=separator,
        has_header=True,
        infer_schema_length=10000,
        null_values=["", "NULL", "null"],
    )
    
    # カラム名の正規化と揺れ吸収
    df = normalize_jpx_columns(df)

    # index_type, security_code を必須列から除外して検証
    required_cols = [
        "trade_date",
        "execution_date",
        "time",
        "trade_price",
        "trade_volume",
        "contract_month",
        "price_type",
        "no",
        "sco_category",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"必須列が不足しています: {missing_cols} | file={file_path}"
        )

    # 必須列を型正規化し、時刻文字列から Datetime 型の `trade_ts` を構築
    df = df.with_columns(
        [
            pl.col("trade_date").cast(pl.Utf8).str.zfill(8).alias("trade_date"),
            pl.col("execution_date")
            .cast(pl.Utf8)
            .str.zfill(8)
            .alias("execution_date"),
            pl.col("time").cast(pl.Utf8).str.zfill(9).alias("time"),
            pl.col("trade_price").cast(pl.Float64).alias("trade_price"),
            pl.col("trade_volume").cast(pl.Int64).alias("trade_volume"),
            pl.col("contract_month").cast(pl.Int32).alias("contract_month"),
            pl.col("price_type").cast(pl.Utf8).alias("price_type"),
            pl.col("no").cast(pl.Int64).alias("sequence_no"),
            pl.col("sco_category").cast(pl.Int8).alias("sco_category"),
        ]
    ).with_columns(
        (pl.col("execution_date") + pl.col("time"))
        .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S%3f", strict=False)
        .alias("trade_ts")
    ).drop_nulls(subset=["trade_ts"])

    # trade_date 単位で期近限月を選別
    near_cm_df = df.group_by("trade_date").agg(
        pl.col("contract_month").min().alias("near_contract_month")
    )
    df = df.join(near_cm_df, on="trade_date", how="left").filter(
        pl.col("contract_month") == pl.col("near_contract_month")
    )

    # Tick Test はザラバ約定 N のみで実施
    df = df.filter(pl.col("price_type") == "N").sort(["trade_date", "trade_ts", "sequence_no"])
    df = df.with_columns(
        pl.col("trade_price").diff().over("trade_date").alias("price_diff")
    )
    
    # Tick Testによる売買方向（買い: 1, 売り: -1）の推測
    df = (
        df.with_columns(
            pl.when(pl.col("price_diff") > 0)
            .then(1)
            .when(pl.col("price_diff") < 0)
            .then(-1)
            .otherwise(None)
            .alias("direction_raw")
        )
        .with_columns(
            pl.col("direction_raw")
            .forward_fill()  # 同値の場合は直前の方向を引き継ぐ (Zero-Tick Test)
            .over("trade_date")
            .fill_null(0)
            .cast(pl.Int8)
            .alias("direction")
        )
        .drop("direction_raw")
    )

    # 買い・売りごとの約定ボリュームを計算
    df = df.with_columns(
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
    ).with_columns(
        [
            pl.col("trade_date")
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("session_date_jst"),
            classify_session_type(pl.col("trade_ts")).alias("session_type"),
            pl.lit("NK225").alias("symbol"),
            pl.col("contract_month").cast(pl.Utf8).alias("contract_code"),
        ]
    )

    # 必要な列のみを選択して返却
    return df.select(
        [
            "session_date_jst",
            "session_type",
            "trade_date",
            "execution_date",
            "trade_ts",
            "contract_month",
            "contract_code",
            "sequence_no",
            "price_type",
            "sco_category",
            pl.col("trade_price").alias("price"),
            "trade_volume",
            "buy_vol",
            "sell_vol",
            "direction",
            "symbol",
        ]
    )


def save_daily_raw_parquet(raw_df: pl.DataFrame, output_base_dir: str) -> None:
    """
    raw DataFrame を 1取引日 1ファイルの parquet として保存します。

    Args:
        raw_df (pl.DataFrame): 保存対象の raw parquet 向け DataFrame
        output_base_dir (str): 出力ベースディレクトリのパス
    """
    # JSTの日付単位でパーティション分割
    grouped = raw_df.partition_by("session_date_jst", as_dict=True)
    
    for trade_date_key, group in grouped.items():
        trade_date = trade_date_key[0] if isinstance(trade_date_key, tuple) else trade_date_key
        year_str = trade_date.strftime("%Y")
        trade_date_str = trade_date.strftime("%Y-%m-%d")
        
        out_dir = os.path.join(
            output_base_dir,
            "raw",
            "jpx_nk225_tick",
            year_str,
        )
        out_path = os.path.join(out_dir, f"{trade_date_str}.parquet")

        os.makedirs(out_dir, exist_ok=True)
        
        # 時系列とシーケンス番号でソートしてから圧縮保存
        group.sort(["trade_ts", "sequence_no"]).write_parquet(
            out_path,
            compression="zstd",
        )
        print(f"Saved: {out_path} ({len(group)} rows)")


def iter_monthly_source_files(root_source_dir: str, symbol: str) -> Iterable[str]:
    """
    月次ソースファイル群を列挙します。

    まず JPX の原本である CSV(.csv.gz) を優先し、互換性維持のために
    既存の TSV(.tsv.gz) も後方互換として受け入れます。

    Args:
        root_source_dir (str): JPX ソースファイル ルートディレクトリ
        symbol (str): 対象となるシンボル名 (例: NK225)

    Returns:
        Iterable[str]: 見つかった入力ファイルパスのリスト
    """
    csv_pattern = os.path.join(root_source_dir, symbol, "**", "future_tick_*.csv.gz")
    tsv_pattern = os.path.join(root_source_dir, symbol, "**", "future_tick_*.tsv.gz")
    
    # CSVとTSVの両方を検索し、重複を排除してソート
    return sorted(set(glob.glob(csv_pattern, recursive=True) + glob.glob(tsv_pattern, recursive=True)))


def main() -> None:
    """
    エントリポイントです。
    コマンドライン引数を解析し、指定されたディレクトリ内のCSV/TSVファイルを順次処理して
    日次単位のParquetファイルに変換します。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-source-dir", default="data/original/JPX", help="入力CSV/TSVファイルのルートディレクトリ")
    parser.add_argument("--output-base-dir", default="data", help="Parquetを出力するベースディレクトリ")
    parser.add_argument("--symbol", default="NK225", help="処理対象のシンボル")
    args = parser.parse_args()

    files = iter_monthly_source_files(args.root_source_dir, args.symbol)
    if not files:
        print("CSV / TSV ファイルが見つかりませんでした。")
        return

    for file_path in files:
        print(f"Processing: {file_path}")
        raw_df = load_jpx_raw_from_csv(file_path)
        
        if raw_df.is_empty():
            print("  -> empty")
            continue
            
        save_daily_raw_parquet(raw_df, args.output_base_dir)

    print("CSV からの raw parquet 生成が完了しました。")


if __name__ == "__main__":
    main()
