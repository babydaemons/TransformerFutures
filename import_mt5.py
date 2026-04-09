# importer/import_mt5.py
"""
File: importer/import_mt5.py

ソースコードの役割:
本モジュールは、MT5由来の外部指標（USDJPY, US500等）のTSV(.gz)形式のミリ秒精度歩み値データを走査し、
指定された秒足（デフォルト1分足）に集約・タイムゾーン変換を行って、
銘柄ごとの階層化された Parquet ファイル（銘柄/年/ファイル名）として出力します。
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


def load_mt5_ticks_from_tsv(file_path: str) -> pl.DataFrame:
    """TSV(.gz)ファイルから歩み値を読み込み、正規化されたDataFrameを返します。

    Saver.cs経由で出力されたMT5のTickデータ(timestamp, bid, ask)を読み込みます。

    Args:
        file_path (str): 読み込み対象のTSVファイル（または .gz）のパス。

    Returns:
        pl.DataFrame: mt5_ts と price を含む正規化済みデータ。
    """
    if not os.path.exists(file_path):
        return pl.DataFrame()

    try:
        # Polarsのread_csvは自動でgzip解凍に対応。拡張子が.tsvでも.tsv.gzでも透過的に読める
        ldf = pl.read_csv(file_path, separator="\t")

        # カラム名の揺れを吸収してタイムスタンプをパース
        if "timestamp" in ldf.columns:
            ldf = ldf.with_columns(
                pl.col("timestamp")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%3f", strict=False)
                .alias("mt5_ts")
            )
        elif "time" in ldf.columns:
            ldf = ldf.with_columns(pl.col("time").str.to_datetime().alias("mt5_ts"))
        else:
            return pl.DataFrame()

        # AskレートをClose価格として採用
        price_col = (
            "ask"
            if "ask" in ldf.columns
            else "close" if "close" in ldf.columns else "last"
        )

        return ldf.select(
            ["mt5_ts", pl.col(price_col).cast(pl.Float64).alias("price")]
        ).drop_nulls(subset=["mt5_ts"])

    except Exception as e:
        print(f"  [Error] {os.path.basename(file_path)} の処理中にエラー: {e}")
        return pl.DataFrame()


def get_weekly_files(root_dir: str, symbol: str) -> List[str]:
    """ディレクトリツリー内からシンボルに該当するTSVファイルを再帰的に取得します。

    Args:
        root_dir (str): データのルートディレクトリ。
        symbol (str): シンボル名（USDJPY 等）。

    Returns:
        List[str]: 重複を除去しソート済みのファイルパスリスト。
    """
    pattern_gz = os.path.join(root_dir, symbol, "**", f"{symbol}-*.tsv.gz")
    pattern_tsv = os.path.join(root_dir, symbol, "**", f"{symbol}-*.tsv")
    files = glob.glob(pattern_gz, recursive=True) + glob.glob(
        pattern_tsv, recursive=True
    )
    return sorted(list(set(files)))


def extract_date_from_filename(path: str) -> str:
    """ファイル名（例: USDJPY-20240101.tsv.gz）から日付部分を抽出します。

    Args:
        path (str): ファイルパス。

    Returns:
        str: 抽出された日付文字列（YYYYMMDD）。見つからない場合は "00000000"。
    """
    match = re.search(r"(\d{8})", os.path.basename(path))
    return match.group(1) if match else "00000000"


def resample_to_bars(tick_df: pl.DataFrame, interval_sec: int) -> pl.DataFrame:
    """Tickデータを指定秒足に集約し、MT5からJSTへのタイムゾーン変換を行います。

    Args:
        tick_df (pl.DataFrame): load_mt5_ticks_from_tsv で取得したデータ。
        interval_sec (int): リサンプリング間隔(秒)。

    Returns:
        pl.DataFrame: 指定秒足に集約された時系列データ。
    """
    interval_str = f"{interval_sec}s"

    # MT5サーバータイム(US/Eastern基準:-7h)を日本標準時(Asia/Tokyo)へ変換
    df = tick_df.with_columns(
        (pl.col("mt5_ts") - pl.duration(hours=7))
        .dt.replace_time_zone("US/Eastern", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("Asia/Tokyo")
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("ns")
        .alias("trade_ts")
    ).drop_nulls(subset=["trade_ts"])

    return (
        df.with_columns(pl.col("trade_ts").dt.truncate(interval_str).alias("trade_ts"))
        .group_by("trade_ts")
        .agg([pl.col("price").last().alias("close")])
        .sort("trade_ts")
    )


def main():
    """インポート処理のエントリポイント。"""
    root_tsv_dir = "C:/transformer_futures_data/tsv"
    output_base_dir = "C:/transformer_futures_data/parquet"
    external_assets = ["USDJPY", "US500", "XAUUSD", "XTIUSD"]

    print(f"MT5 インポート開始: Root={root_tsv_dir}, Output={output_base_dir}")

    for ext_s in external_assets:
        files = get_weekly_files(root_tsv_dir, ext_s)
        if not files:
            print(f"  [Skip] {ext_s} のファイルが見つかりませんでした。")
            continue

        print(f"\n[Asset: {ext_s}]")
        for file_path in files:
            target_date = extract_date_from_filename(file_path)
            raw_ticks = load_mt5_ticks_from_tsv(file_path)
            if raw_ticks.is_empty():
                continue

            bars_df = resample_to_bars(raw_ticks, BAR_SECONDS)

            # --- 保存先パスの階層化 (銘柄/年/ファイル名) ---
            year_str = target_date[:4]
            out_dir = os.path.join(output_base_dir, ext_s, year_str)
            out_filename = f"{ext_s}-{BAR_SECONDS}-{target_date}.parquet"
            out_path = os.path.join(out_dir, out_filename)

            try:
                os.makedirs(out_dir, exist_ok=True)
                bars_df.write_parquet(out_path)
                print(f"  => Saved: {out_path} ({len(bars_df)} rows)")
            except Exception as e:
                print(f"  [Save Error] {out_path}: {e}")

    print("\nすべての外部指標データ処理が完了しました。")


if __name__ == "__main__":
    main()
