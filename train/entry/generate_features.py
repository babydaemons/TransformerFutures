# train/entry/generate_features.py
"""
File: train/entry/generate_features.py

ソースコードの役割:
日経平均先物ミニと外部指標の30秒足Parquetデータを読み込み、
Transformerのエントリー可否判定用特徴量とラベルを生成して保存します。

PolarsのLazyFrameを活用し、数年分のデータでも省メモリで
高速に処理（ストリーミング実行）できるようにしています。

本モジュールは以下の責務を持ちます。
1. NK225 30秒 bar parquet を列挙・読み込みする（LazyFrame）
2. 外部指標 30秒 bar parquet をシンボル別に列挙・読み込みする（LazyFrame）
3. 同一取引日の bar を結合対象として揃える
4. 特徴量およびラベルを生成するための実行計画を構築する
5. 学習用 parquet を collect() → write_parquet() でソート順を保証して保存する

入力想定:
- NK225:
  data/bars/nk225_30s/<YEAR>/<YYYY-MM-DD>.parquet
- 外部指標:
  data/bars/external_30s/<SYMBOL>/<YEAR>/<YYYY-MM-DD>.parquet

出力想定:
- 学習用特徴量:
  data/features/entry/<YEAR>/<YYYY-MM-DD>.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import polars as pl

from train.entry.features import build_entry_feature_frame


def iter_nk225_bar_files(bar_base_dir: str) -> Iterable[str]:
    """
    NK225 の 30秒 bar parquet ファイルを列挙します。

    Args:
        bar_base_dir (str): data ディレクトリなどのベースディレクトリ

    Returns:
        Iterable[str]: 見つかった NK225 bar parquet のパス一覧
    """
    pattern = os.path.join(bar_base_dir, "bars", "nk225_30s", "*", "*.parquet")
    return glob.glob(pattern)


def parse_date(filepath: str) -> str:
    """
    parquet ファイル名から取引日を抽出します。

    Args:
        filepath (str): parquet ファイルパス

    Returns:
        str: YYYY-MM-DD
    """
    return Path(filepath).stem


def main() -> None:
    """
    エントリポイントです。
    NK225 / 外部指標の 30秒 bar parquet から学習用特徴量 parquet を生成します。
    """
    parser = argparse.ArgumentParser(
        description="30秒 bar parquet から学習用 entry features parquet を生成します。"
    )
    parser.add_argument(
        "--bar-base-dir",
        default="data",
        help="bars parquet を含むベースディレクトリ",
    )
    parser.add_argument(
        "--output-base-dir",
        default="data",
        help="features parquet の出力ベースディレクトリ",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="開始日 (YYYY-MM-DD)。未指定時は全期間。",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="終了日 (YYYY-MM-DD)。未指定時は全期間。",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=240,  # 2時間後の効率比ラベル (30秒足で240本) を既定値にします。
        help="ラベル生成に使う未来参照本数。30秒足なので 60=30分, 120=60分, 240=120分。",
    )
    args = parser.parse_args()

    nk225_files = sorted(iter_nk225_bar_files(args.bar_base_dir))
    # MT5 シンボル名 → features.py が期待する短縮 prefix のマッピング。
    # add_cross_asset_features はこの短縮 prefix をハードコードで参照するため、
    # ここで必ず変換してから external_frames に格納する必要があります。
    SYMBOL_TO_PREFIX: dict[str, str] = {
        "USDJPY": "usdjpy",
        "US500":  "sp500",
        "NAS100": "nasdaq",
        "XAUUSD": "xau",
        "XTIUSD": "xti",
    }
    external_symbols = list(SYMBOL_TO_PREFIX.keys())

    for filepath in nk225_files:
        date_str = parse_date(filepath)

        if args.date_from and date_str < args.date_from:
            continue
        if args.date_to and date_str > args.date_to:
            continue

        year_str = date_str[:4]

        external_frames = {}
        for sym in external_symbols:
            ext_path = os.path.join(
                args.bar_base_dir,
                "bars",
                "external_30s",
                sym,
                year_str,
                f"{date_str}.parquet",
            )
            if os.path.exists(ext_path):
                # ディレクトリ名は MT5 シンボル名のまま、キーを短縮 prefix に変換する
                external_frames[SYMBOL_TO_PREFIX[sym]] = pl.scan_parquet(ext_path)

        if len(external_frames) != len(external_symbols):
            print(f"[SKIP] Missing external bars for {date_str}")
            continue

        print(f"[LOAD] NK225: {filepath}")
        nk225_df = pl.scan_parquet(filepath)

        feature_df = build_entry_feature_frame(
            nk225_df=nk225_df,
            external_frames=external_frames,
            label_horizon=args.label_horizon,
        )

        year_str = date_str[:4]
        out_dir = os.path.join(args.output_base_dir, "features", "entry", year_str)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.parquet")

        # sink_parquet はストリーミング実行のためグローバルなソート順が保証されない。
        # collect() で eager に変換してからファイルへ書き込む。
        feature_df.collect().write_parquet(out_path, compression="zstd")
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
