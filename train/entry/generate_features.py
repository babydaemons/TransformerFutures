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
3. 同一 session_date_jst / session_type の bar を結合対象として揃える
4. 特徴量およびラベルを生成するための実行計画を構築する
5. 学習用 parquet を sink_parquet を用いてストリーミング保存する

入力想定:
- NK225:
  data/bars/nk225_30s/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.parquet
- 外部指標:
  data/bars/external_30s/<SYMBOL>/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.parquet

出力想定:
- 学習用特徴量:
  data/features/entry/<YEAR>/<YYYY-MM-DD>-<DAY|NIGHT>.parquet
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
    pattern = os.path.join(
        bar_base_dir,
        "bars",
        "nk225_30s",
        "*",
        "*.parquet",
    )
    return sorted(glob.glob(pattern))


def parse_date_and_session(filepath: str) -> tuple[str, str]:
    """
    parquet ファイル名から取引日と session_type を抽出します。

    Args:
        filepath (str): parquet ファイルパス

    Returns:
        tuple[str, str]: (YYYY-MM-DD, DAY|NIGHT)
    """
    stem = Path(filepath).stem
    # 右から最初の '-' で分割します。
    # 例: 2018-01-04-DAY -> ("2018-01-04", "DAY")
    date_str, session = stem.rsplit("-", 1)
    return date_str, session


def extract_year_from_path(file_path: str) -> str:
    """
    parquet のパスから年ディレクトリを抽出します。

    Args:
        file_path (str): parquet ファイルパス

    Returns:
        str: 年文字列
    """
    return os.path.basename(os.path.dirname(file_path))


def build_external_file_path(
    bar_base_dir: str,
    symbol: str,
    session_type: str,
    year_str: str,
    trade_date_str: str,
) -> str:
    """
    外部指標 parquet の期待パスを構築します。

    Args:
        bar_base_dir (str): data ディレクトリなどのベースディレクトリ
        symbol (str): シンボル名
        session_type (str): DAY または NIGHT
        year_str (str): 年
        trade_date_str (str): YYYY-MM-DD

    Returns:
        str: 外部指標 parquet の期待パス
    """
    return os.path.join(
        bar_base_dir,
        "bars",
        "external_30s",
        symbol,
        year_str,
        f"{trade_date_str}-{session_type}.parquet",
    )


def load_parquet_if_exists(file_path: str) -> pl.LazyFrame | None:
    """
    parquet が存在する場合のみ遅延評価(LazyFrame)で読み込みます。

    Args:
        file_path (str): parquet ファイルパス

    Returns:
        pl.LazyFrame | None: 読み込んだ LazyFrame。存在しない場合は None
    """
    if not os.path.exists(file_path):
        return None
    return pl.scan_parquet(file_path)


def load_external_frames_for_day(
    bar_base_dir: str,
    session_type: str,
    year_str: str,
    trade_date_str: str,
    symbol_map: dict[str, str],
) -> dict[str, pl.LazyFrame]:
    """
    指定日・指定セッションの外部指標 LazyFrame 群を読み込みます。

    Args:
        bar_base_dir (str): data ベースディレクトリ
        session_type (str): DAY または NIGHT
        year_str (str): 年
        trade_date_str (str): YYYY-MM-DD
        symbol_map (dict[str, str]): features.py に渡す prefix -> 実ファイル上の symbol 名

    Returns:
        dict[str, pl.LazyFrame]: prefix -> LazyFrame
    """
    external_lfs: dict[str, pl.LazyFrame] = {}

    for prefix, symbol in symbol_map.items():
        path = build_external_file_path(
            bar_base_dir=bar_base_dir,
            symbol=symbol,
            session_type=session_type,
            year_str=year_str,
            trade_date_str=trade_date_str,
        )
        lf = load_parquet_if_exists(path)
        if lf is None:
            print(f"  [WARN] Missing external parquet: prefix={prefix}, symbol={symbol}, path={path}")
            continue

        external_lfs[prefix] = lf

    return external_lfs


def save_feature_parquet(
    feature_lf: pl.LazyFrame,
    output_base_dir: str,
    session_type: str,
    year_str: str,
    trade_date_str: str,
) -> str:
    """
    特徴量 parquet をストリーミング保存します。

    Args:
        feature_lf (pl.LazyFrame): 特徴量実行計画を持つ LazyFrame
        output_base_dir (str): 出力ベースディレクトリ
        session_type (str): DAY または NIGHT
        year_str (str): 年
        trade_date_str (str): YYYY-MM-DD

    Returns:
        str: 保存先パス
    """
    out_dir = os.path.join(
        output_base_dir,
        "features",
        "entry",
        year_str,
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{trade_date_str}-{session_type}.parquet")

    # sink_parquet を使用してストリーミング処理で書き出します。
    feature_lf.sort("bar_start_jst").sink_parquet(
        out_path,
        compression="zstd",
    )
    return out_path


def process_one_nk225_file(
    nk225_file_path: str,
    bar_base_dir: str,
    output_base_dir: str,
    symbol_map: dict[str, str],
    label_horizon: int,
) -> None:
    """
    NK225 の 1日1セッション分の parquet から学習用特徴量 parquet を生成します。

    Args:
        nk225_file_path (str): NK225 parquet ファイルパス
        bar_base_dir (str): data ベースディレクトリ
        output_base_dir (str): 出力ベースディレクトリ
        symbol_map (dict[str, str]): features.py に渡す prefix -> 実ファイル上の symbol 名
        label_horizon (int): ラベル horizon。本数単位
    """
    year_str = extract_year_from_path(nk225_file_path)
    trade_date_str, session_type = parse_date_and_session(nk225_file_path)

    print(f"Scanning NK225: {nk225_file_path}")

    nk225_lf = pl.scan_parquet(nk225_file_path)

    external_lfs = load_external_frames_for_day(
        bar_base_dir=bar_base_dir,
        session_type=session_type,
        year_str=year_str,
        trade_date_str=trade_date_str,
        symbol_map=symbol_map,
    )

    # 全ての外部指標が揃っていない場合は処理をスキップして出力しません。
    if len(external_lfs) < len(symbol_map):
        print(
            f"  [SKIP] 外部指標が揃っていません "
            f"({len(external_lfs)}/{len(symbol_map)})。特徴量の出力をスキップします。"
        )
        return

    print("  -> Building execution plan for features and labels...")
    feature_lf = build_entry_feature_frame(
        nk225_df=nk225_lf,
        external_frames=external_lfs if external_lfs else None,
        label_horizon=label_horizon,
    )

    print(f"  -> Executing and saving to {output_base_dir}...")
    out_path = save_feature_parquet(
        feature_lf=feature_lf,
        output_base_dir=output_base_dir,
        session_type=session_type,
        year_str=year_str,
        trade_date_str=trade_date_str,
    )
    print(f"  -> Saved: {out_path}")


def build_default_symbol_map() -> dict[str, str]:
    """
    features.py の prefix と、実際の external_30s ディレクトリ名との対応表を返します。

    Returns:
        dict[str, str]: prefix -> symbol
    """
    return {
        "sp500": "US500",
        "nasdaq": "NAS100",
        "xau": "XAUUSD",
        "xti": "XTIUSD",
    }


def filter_nk225_files(
    files: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
    session_type: str | None = None,
) -> list[str]:
    """
    NK225 parquet ファイル一覧に対して日付・セッションフィルタを適用します。

    Args:
        files (list[str]): NK225 parquet ファイル一覧
        date_from (str | None, optional): 開始日 YYYY-MM-DD
        date_to (str | None, optional): 終了日 YYYY-MM-DD
        session_type (str | None, optional): DAY / NIGHT

    Returns:
        list[str]: フィルタ後のファイル一覧
    """
    filtered: list[str] = []

    for path in files:
        trade_date_str, current_session_type = parse_date_and_session(path)

        if date_from is not None and trade_date_str < date_from:
            continue
        if date_to is not None and trade_date_str > date_to:
            continue
        if session_type is not None and current_session_type != session_type:
            continue

        filtered.append(path)

    return filtered


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
        "--session-type",
        default=None,
        choices=["DAY", "NIGHT"],
        help="DAY または NIGHT のみ処理したい場合に指定します。",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=240,  # 2時間後の効率比ラベル (30秒足で240本) を既定値にします。
        help="ラベル生成に使う未来参照本数。30秒足なので 60=30分, 120=60分, 240=120分。",
    )
    parser.add_argument(
        "--require-all-external",
        action="store_true",
        help="[非推奨] 現在はデフォルトで全銘柄必須となり、揃わない日は自動スキップされます。",
    )
    args = parser.parse_args()

    nk225_files = list(iter_nk225_bar_files(args.bar_base_dir))
    if not nk225_files:
        print("NK225 bar parquet が見つかりませんでした。")
        return

    nk225_files = filter_nk225_files(
        files=nk225_files,
        date_from=args.date_from,
        date_to=args.date_to,
        session_type=args.session_type,
    )

    if not nk225_files:
        print("条件に一致する NK225 bar parquet が見つかりませんでした。")
        return

    symbol_map = build_default_symbol_map()

    print(f"NK225 target files: {len(nk225_files)}")
    print(f"label_horizon: {args.label_horizon}")
    print(f"external symbol map: {symbol_map}")

    for nk225_file_path in nk225_files:
        process_one_nk225_file(
            nk225_file_path=nk225_file_path,
            bar_base_dir=args.bar_base_dir,
            output_base_dir=args.output_base_dir,
            symbol_map=symbol_map,
            label_horizon=args.label_horizon,
        )

    print("Entry feature parquet generation completed.")


if __name__ == "__main__":
    main()
