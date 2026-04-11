# train/entry/features.py
"""
File: train/entry/features.py

ソースコードの役割:
NK225 30秒足と MT5 外部指標30秒足を入力とし、
Transformer のエントリー可否判定に使う特徴量および目的変数を生成します。

本モジュールは以下の責務を持ちます。
1. NK225 の価格・出来高・約定フロー特徴量を生成する
2. 外部指標の価格・スプレッド・tick密度特徴量を生成する
3. NK225 と外部指標の相対強弱・追随遅れ特徴量を生成する
4. 欠損補完とセッション境界を考慮した安全な前処理を行う
5. 深層学習モデル向けに非定常な価格系列を定常化（差分変換）する
6. 学習用ラベル（将来の効率比 / Efficiency Ratio）を生成する
"""

from __future__ import annotations

import polars as pl


EPS = 1e-9

# 30秒足の本数換算
BAR_1M = 2      # 1分
BAR_2M = 4      # 2分
BAR_5M = 10     # 5分
BAR_10M = 20    # 10分
BAR_15M = 30    # 15分
BAR_30M = 60    # 30分
BAR_60M = 120   # 60分


def _safe_ratio(numer: pl.Expr, denom: pl.Expr, default: float = 0.0) -> pl.Expr:
    """
    0除算を避けた比率を返します。
    
    Args:
        numer (pl.Expr): 分子の式。
        denom (pl.Expr): 分母の式。
        default (float, optional): 分母が0に近い場合に返すデフォルト値。デフォルトは 0.0。
        
    Returns:
        pl.Expr: 計算された比率の式。
    """
    return (
        pl.when(denom.abs() > EPS)
        .then(numer / denom)
        .otherwise(pl.lit(default))
    )


def _safe_log1p(col: str) -> pl.Expr:
    """
    負値を許容しない列に対して安全な log1p (log(1 + x)) を返します。
    
    Args:
        col (str): 対象の列名。
        
    Returns:
        pl.Expr: 計算された対数の式。負値の場合は None を返します。
    """
    return pl.when(pl.col(col) > -1.0).then((pl.col(col) + 1.0).log()).otherwise(None)


def _log_ret(col: str, periods: int = 1) -> pl.Expr:
    """
    対数リターンを返します。
    
    Args:
        col (str): 対象の列名。
        periods (int, optional): 差分をとる期間（ラグ）。デフォルトは 1。
        
    Returns:
        pl.Expr: 対数リターンの式。
    """
    return (
        (pl.col(col).log() - pl.col(col).shift(periods).log())
        .alias(f"{col}_log_ret_{periods}")
    )


def _pct_ret(col: str, periods: int = 1) -> pl.Expr:
    """
    単純リターン (変化率) を返します。
    
    Args:
        col (str): 対象の列名。
        periods (int, optional): 差分をとる期間（ラグ）。デフォルトは 1。
        
    Returns:
        pl.Expr: 単純リターンの式。
    """
    return (
        _safe_ratio(
            pl.col(col) - pl.col(col).shift(periods),
            pl.col(col).shift(periods),
        ).alias(f"{col}_ret_{periods}")
    )


def _rolling_zscore(col: str, window: int, min_periods: int | None = None) -> list[pl.Expr]:
    """
    ローリング平均・標準偏差を使った z-score (標準化) を返します。
    
    Args:
        col (str): 対象の列名。
        window (int): ローリングウィンドウのサイズ。
        min_periods (int | None, optional): 最小計算期間。指定がない場合は window の 1/4 (最低5)。
        
    Returns:
        list[pl.Expr]: 平均、標準偏差、z-score を計算する式のリスト。
    """
    mp = min_periods or max(5, window // 4)
    mean_col = f"__{col}_mean_{window}"
    std_col = f"__{col}_std_{window}"
    z_col = f"{col}_z_{window}"
    
    mean_expr = pl.col(col).rolling_mean(window_size=window, min_periods=mp)
    std_expr = pl.col(col).rolling_std(window_size=window, min_periods=mp)
    
    return [
        mean_expr.alias(mean_col),
        std_expr.alias(std_col),
        pl.when(std_expr > EPS)
        .then((pl.col(col) - mean_expr) / std_expr)
        .otherwise(0.0)
        .alias(z_col),
    ]


def _linear_slope(col: str, window: int) -> pl.Expr:
    """
    簡易傾き特徴量を計算します。
    厳密な最小二乗ではなく、window 本前との差分を window で割った近似傾きです。
    まずは軽量・堅牢性を優先します。
    
    Args:
        col (str): 対象の列名。
        window (int): 比較する過去の期間。
        
    Returns:
        pl.Expr: 簡易傾きの式。
    """
    return (
        (pl.col(col) - pl.col(col).shift(window)) / float(window)
    ).alias(f"{col}_slope_{window}")


def _session_group_keys() -> list[str]:
    """セッション内 forward fill / shift の境界キーを返します。"""
    return ["session_date_jst", "session_type"]


def prepare_external_symbol_frame(external_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """
    外部指標の 30秒 bar DataFrame を join 用に正規化します。

    Args:
        external_df (pl.DataFrame): build_mt5_external_bars.py の出力 DataFrame。
        prefix (str): 外部指標のプレフィックス（例: 'sp500', 'nasdaq', 'dow', 'xau', 'xti'）。

    Returns:
        pl.DataFrame: join 済み前提の prefix 付き DataFrame。
    """
    return external_df.select(
        [
            "bar_start_jst",
            "session_date_jst",
            "session_type",
            pl.col("open").alias(f"{prefix}_open"),
            pl.col("high").alias(f"{prefix}_high"),
            pl.col("low").alias(f"{prefix}_low"),
            pl.col("close").alias(f"{prefix}_close"),
            pl.col("spread_close").alias(f"{prefix}_spread_close"),
            pl.col("tick_count").alias(f"{prefix}_tick_count"),
            pl.col("is_complete").alias(f"{prefix}_is_complete"),
        ]
    )


def join_external_frames(
    nk_df: pl.DataFrame,
    external_frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    NK225 の 30秒 bar に外部指標 30秒 bar を左結合します。
    
    Args:
        nk_df (pl.DataFrame): NK225のデータフレーム。
        external_frames (dict[str, pl.DataFrame]): 外部指標のデータフレームを格納した辞書。
        
    Returns:
        pl.DataFrame: 結合・補完済みのデータフレーム。
        
    Notes:
        - build_*_bars.py はどちらも bar_start_jst / session_date_jst / session_type を持ちます。
        - 同じ 30秒境界で集約済みなので asof ではなく等値 join を使います。
    """
    df = nk_df
    join_keys = ["bar_start_jst", "session_date_jst", "session_type"]

    for prefix, ext_df in external_frames.items():
        df = df.join(
            prepare_external_symbol_frame(ext_df, prefix=prefix),
            on=join_keys,
            how="left",
        )

    # 外部指標はセッション内で前方補完します。
    # セッションを跨いだ補完は行いません。
    fill_cols: list[str] = []
    for prefix in external_frames.keys():
        fill_cols.extend(
            [
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
                f"{prefix}_spread_close",
                f"{prefix}_tick_count",
            ]
        )

    for col_name in fill_cols:
        df = df.with_columns(
            pl.col(col_name).forward_fill().over(_session_group_keys()).alias(col_name)
        )

    # 補完後も先頭側で欠損が残る場合は 0 埋めではなく NULL のままにし、
    # 後段の drop_nulls 対象またはモデル入力前スケーラへ渡します。
    return df


def add_nk225_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    NK225 の基本特徴量を生成します。

    Args:
        df (pl.DataFrame): 結合済みの基礎データフレーム。
        
    Returns:
        pl.DataFrame: 基本特徴量が追加されたデータフレーム。
        
    前提列:
        open, high, low, close, volume, tick_count,
        buy_volume, sell_volume, signed_volume, vwap_30s
    """
    df = df.with_columns(
        [
            # 価格系の基本
            ((pl.col("high") - pl.col("low"))).alias("price_spread"),
            _safe_ratio(
                pl.col("close") - pl.col("low"),
                pl.col("high") - pl.col("low"),
                default=0.5,
            ).alias("close_pos_in_bar"),
            _safe_ratio(
                (pl.col("close") - pl.col("open")).abs(),
                pl.col("high") - pl.col("low"),
                default=0.0,
            ).alias("body_ratio"),
            ((pl.col("close") + pl.col("high") + pl.col("low")) / 3.0).alias("typical_price"),

            # 出来高・約定フロー
            _safe_ratio(pl.col("buy_volume"), pl.col("volume")).alias("buy_ratio"),
            _safe_ratio(pl.col("sell_volume"), pl.col("volume")).alias("sell_ratio"),
            _safe_ratio(pl.col("signed_volume"), pl.col("volume")).alias("signed_volume_ratio"),
            _safe_ratio(pl.col("volume"), pl.col("tick_count")).alias("trade_size_mean"),
            _safe_ratio(pl.col("buy_volume") + 1.0, pl.col("sell_volume") + 1.0).log().alias("volume_pressure"),
            (pl.col("close") - pl.col("vwap_30s")).alias("dist_vwap_30s"),
            _safe_ratio(pl.col("close") - pl.col("vwap_30s"), pl.col("close")).alias("dist_vwap_30s_pct"),

            # 対数系
            (pl.col("volume") + 1.0).log().alias("log_vol"),
            (pl.col("buy_volume") + 1.0).log().alias("log_buy_vol"),
            (pl.col("sell_volume") + 1.0).log().alias("log_sell_vol"),
        ]
    )

    df = df.with_columns(
        [
            _log_ret("close", 1).alias("log_ret"),
            _pct_ret("close", BAR_1M).alias("ret_1m"),
            _pct_ret("close", BAR_2M).alias("ret_2m"),
            _pct_ret("close", BAR_5M).alias("ret_5m"),
            _pct_ret("close", BAR_10M).alias("ret_10m"),
            _pct_ret("close", BAR_30M).alias("ret_30m"),
            _pct_ret("close", BAR_60M).alias("ret_60m"),
            _linear_slope("close", BAR_5M).alias("close_slope_5m"),
            _linear_slope("close", BAR_10M).alias("close_slope_10m"),
            _linear_slope("signed_volume", BAR_5M).alias("signed_volume_slope_5m"),
            _linear_slope("tick_count", BAR_5M).alias("tick_count_slope_5m"),
        ]
    )

    df = df.with_columns(
        [
            # 実現ボラの軽量版
            pl.col("log_ret").rolling_std(window_size=BAR_5M, min_periods=5).alias("realized_vol_5m"),
            pl.col("log_ret").rolling_std(window_size=BAR_10M, min_periods=10).alias("realized_vol_10m"),
            pl.col("log_ret").rolling_std(window_size=BAR_30M, min_periods=20).alias("realized_vol_30m"),

            # Garman-Klass 近似
            (
                0.5 * (pl.col("high").log() - pl.col("low").log()) ** 2
                - (2.0 * 0.6931471805599453 - 1.0) * (pl.col("close").log() - pl.col("open").log()) ** 2
            ).clip(lower_bound=0.0).sqrt().alias("garman_klass_vol"),

            # 累積差分系
            pl.col("signed_volume").rolling_sum(window_size=BAR_2M, min_periods=2).alias("signed_volume_sum_2m"),
            pl.col("signed_volume").rolling_sum(window_size=BAR_5M, min_periods=5).alias("signed_volume_sum_5m"),
            pl.col("signed_volume").rolling_sum(window_size=BAR_10M, min_periods=10).alias("signed_volume_sum_10m"),
            pl.col("volume").rolling_sum(window_size=BAR_5M, min_periods=5).alias("volume_sum_5m"),
            pl.col("tick_count").rolling_mean(window_size=BAR_5M, min_periods=5).alias("tick_count_ma_5m"),

            # VWAP 乖離の多窓
            (
                pl.col("close")
                - (
                    (pl.col("typical_price") * pl.col("volume")).rolling_sum(window_size=BAR_15M, min_periods=10)
                    / pl.col("volume").rolling_sum(window_size=BAR_15M, min_periods=10)
                )
            ).alias("dist_vwap_15m"),
        ]
    )

    df = df.with_columns(
        [
            _safe_ratio(pl.col("tick_count"), pl.col("tick_count_ma_5m")).alias("tick_speed_ratio"),
            (pl.col("signed_volume_ratio") - pl.col("signed_volume_ratio").shift(1)).alias("signed_volume_ratio_change"),
            (pl.col("buy_ratio") - pl.col("buy_ratio").shift(1)).alias("buy_share_change"),
            (pl.col("sell_ratio") - pl.col("sell_ratio").shift(1)).alias("sell_share_change"),
            _safe_ratio(pl.col("price_spread"), pl.col("volume") + 1.0).alias("range_per_volume"),
            _safe_ratio(pl.col("tick_count"), pl.col("volume") + 1.0).alias("tick_per_volume"),
        ]
    )

    # z-score 系
    for col_name, window in [
        ("volume", BAR_5M),
        ("volume", BAR_30M),
        ("tick_count", BAR_5M),
        ("signed_volume", BAR_5M),
        ("signed_volume_ratio", BAR_5M),
        ("price_spread", BAR_5M),
        ("dist_vwap_30s", BAR_5M),
    ]:
        df = df.with_columns(_rolling_zscore(col_name, window))

    # セッション内 VWAP 乖離
    session_turnover = "__session_turnover_cum"
    session_volume = "__session_volume_cum"
    df = df.with_columns(
        [
            (pl.col("typical_price") * pl.col("volume")).cum_sum().over(_session_group_keys()).alias(session_turnover),
            pl.col("volume").cum_sum().over(_session_group_keys()).alias(session_volume),
        ]
    ).with_columns(
        [
            _safe_ratio(pl.col(session_turnover), pl.col(session_volume), default=0.0).alias("session_vwap"),
        ]
    ).with_columns(
        [
            (pl.col("close") - pl.col("session_vwap")).alias("dist_vwap_session"),
            _safe_ratio(pl.col("close") - pl.col("session_vwap"), pl.col("close")).alias("dist_vwap_session_pct"),
        ]
    )

    return df


def add_external_symbol_features(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """
    外部指標 1銘柄分の特徴量を生成します。

    Args:
        df (pl.DataFrame): データフレーム。
        prefix (str): 外部指標のプレフィックス。
        
    Returns:
        pl.DataFrame: 外部指標の特徴量が追加されたデータフレーム。
        
    前提列:
        {prefix}_open, {prefix}_high, {prefix}_low, {prefix}_close,
        {prefix}_spread_close, {prefix}_tick_count

    Notes:
        build_mt5_external_bars.py の volume は 0 固定なので使いません。
    """
    close_col = f"{prefix}_close"
    open_col = f"{prefix}_open"
    high_col = f"{prefix}_high"
    low_col = f"{prefix}_low"
    spread_col = f"{prefix}_spread_close"
    tick_col = f"{prefix}_tick_count"

    df = df.with_columns(
        [
            ((pl.col(high_col) - pl.col(low_col))).alias(f"{prefix}_price_spread"),
            _safe_ratio(
                pl.col(close_col) - pl.col(low_col),
                pl.col(high_col) - pl.col(low_col),
                default=0.5,
            ).alias(f"{prefix}_close_pos_in_bar"),
            _safe_ratio(
                (pl.col(close_col) - pl.col(open_col)).abs(),
                pl.col(high_col) - pl.col(low_col),
                default=0.0,
            ).alias(f"{prefix}_body_ratio"),
            (pl.col(close_col).log() - pl.col(close_col).shift(1).log()).alias(f"{prefix}_log_ret"),
            _safe_ratio(pl.col(close_col) - pl.col(close_col).shift(BAR_1M), pl.col(close_col).shift(BAR_1M)).alias(f"{prefix}_ret_1m"),
            _safe_ratio(pl.col(close_col) - pl.col(close_col).shift(BAR_5M), pl.col(close_col).shift(BAR_5M)).alias(f"{prefix}_ret_5m"),
            _safe_ratio(pl.col(close_col) - pl.col(close_col).shift(BAR_10M), pl.col(close_col).shift(BAR_10M)).alias(f"{prefix}_ret_10m"),
            _linear_slope(close_col, BAR_5M).alias(f"{prefix}_close_slope_5m"),
            _linear_slope(close_col, BAR_10M).alias(f"{prefix}_close_slope_10m"),
            _linear_slope(tick_col, BAR_5M).alias(f"{prefix}_tick_count_slope_5m"),
        ]
    )

    df = df.with_columns(
        [
            pl.col(f"{prefix}_log_ret").rolling_std(window_size=BAR_5M, min_periods=5).alias(f"{prefix}_realized_vol_5m"),
            pl.col(f"{prefix}_log_ret").rolling_std(window_size=BAR_10M, min_periods=10).alias(f"{prefix}_realized_vol_10m"),
            pl.col(tick_col).rolling_mean(window_size=BAR_5M, min_periods=5).alias(f"{prefix}_tick_count_ma_5m"),
        ]
    )

    # NOTE:
    # LazyFrame では、同一 with_columns() 内で新規 alias 列を即参照すると
    # スキーマ解決時に ColumnNotFoundError になる場合があります。
    # そのため、移動平均列の作成と tick_speed_ratio の作成を 2 段に分けます。
    df = df.with_columns(
        [
            _safe_ratio(pl.col(tick_col), pl.col(f"{prefix}_tick_count_ma_5m")).alias(f"{prefix}_tick_speed_ratio"),
        ]
    )

    for col_name, window in [
        (tick_col, BAR_5M),
        (spread_col, BAR_5M),
        (f"{prefix}_price_spread", BAR_5M),
        (f"{prefix}_ret_1m", BAR_5M),
    ]:
        df = df.with_columns(_rolling_zscore(col_name, window))

    return df


def add_cross_asset_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    NK225 と外部指標の相対強弱・追随遅れ特徴量を生成します。

    Args:
        df (pl.DataFrame): 外部指標特徴量追加済みのデータフレーム。
        
    Returns:
        pl.DataFrame: クロスアセット特徴量が追加されたデータフレーム。
        
    利用想定 prefix:
        sp500, nasdaq, dow, xau, xti
    """
    # NOTE:
    # LazyFrame.columns は内部でスキーマ解決を走らせるため PerformanceWarning の原因になります。
    # collect_schema().names() を使って明示的に列名だけ取得します。
    if isinstance(df, pl.LazyFrame):
        available = set(df.collect_schema().names())
    else:
        available = set(df.columns)

    def has(*cols: str) -> bool:
        return all(col in available for col in cols)

    exprs: list[pl.Expr] = []

    # 米株3指数の合成 risk-on
    if has("sp500_ret_1m", "nasdaq_ret_1m", "dow_ret_1m"):
        exprs.extend(
            [
                (
                    pl.col("sp500_ret_1m") * 0.4
                    + pl.col("nasdaq_ret_1m") * 0.4
                    + pl.col("dow_ret_1m") * 0.2
                ).alias("us_risk_on_ret_1m"),
                (
                    pl.col("sp500_ret_5m") * 0.4
                    + pl.col("nasdaq_ret_5m") * 0.4
                    + pl.col("dow_ret_5m") * 0.2
                ).alias("us_risk_on_ret_5m"),
                pl.concat_list(
                    [
                        pl.col("sp500_ret_1m"),
                        pl.col("nasdaq_ret_1m"),
                        pl.col("dow_ret_1m"),
                    ]
                ).list.std().alias("us_index_dispersion_1m"),
            ]
        )

    # NK225 vs 米株指数の相対強弱
    if has("ret_1m", "sp500_ret_1m"):
        exprs.extend(
            [
                (pl.col("ret_1m") - pl.col("sp500_ret_1m")).alias("rel_strength_vs_sp500_1m"),
                (pl.col("ret_5m") - pl.col("sp500_ret_5m")).alias("rel_strength_vs_sp500_5m"),
                (pl.col("ret_1m") - pl.col("sp500_ret_1m").shift(1)).alias("lead_lag_vs_sp500_1m_lag1"),
                (pl.col("ret_1m") - pl.col("sp500_ret_1m").shift(2)).alias("lead_lag_vs_sp500_1m_lag2"),
            ]
        )

    if has("ret_1m", "nasdaq_ret_1m"):
        exprs.extend(
            [
                (pl.col("ret_1m") - pl.col("nasdaq_ret_1m")).alias("rel_strength_vs_nasdaq_1m"),
                (pl.col("ret_5m") - pl.col("nasdaq_ret_5m")).alias("rel_strength_vs_nasdaq_5m"),
                (pl.col("ret_1m") - pl.col("nasdaq_ret_1m").shift(1)).alias("lead_lag_vs_nasdaq_1m_lag1"),
                (pl.col("ret_1m") - pl.col("nasdaq_ret_1m").shift(2)).alias("lead_lag_vs_nasdaq_1m_lag2"),
            ]
        )

    if has("ret_1m", "dow_ret_1m"):
        exprs.extend(
            [
                (pl.col("ret_1m") - pl.col("dow_ret_1m")).alias("rel_strength_vs_dow_1m"),
                (pl.col("ret_5m") - pl.col("dow_ret_5m")).alias("rel_strength_vs_dow_5m"),
            ]
        )

    # 金・原油のショック方向
    if has("xau_ret_1m", "sp500_ret_1m"):
        exprs.extend(
            [
                (pl.col("xau_ret_1m") - pl.col("sp500_ret_1m")).alias("gold_vs_sp500_divergence_1m"),
                (pl.col("xau_ret_5m") - pl.col("sp500_ret_5m")).alias("gold_vs_sp500_divergence_5m"),
            ]
        )

    if has("xti_ret_1m", "sp500_ret_1m"):
        exprs.extend(
            [
                (pl.col("xti_ret_1m") - pl.col("sp500_ret_1m")).alias("oil_vs_sp500_divergence_1m"),
                (pl.col("xti_ret_5m") - pl.col("sp500_ret_5m")).alias("oil_vs_sp500_divergence_5m"),
            ]
        )

    # リスクオン/オフ一致度
    if has("sp500_ret_1m", "nasdaq_ret_1m", "dow_ret_1m"):
        exprs.append(
            (
                pl.col("sp500_ret_1m").sign()
                + pl.col("nasdaq_ret_1m").sign()
                + pl.col("dow_ret_1m").sign()
            ).alias("us_sign_agreement_1m")
        )

    if exprs:
        df = df.with_columns(exprs)

    return df


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    時間帯・セッション特徴量を生成します。
    
    Args:
        df (pl.DataFrame): データフレーム。
        
    Returns:
        pl.DataFrame: カレンダー特徴量が追加されたデータフレーム。
    """
    minute_of_day = (
        pl.col("bar_start_jst").dt.hour().cast(pl.Int32) * 60
        + pl.col("bar_start_jst").dt.minute().cast(pl.Int32)
    ).alias("minute_of_day")

    df = df.with_columns(
        [
            minute_of_day,
            pl.col("bar_start_jst").dt.weekday().alias("weekday"),
            (pl.col("session_type") == "DAY").cast(pl.Int8).alias("is_day_session"),
            (pl.col("session_type") == "NIGHT").cast(pl.Int8).alias("is_night_session"),
        ]
    )

    # DAY: 08:45開始 / NIGHT: 17:00開始
    day_open_min = 8 * 60 + 45
    night_open_min = 17 * 60

    df = df.with_columns(
        [
            pl.when(pl.col("session_type") == "DAY")
            .then(pl.col("minute_of_day") - day_open_min)
            .otherwise(
                pl.when(pl.col("minute_of_day") >= night_open_min)
                .then(pl.col("minute_of_day") - night_open_min)
                .otherwise(pl.col("minute_of_day") + (24 * 60 - night_open_min))
            )
            .alias("minutes_from_open"),
        ]
    )

    df = df.with_columns(
        [
            (2.0 * 3.141592653589793 * pl.col("minute_of_day") / 1440.0).sin().alias("tod_sin"),
            (2.0 * 3.141592653589793 * pl.col("minute_of_day") / 1440.0).cos().alias("tod_cos"),
            (2.0 * 3.141592653589793 * pl.col("weekday") / 7.0).sin().alias("dow_sin"),
            (2.0 * 3.141592653589793 * pl.col("weekday") / 7.0).cos().alias("dow_cos"),
            (pl.col("minutes_from_open") <= 30).cast(pl.Int8).alias("is_opening_window"),
            ((pl.col("session_type") == "DAY") & (pl.col("minute_of_day") >= 11 * 60)).cast(pl.Int8).alias("is_lunch_edge"),
            ((pl.col("session_type") == "NIGHT") & (pl.col("minute_of_day") <= 3 * 60)).cast(pl.Int8).alias("is_night_late"),
        ]
    )

    return df


def finalize_feature_frame(df: pl.DataFrame) -> pl.DataFrame:
    """
    学習直前の欠損整理を行います。
    
    Args:
        df (pl.DataFrame): 全特徴量が追加されたデータフレーム。
        
    Returns:
        pl.DataFrame: 欠損処理済みの最終データフレーム。

    欠損処理方針:
        - OHLC などの主系列欠損行は破棄
        - 派生特徴量の初期窓欠損は 0.0 埋め
        - 外部系列未到着の先頭欠損も 0.0 埋め
    """
    required_cols = [
        "bar_start_jst",
        "session_date_jst",
        "session_type",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tick_count",
        "buy_volume",
        "sell_volume",
        "signed_volume",
        "vwap_30s",
    ]

    df = df.drop_nulls(subset=required_cols)

    # 識別子列以外の数値列は 0 埋め
    numeric_cols = [
        col_name
        for col_name, dtype in df.schema.items()
        if dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
        and col_name not in {"weekday", "minute_of_day"}
    ]

    return df.with_columns([pl.col(col_name).fill_null(0) for col_name in numeric_cols])


def add_efficiency_ratio_label(df: pl.DataFrame, horizon: int = 240) -> pl.DataFrame:
    """
    将来のトレンド直進性（効率比 / Efficiency Ratio）をラベルとして生成します。
    
    Args:
        df (pl.DataFrame): データフレーム。
        horizon (int): 未来の参照期間（デフォルト240本 = 30秒足で2時間）。
        
    Returns:
        pl.DataFrame: 効率比ラベルが追加されたデータフレーム。
    """
    fwd_displacement = (pl.col("close").shift(-horizon) - pl.col("close")).abs()
    
    path_length = (
        (pl.col("close") - pl.col("close").shift(1)).abs()
        .rolling_sum(window_size=horizon)
        .shift(-horizon)
    )
    
    return df.with_columns(
        [
            _safe_ratio(fwd_displacement, path_length).alias(f"label_efficiency_{horizon}")
        ]
    )


def make_prices_stationary(df: pl.DataFrame, external_prefixes: list[str] | None = None) -> pl.DataFrame:
    """
    非定常（Non-Stationary）な生の価格列を、1つ前の行からの差分（1階差分）に変換し定常化します。
    
    Args:
        df (pl.DataFrame): データフレーム。
        external_prefixes (list[str] | None): 外部指標のプレフィックスリスト。
        
    Returns:
        pl.DataFrame: 価格が差分に変換され、先頭の欠損行がドロップされたデータフレーム。
    """
    price_cols = [
        "open", "high", "low", "close", 
        "vwap_30s", "typical_price", "session_vwap"
    ]
    
    if external_prefixes:
        for prefix in external_prefixes:
            price_cols.extend([
                f"{prefix}_open", f"{prefix}_high", 
                f"{prefix}_low", f"{prefix}_close"
            ])
            
    # NOTE: add_cross_asset_features と同様に LazyFrame 時の PerformanceWarning を回避
    if isinstance(df, pl.LazyFrame):
        available_cols = set(df.collect_schema().names())
    else:
        available_cols = set(df.columns)
        
    # 実際にDataFrameに存在する列のみを対象とする
    target_cols = [col for col in price_cols if col in available_cols]
    
    # 1行前からの差分に置き換える（絶対値のスケール依存を排除）
    df = df.with_columns(
        [(pl.col(col) - pl.col(col).shift(1)).alias(col) for col in target_cols]
    )
    
    # shift(1) により先頭行が必ず null になるためドロップ
    return df.drop_nulls(subset=target_cols)


def build_entry_feature_frame(
    nk225_df: pl.DataFrame,
    external_frames: dict[str, pl.DataFrame] | None = None,
    label_horizon: int = 240,
) -> pl.DataFrame:
    """
    学習用の最終特徴量 DataFrame を構築します。

    Args:
        nk225_df (pl.DataFrame): NK225のデータフレーム。
        external_frames (dict[str, pl.DataFrame] | None, optional): 外部指標データフレームの辞書。
        label_horizon (int, optional): ラベル計算用の未来参照ウィンドウ。デフォルトは240（2時間）。
        
    Returns:
        pl.DataFrame: Transformer入力用の最終特徴量データフレーム。
        
    external_frames のキー例:
        {
            "sp500": sp500_df,
            "nasdaq": nasdaq_df,
            "dow": dow_df,
            "xau": xau_df,
            "xti": xti_df,
        }
    """
    df = nk225_df.sort(["session_date_jst", "session_type", "bar_start_jst"])

    if external_frames:
        df = join_external_frames(df, external_frames)

    df = add_nk225_base_features(df)

    if external_frames:
        for prefix in external_frames.keys():
            df = add_external_symbol_features(df, prefix=prefix)
        df = add_cross_asset_features(df)

    df = add_calendar_features(df)
    
    # --- ラベル（目的変数）の生成 ---
    df = add_efficiency_ratio_label(df, horizon=label_horizon)
    
    # 生の価格列を差分（定常化）に変換し、欠損となった先頭行をドロップ
    prefixes = list(external_frames.keys()) if external_frames else None
    df = make_prices_stationary(df, external_prefixes=prefixes)
    
    # ラベル計算で未来参照したことにより生じた、末尾の欠損行を確実にドロップ
    df = df.drop_nulls(subset=[f"label_efficiency_{label_horizon}"])
    
    df = finalize_feature_frame(df)
    return df
