# train/entry/dataset.py
"""
File: train/entry/dataset.py

ソースコードの役割:
特徴量Parquetファイルを読み込み、Transformerに入力可能な
3次元テンソル (Batch, SeqLen, Features) を生成するPyTorchデータセットです。

セッション（1日）を跨いだシーケンス生成（未来情報のリークや不連続な価格の混入）を
防ぐため、1ファイルごとに独立してシーケンスをスライディングウィンドウで抽出します。
さらに、Trainデータから算出した統計量（平均・標準偏差）を用いてZ-score正規化を
実行し、検証・テストデータへの情報漏洩（Data Leakage）を防止します。
また、分類タスクへの対応として、Trainデータから算出した閾値（上位パーセンタイル）を
用いて連続値ラベルを2値化（0 or 1）する機能も提供します。
"""

from typing import List, Tuple, Dict, Optional
import polars as pl
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def compute_train_statistics(train_files: List[str], label_col: str, top_percentile: float = 80.0) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Trainデータ全体から各特徴量の平均・標準偏差と、ラベルの2値化閾値を事前計算します。

    Args:
        train_files (List[str]): 学習用データのParquetファイルパスのリスト。
        label_col (str): 目的変数の列名。統計量計算から除外するため。
        top_percentile (float, optional): 上位何%を正例とするかのパーセンタイル値。デフォルトは 80.0 (上位20%)。

    Returns:
        Tuple[Dict[str, float], Dict[str, float], float]:
            - means: 各特徴量の平均値を格納した辞書。
            - stds: 各特徴量の標準偏差を格納した辞書。
            - label_threshold: 目的変数を2値化するための閾値。
    """
    # スキーマ（列情報）を取得するために先頭ファイルを読み込む
    df_sample = pl.read_parquet(train_files[0])
    feature_cols = [
        name for name, dtype in df_sample.schema.items()
        if dtype.is_numeric() and name != label_col
    ]

    # 遅延評価（Lazy API）を用いて複数ファイルの統計量を効率的に計算
    lf = pl.scan_parquet(train_files)
    mean_exprs = [pl.col(c).mean().alias(f"{c}_mean") for c in feature_cols]
    std_exprs = [pl.col(c).std().alias(f"{c}_std") for c in feature_cols]
    label_expr = pl.col(label_col).quantile(top_percentile / 100.0).alias("label_threshold")

    stats_df = lf.select(mean_exprs + std_exprs + [label_expr]).collect()

    means = {c: stats_df[f"{c}_mean"][0] for c in feature_cols}
    
    # ゼロ除算を避けるため、標準偏差が 0 または null の場合は 1.0 にフォールバックします
    stds = {c: stats_df[f"{c}_std"][0] if stats_df[f"{c}_std"][0] else 1.0 for c in feature_cols}
    stds = {c: val if val != 0.0 else 1.0 for c, val in stds.items()}
    
    label_threshold = stats_df["label_threshold"][0]

    return means, stds, label_threshold


class SessionSequenceDataset(Dataset):
    """1セッション（1Parquetファイル）からシーケンスを抽出するPyTorchデータセットです。"""

    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 60,
        label_col: str = "label_efficiency_240",
        feature_means: Optional[Dict[str, float]] = None,
        feature_stds: Optional[Dict[str, float]] = None,
        label_threshold: Optional[float] = None
    ):
        """
        Args:
            parquet_path (str): 読み込むParquetファイルのパス。
            seq_len (int, optional): シーケンス長（過去何本分の足を入力するか）。デフォルトは 60。
            label_col (str, optional): 目的変数の列名。デフォルトは "label_efficiency_240"。
            feature_means (Optional[Dict[str, float]], optional): 正規化に使用する平均値の辞書。
            feature_stds (Optional[Dict[str, float]], optional): 正規化に使用する標準偏差の辞書。
            label_threshold (Optional[float], optional): ラベルを2値化(0/1)するための閾値。
        """
        df = pl.read_parquet(parquet_path)
        
        # モデル入力に適さない非数値列（日時、文字列など）やラベル列を除外し、特徴量(X)を特定
        feature_cols = [
            name for name, dtype in df.schema.items()
            if dtype.is_numeric() and name != label_col
        ]
        self.feature_names = feature_cols
        
        # Z-score Normalization (標準化)
        # 外部から渡された学習データの統計量を用いて正規化し、Data Leakageを防止
        if feature_means is not None and feature_stds is not None:
            scale_exprs = [
                ((pl.col(col) - feature_means.get(col, 0.0)) / feature_stds.get(col, 1.0)).alias(col)
                for col in feature_cols
            ]
            df = df.with_columns(scale_exprs)
            
        # 連続値のラベルを、学習データから算出した閾値で2値化(0.0 or 1.0)
        if label_threshold is not None:
            df = df.with_columns([
                (pl.col(label_col) >= label_threshold).cast(pl.Float32).alias(label_col)
            ])

        # PyTorchテンソルへの変換 (Float32)
        self.X_data = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
        self.y_data = torch.tensor(df.select([label_col]).to_numpy(), dtype=torch.float32)
        
        self.seq_len = seq_len
        
        # 生成可能なサンプル数（行数がseq_len未満の場合は0になる）
        self.num_samples = max(0, len(df) - seq_len + 1)

    def __len__(self) -> int:
        """データセット内のサンプル数を返します。"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスからseq_len分の特徴量シーケンスと、
        シーケンス最後の足の時点でのラベルを返します。

        Args:
            idx (int): 取得するサンプルのインデックス。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特徴量シーケンス, ラベル)
        """
        # (SeqLen, Features)の3次元入力用テンソル
        X_seq = self.X_data[idx : idx + self.seq_len]
        
        # ラベルは「シーケンスの最後（現在時刻）」に対応する未来の効率比
        y_val = self.y_data[idx + self.seq_len - 1]
        
        return X_seq, y_val


def create_dataloaders(
    file_paths: List[str],
    seq_len: int = 60,
    batch_size: int = 64,
    train_days: int = 60,
    valid_days: int = 20,
    test_days: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    ファイルリストを時系列順に分割し、Train/Valid/TestのDataLoaderを作成します。
    
    Args:
        file_paths (List[str]): Parquetファイルのパスリスト。
        seq_len (int, optional): シーケンス長。デフォルトは 60。
        batch_size (int, optional): バッチサイズ。デフォルトは 64。
        train_days (int, optional): 学習データの日数（ファイル数）。デフォルトは 60。
        valid_days (int, optional): 検証データの日数（ファイル数）。デフォルトは 20。
        test_days (int, optional): テストデータの日数（ファイル数）。デフォルトは 1。
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, int]: 
            (Train DataLoader, Valid DataLoader, Test DataLoader, 特徴量の次元数)
            
    Raises:
        ValueError: 指定された日数に対してファイル数が不足している場合、または抽出可能なシーケンスがない場合。
    """
    # 日付が含まれるパス名でソートし、時系列を保証
    sorted_files = sorted(file_paths)
    total_required_days = train_days + valid_days + test_days
    
    if len(sorted_files) < total_required_days:
        raise ValueError(
            f"ファイル数が不足しています。必要:{total_required_days}, 実際:{len(sorted_files)}"
        )
        
    # 最新（末尾）のデータから遡って分割し、Walk-Forwardアプローチの基盤を作る
    target_files = sorted_files[-total_required_days:]
    
    train_files = target_files[:train_days]
    valid_files = target_files[train_days : train_days + valid_days]
    test_files = target_files[-test_days:]
    
    print(f"Data split: Train={len(train_files)}days, Valid={len(valid_files)}days, Test={len(test_files)}days")

    # Trainデータのみから統計量を計算（Data Leakage防止）
    print("Computing scaling statistics and label threshold from Train data...")
    means, stds, label_threshold = compute_train_statistics(train_files, label_col="label_efficiency_240", top_percentile=80.0)
    print(f"Computed label threshold (top 20%): {label_threshold:.5f}")

    def build_concat_dataset(files: List[str], feature_means: Dict[str, float], feature_stds: Dict[str, float], label_threshold: float) -> Optional[ConcatDataset]:
        """複数ファイルのDatasetを1つのDatasetに結合するヘルパー関数。"""
        datasets = []
        for f in files:
            ds = SessionSequenceDataset(
                f, 
                seq_len=seq_len, 
                feature_means=feature_means, 
                feature_stds=feature_stds,
                label_threshold=label_threshold
            )
            # データが存在する（seq_len以上の行数がある）場合のみ追加
            if len(ds) > 0:
                datasets.append(ds)
                
        if not datasets:
            return None
        return ConcatDataset(datasets)

    train_ds = build_concat_dataset(train_files, means, stds, label_threshold)
    valid_ds = build_concat_dataset(valid_files, means, stds, label_threshold)
    test_ds = build_concat_dataset(test_files, means, stds, label_threshold)
    
    if train_ds is None or valid_ds is None or test_ds is None:
        raise ValueError("抽出可能なシーケンスが存在しません（ファイル内のデータ行数がseq_len未満です）。")

    # DataLoaderの作成
    # Trainはシャッフルして学習効率を高め、Valid/Testは時系列順のまま評価する
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 特徴量の次元数を取得（モデル初期化用）
    # ConcatDatasetの最初の要素(SessionSequenceDataset)から取得
    num_features = train_ds.datasets[0].X_data.shape[1]

    return train_loader, valid_loader, test_loader, num_features
