# train/entry/model.py
"""
ソースコードの役割:
時系列特徴量を用いた回帰タスク用のTransformerモデルです。
連続値特徴量を線形投影し、位置情報を付与した上でTransformerEncoderで処理します。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Transformerに時系列の順序情報を付与するための位置エンコーディングです。
    標準的なサイン・コサイン波を用いています。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 位置エンコーディング行列 (max_len, d_model) を作成
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 奇数・偶数次元でsin/cosを適用
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # バッチ次元用の拡張 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 学習パラメータではないので、register_bufferで保存（デバイス同期のため）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力テンソル (Batch, SeqLen, d_model)
        Returns:
            torch.Tensor: 位置情報が付加されたテンソル
        """
        # シーケンス長に合わせてスライスして加算
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TimeSeriesTransformer(nn.Module):
    """
    システムトレードのエントリー可否判定（効率比予測）を行うTransformerモデルです。
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling_type: str = "last"
    ):
        """
        Args:
            num_features (int): 入力特徴量の次元数（Xの列数）。
            d_model (int, optional): Transformerの隠れ層の次元数。デフォルトは 128。
            nhead (int, optional): Multi-Head Attentionのヘッド数。デフォルトは 8。
            num_layers (int, optional): Encoderのレイヤー数。デフォルトは 3。
            dim_feedforward (int, optional): FFNの中間次元数。デフォルトは 512。
            dropout (float, optional): ドロップアウト率。デフォルトは 0.1。
            pooling_type (str, optional): "last" (最後のステップ) または "mean" (Global Average Pooling)。
        """
        super().__init__()
        self.pooling_type = pooling_type
        
        # 1. 連続値入力の線形投影層（Continuous Input Projection）
        # (Batch, SeqLen, num_features) -> (Batch, SeqLen, d_model)
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (Batch, SeqLen, Features) を前提とする
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 4. 出力層（回帰: 1次元出力）
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 特徴量テンソル (Batch, SeqLen, num_features)
            
        Returns:
            torch.Tensor: 予測ラベル (Batch, 1)
        """
        # 線形投影と位置エンコーディング
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformerによる系列モデリング
        x = self.transformer_encoder(x)
        
        # プーリング処理
        if self.pooling_type == "last":
            # 最後のステップの表現のみ抽出 (Batch, d_model)
            # -> 現在時刻の予測として最も直感的
            x = x[:, -1, :]
        elif self.pooling_type == "mean":
            # 系列全体の特徴を平均化 (Batch, d_model)
            x = x.mean(dim=1)
        else:
            raise ValueError(f"サポートされていないpooling_type: {self.pooling_type}")
            
        # 回帰出力
        out = self.output_layer(x)
        
        return out
