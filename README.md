# Transformer Futures Trading System

本プロジェクトは、日経225先物ミニ（NK225M）を対象とした **30秒足デイトレードシステム** の実装です。
**改良型 Temporal Fusion Transformer (TFT)** アーキテクチャを採用し、JPXの歩み値から生成された市場微細構造（Microstructure）指標と、外部環境（USDJPY, S&P500）の先行遅行効果、および時間帯特性を統合しています。
スキャルピングではなく、**1分足〜数時間ホールドを許容するデイトレード時間枠** でポジションをホールドし、往復コスト（15円）を確実に上回るボラティリティの拡大局面（Edge）を抽出・捕捉することを目指します。

> **現在の実験ステータス（2026年3月時点）**
> Walk-Forward 118フォールドを完走。OOSトレードはまだゼロ。Neutral比率・ValLossは目標範囲内に収束。Direction Head の単調低下（dir_acc Ep0→最終で約1.4%pt低下）が残存する課題。

## 1. 外部仕様 (External Specification)

### 1.1 システムの全体像
本システムは、以下の3つの主要フェーズで構成されます。
1.  **Feature Generation:** 30秒足のバーデータから学習用特徴量と目的変数（効率比）を生成。
2.  **Walk-Forward Training:** 過去の一定期間で学習し、次の期間で検証・テストするプロセスを繰り返す堅牢な学習。
3.  **Hybrid Backtesting:** AIの「トレンド発生予測」をフィルターとし、生の価格の「モメンタム」をトリガーとする戦略シミュレーション。

### 1.2 コマンドライン・インターフェース

#### 1.2.1 特徴量生成 (`generate_features.py`)
```bash
python -m train.entry.generate_features
