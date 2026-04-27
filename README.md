# Transformer Futures Trading System

本プロジェクトは、日経225先物を対象とした、Transformerモデルによるトレンド予測フィルターと価格モメンタムを組み合わせた自動売買システムの研究・開発プラットフォームです。

## 1. 外部仕様 (External Specification)

### 1.1 システムの全体像

本システムは、以下の3つの主要フェーズで構成されます。

1. **Feature Generation:** 30秒足のバーデータから学習用特徴量と目的変数（効率比）を生成。
2. **Walk-Forward Training:** 過去の一定期間で学習し、次の期間で検証・テストするプロセスを繰り返す堅牢な学習。
3. **Hybrid Backtesting:** AIの「トレンド発生予測」をフィルターとし、生の価格の「モメンタム」をトリガーとする戦略シミュレーション。

---

### 1.2 コマンドライン・インターフェース

#### 1.2.1 特徴量生成 (`generate_features.py`)

```bash
python -m train.entry.generate_features \
  [--bar-base-dir data] \
  [--output-base-dir data] \
  [--date-from YYYY-MM-DD] \
  [--date-to YYYY-MM-DD] \
  [--label-horizon 240]
```

| 引数                | デフォルト | 説明                                                       |
| ------------------- | ---------- | ---------------------------------------------------------- |
| `--bar-base-dir`    | `data`     | 30秒足 Parquet を含むベースディレクトリ                    |
| `--output-base-dir` | `data`     | 特徴量 Parquet の出力先                                    |
| `--date-from`       | (全期間)   | 処理開始日                                                 |
| `--date-to`         | (全期間)   | 処理終了日                                                 |
| `--label-horizon`   | `240`      | ラベル生成に使う未来参照本数（30秒足: 60=30分, 240=2時間） |

- **入力:** `data/bars/nk225_30s/<YEAR>/<YYYY-MM-DD>.parquet` および `data/bars/external_30s/<SYMBOL>/<YEAR>/<YYYY-MM-DD>.parquet`
- **出力:** `data/features/entry/<YEAR>/<YYYY-MM-DD>.parquet`
- **外部指標:** USDJPY / US500 / NAS100 / XAUUSD / XTIUSD の5銘柄（内部 prefix: usdjpy / sp500 / nasdaq / xau / xti）

#### 1.2.2 ウォークフォワード学習 (`train.py`)

```bash
python -m train.entry.train \
  --train-days 120 \
  --valid-days 20 \
  --test-days 1 \
  --start 2019-12-24
```

| 引数                     | デフォルト | 説明                                   |
| ------------------------ | ---------- | -------------------------------------- |
| `--train-days`           | `60`       | 学習に使う日数                         |
| `--valid-days`           | `20`       | 検証に使う日数（Early Stopping 判定）  |
| `--test-days`            | `1`        | テスト対象日数（通常 1 日）            |
| `--start`                | (全期間)   | 処理を開始するテスト対象日             |
| `--high-conf-percentile` | `80.0`     | 正例とみなす効率比の上位パーセンタイル |

- **出力:** `data/entry/<YEAR>/<YYYY-MM-DD>-<DAY\|NIGHT>.pth`（モデル）、同名の `.json`（エッジ情報・特徴量数含む）

#### 1.2.3 戦略バックテスト (`backtest.py`)

```bash
python -m train.entry.backtest \
  --start 2020-07-01 \
  --prob-threshold 0.82 \
  --max-prob 0.87 \
  --max-daily-trades 2 \
  --sl-ticks 20 \
  --tp-ticks 50 \
  --be-ticks 20 \
  --be-min-bars 20 \
  --edge 15
```

**エントリーフィルター引数:**

| 引数                 | デフォルト | 説明                                                             |
| -------------------- | ---------- | ---------------------------------------------------------------- |
| `--prob-threshold`   | `0.82`     | エントリー確率の下限閾値                                         |
| `--max-prob`         | `1.0`      | エントリー確率の上限閾値（高確率ゾーンの過適合抑制）             |
| `--max-daily-trades` | `0`        | session_date × session_type あたりの最大エントリー数（0=無制限） |
| `--min-prob-rise`    | `0.0`      | エントリーに必要な確率の最小上昇幅（0以下=無効）                 |
| `--min-momentum-abs` | `0.0`      | エントリーに必要な最小モメンタム幅（0以下=無効）                 |
| `--max-momentum-abs` | `80.0`     | 飛び乗り抑制の最大モメンタム幅（0以下=無効）                     |
| `--edge`             | (無効)     | 指定エッジ(%)以上のモデルのみ実行                                |

**リスク管理引数:**

| 引数                | デフォルト | 説明                                                    |
| ------------------- | ---------- | ------------------------------------------------------- |
| `--sl-ticks`        | `20`       | 損切り幅（1 tick = 5円）                                |
| `--tp-ticks`        | `50`       | 利食い幅（1 tick = 5円）                                |
| `--hold-horizon`    | `240`      | 最大ホールド期間（本数）。ラベル horizon に合わせること |
| `--be-ticks`        | `0`        | ブレークイーブンストップのトリガー幅（0=無効）          |
| `--be-min-bars`     | `0`        | BE 発動を許可する最低保有本数（0=制限なし）             |
| `--weak-exit-bars`  | `0`        | ウィーク決済の評価タイミング（本数, 0=無効）            |
| `--weak-exit-ticks` | `0.0`      | ウィーク決済をトリガーする MFE 上限（tick, 0=無効）     |

**出力:**

- ターミナルに損益統計（PF、勝率、最大 DD、Exit reason 別内訳、平均 MFE）
- `data/entry/equity_curve.png`（資産曲線）

---

## 2. 内部詳細設計 (Internal Design)

### 2.1 特徴量エンジニアリング (`features.py`)

Transformerが相場の微細な変化とマクロな構造を同時に理解できるよう、多様な観点から特徴量を抽出しています。

#### 価格・ボラティリティ特徴量

- **対数リターンと複数時間足のモメンタム:** 1分〜60分の複数時間窓（`BAR_1M`〜`BAR_60M`）における対数リターンと単純リターン。
- **実現ボラティリティ:** 5分、10分、30分のローリング標準偏差。
- **Garman-Klass 近似ボラティリティ:** OHLC をフル活用した高精度なボラティリティ推定。
- **バー内位置 (Close Position in Bar):** 終値が高値・安値の間のどこに位置するか（ヒゲの方向性）。

すべての `shift` / `rolling` 系計算は `.over(["session_date_jst", "session_type"])` でセッション内に閉じており、DAY/NIGHT の境界をまたいだ誤った差分は生じません。

#### 出来高・約定フロー特徴量 (Order Flow)

- **売買比率:** 歩み値（Tick Test）から復元された Volume Pressure。
- **累積ネットボリューム:** 直近数分の `signed_volume` の累積値（買い/売り圧力の蓄積）。
- **Tick 密度・スピード:** 直近の平均 Tick 速度との比率（`tick_speed_ratio`）でアルゴの介入を識別。
- **VWAP 乖離率:** 30秒 VWAP、15分 VWAP、セッション開始からの累積 VWAP（`session_vwap`）からの乖離。

#### クロスアセット特徴量（外部指標との連動性）

USDJPY・S&P500・NASDAQ・XAUUSD・XTIUSD の5銘柄を非同期結合（セッション内 Forward Fill）し、国際市場との連動性を評価します。MT5 シンボル名から内部 prefix へのマッピングは `generate_features.py` の `SYMBOL_TO_PREFIX` で一元管理しており、`features.py` のハードコード prefix と一致しています。

- **米国株リスクオン指標:** S&P500 / NASDAQ / DOW の合成リターン（`us_risk_on_ret`）と指数間ばらつき（`us_index_dispersion`）。
- **相対強弱と遅行 (Lead-Lag):** NK225 と S&P500 / NASDAQ のリターン差、および1〜2本ラグの追随遅れ。
- **ショック・ダイバージェンス:** Gold・原油と株価指数の乖離（`gold_vs_sp500_divergence` 等）。
- **USDJPY:** NK225 との相対強弱・リード/ラグ（`rel_strength_vs_usdjpy_1m` 等）。

#### カレンダー・セッション特徴量

- **時間帯エンコーディング:** `minute_of_day` / `weekday` を sin/cos 波に変換し周期性を学習。
- **セッション固有フラグ:** 寄り付き直後（`is_opening_window`）、昼休み前（`is_lunch_edge`）、深夜（`is_night_late`）。

#### データの定常化とリーク防止

- **1階差分による定常化:** `open`, `high`, `low`, `close`, `vwap` 等の絶対価格はすべてセッション内差分に変換（`make_prices_stationary`）。セッション先頭バーは null となり自動的に除外されます。
- **絶対価格の保護:** 定常化前に生の絶対価格を `raw_open_abs` / `raw_high_abs` / `raw_low_abs` / `raw_close_abs` として保持し、バックテストの SL/TP 判定に使用します。

#### 効率比 (Efficiency Ratio) ラベル

- `label_efficiency_240`（2時間先）: 「価格の移動距離 ÷ 走行距離」で算出（0.0〜1.0）。値が 1 に近いほど直線的なトレンド、0 に近いほど乱高下を意味します。

---

### 2.2 データセットと動的ラベリング (`dataset.py`)

- **Z-Score 正規化:** Train データの平均・標準偏差を算出し、Valid / Test に適用（リーク防止）。
- **動的閾値による2値分類:** Train 期間内の効率比の上位 N%（デフォルト 20%）を正例とすることで、相場環境の変化に追従。
- **レガシーモデル互換:** バックテスト時に `state_dict["input_projection.weight"].shape[1]` からモデルの特徴量次元数を自動判定し、旧コード（`raw_*_abs` を特徴量に含む 227次元）と新コード（223次元）が混在しても正しく推論します。

---

### 2.3 モデルアーキテクチャ (`model.py`)

- **Time Series Transformer**
  - **Positional Encoding:** 時系列の順序情報を保持。
  - **Multi-Head Attention:** 過去 60本（30分）のシーケンス内の複数の相関を同時に注視。
  - **出力:** シグモイド前段のロジットを出力。推論時に `torch.sigmoid()` で確率に変換。

---

### 2.4 戦略ロジックとリスク管理 (`backtest.py`)

#### エントリーロジック

- **AI フィルター:** `prob_threshold` ≤ prob ≤ `max_prob` の確率帯のみ「トレンド窓が開いた」と判定。高確率ゾーンの過適合を上限閾値で抑制。
- **モメンタムトリガー:** 直近 `direction-lookback-bars` 本（デフォルト 20本 = 10分）の価格変化がプラスなら買い、マイナスなら売り。
- **確率上昇フィルター:** `--min-prob-rise` を指定すると、直近 `prob-rise-bars` 本に対して確率が上昇中の場合のみエントリー。
- **モメンタム幅フィルター:** `--min-momentum-abs` で初動が弱いシグナルを除外。`--max-momentum-abs` で出遅れエントリーを抑制（デフォルト 80.0円 = 16 tick）。
- **1日あたり上限:** `--max-daily-trades` で session_date × session_type 単位のエントリー数を制限。
- **クールダウン:** 決済後 60本（30分）はエントリーを禁止。

#### 決済ロジック

| reason    | 説明                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `TP`      | バー高値 / 安値が TP 価格に到達。バー内約定価格で決済。                                                                                    |
| `SL`      | バー高値 / 安値が SL 価格に到達。バー内約定価格で決済。BE 発動後は SL がエントリー価格に引き上げられるため、実損失はコスト分（80円）のみ。 |
| `SESSION` | DAY → NIGHT など、セッション境界到達で翌バー始値決済。タイムスタンプ逆転を防止。                                                           |
| `TIME`    | 最大ホールド期間（`hold-horizon` 本）到達で翌バー始値決済。                                                                                |
| `WEAK`    | 指定本数後も MFE が閾値未満の場合にクローズ価格で軟決済。初動確認が取れないトレードの早期撤退。                                            |

#### ブレークイーブンストップ (BE)

- `--be-ticks` tick 以上の含み益を記録した**翌バー**から SL をエントリー価格に引き上げます。`--be-min-bars` で発動抑止期間を設けることでノイズによる過剰発動を防ぎます。
- MFE 更新と BE 発動は SL/TP チェックの**後**に実行されるため、発動バーと同一バーで即 BE 決済されるバグはありません。

#### ウィーク決済 (Weak Exit)

- `--weak-exit-bars` 本後の時点で MFE が `--weak-exit-ticks` tick 未満の場合、その時点のクローズ価格で `reason=WEAK` 決済します。フル SL になる前に方向確認が取れないトレードを早期撤退させます。

#### バックテスト結果サマリー

```
Total Trades : N
Win Rate     : X.X% (NW / NL)
Profit Factor: X.XXX
Gross Profit : +NNN,NNN JPY
Gross Loss   : -NNN,NNN JPY
Net Profit   : +NNN,NNN JPY
Max Drawdown : NNN,NNN JPY
--------------------------------------------------
  TP     :    N  WR=X%  +NNN,NNN JPY
  SESSION:    N  WR=X%  +NNN,NNN JPY
  SL     :    N  WR=X%  -NNN,NNN JPY
  TIME   :    N  WR=X%  ±NNN,NNN JPY
  WEAK   :    N  WR=X%  ±NNN,NNN JPY
  Avg MFE  : X.X ticks  (MFE>0: N/N件)
```

各トレードのログフォーマット:

```
[TRADE] YYYY-MM-DD HH:MM:SS -> YYYY-MM-DD HH:MM:SS | LONG  | entry=NNN,NNN  exit=NNN,NNN | TP      | +NN,NNN JPY | prob=0.XXX | bars=NNN | mfe=NNtk | be=Y | session=YYYY-MM-DD_DAY
```

---

### 2.5 データパイプライン上の既知の問題と対処

| 問題                                                          | 対処                                                                                                                                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OHLC の始値・終値が不正確** (`build_nk225_bars.py`)         | `group_by().agg()` 内で `price.sort_by(trade_ts).first()` / `.last()` を使用し、Tick 時刻順を保証。                                                                                                     |
| **セッション境界でのタイムスタンプ逆転** (`features.py`)      | `build_entry_feature_frame` のソートキーから `session_type` を除去し `bar_start_jst` のみでソート。`session_type` のアルファベット順（DAY < NIGHT）による並び替えが月曜日に逆転を引き起こす問題を修正。 |
| **Parquet ファイルのソート順未保証** (`generate_features.py`) | `sink_parquet`（ストリーミング）を `collect().write_parquet()` に変更しグローバルソート順を保証。                                                                                                       |
| **差分計算がセッション境界をまたぐ** (`features.py`)          | `_pct_ret` / `_log_ret` / `_linear_slope` / `rolling_*` に `.over(["session_date_jst", "session_type"])` を追加。`make_prices_stationary` も同様にセッション内 shift に変更。                           |
| **シンボル名ミスマッチ** (`generate_features.py`)             | MT5 シンボル名 → 内部 prefix の変換テーブル `SYMBOL_TO_PREFIX` を追加。`features.py` のハードコード prefix と対応を一致させた。                                                                         |

---

## 3. ディレクトリ構造

```text
TransformerFutures/
├── data/
│   ├── bars/
│   │   ├── nk225_30s/            # NK225 30秒足 Parquet
│   │   └── external_30s/         # 外部指標 30秒足 Parquet（シンボル別）
│   ├── features/
│   │   └── entry/                # 生成済み特徴量 Parquet
│   └── entry/                    # 学習済みモデル・JSON・equity_curve.png
├── train/
│   └── entry/
│       ├── features.py           # 特徴量定義（セッション対応 rolling 含む）
│       ├── generate_features.py  # 特徴量生成実行
│       ├── dataset.py            # Dataset / DataLoader / 正規化統計 / レガシー互換
│       ├── model.py              # Time Series Transformer 定義
│       ├── train.py              # 学習・検証ループ（Walk-Forward）
│       └── backtest.py           # 戦略シミュレーター（BE / WEAK / SESSION 決済含む）
├── build_nk225_bars.py           # NK225 30秒足バー生成
├── build_mt5_external_bars.py    # 外部指標 30秒足バー生成
├── import_jpx.py                 # JPX 歩み値インポート
├── import_jpx_raw.py             # JPX raw データインポート
├── import_mt5.py                 # MT5 データインポート
└── README.md
```
