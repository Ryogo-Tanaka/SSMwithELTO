# Deep Fictitious Instrumental Variables (DFIV) - コード実装ガイド

Deep Fictitious Instrumental Variables (DFIV) を用いた状態空間モデル学習フレームワーク

---

## 📁 プロジェクト構造

```
SSMwithELTO/
├── configs/                          # 設定ファイル
│   ├── quad_image_reconstruction_config.yaml     # 画像再構成用設定
│   └── quad_target_prediction_config.yaml        # ターゲット予測用設定
│
├── src/                              # ソースコード
│   ├── models/                       # モデルアーキテクチャ
│   │   ├── encoder.py               # エンコーダFactory
│   │   ├── decoder.py               # デコーダFactory
│   │   ├── inference_model.py       # 推論モデル
│   │   └── architectures/
│   │       └── rkn.py               # RKN CNN Encoder/Decoder
│   │
│   ├── ssm/                         # 状態空間モデルコア
│   │   ├── realization.py           # 確率的実現（状態推定）
│   │   ├── cross_fitting.py         # クロスフィッティング管理
│   │   ├── df_state_layer.py        # DF-A（状態層）
│   │   ├── df_observation_layer.py  # DF-B（観測層）
│   │   └── observation.py           # CME観測モデル
│   │
│   ├── training/                    # 学習フレームワーク
│   │   └── two_stage_trainer.py     # 2段階学習トレーナー
│   │
│   ├── inference/                   # 推論・Kalman（オプション）
│   │   ├── kalman_filter.py         # Kalmanフィルタ
│   │   ├── noise_covariance.py      # ノイズ共分散推定
│   │   └── state_estimator.py       # 状態推定器
│   │
│   ├── evaluation/                  # 評価・分析
│   │   ├── metrics.py               # 評価指標計算
│   │   └── mode_decomposition.py    # モード分解分析
│   │
│   └── utils/                       # ユーティリティ
│       ├── data_loader.py           # データ読み込み
│       └── gpu_utils.py             # GPU選択
│
└── data/                            # データ格納ディレクトリ
    └── your_dataset/
        └── your_data.npz            # .npz形式で配置
```

---

## 📊 データ配置・形式

### データ配置
```
data/
└── your_dataset/         # 任意のデータセット名
    └── your_data.npz     # NumPy圧縮形式
```

### データ形式（.npzファイル内容）

#### **再構成モード**
```python
# 必須キー: 'y'
np.savez(
    'data/your_dataset/your_data.npz',
    y=observation_data  # shape: (N, D)
)

# N: サンプル数
# D: 観測次元数（画像ならflatten後の次元）
```

**画像データの例**:
```python
# 元データ: (1500, 48, 48, 1) - 1500枚の48x48グレースケール画像
# 保存形式: (1500, 2304) - flattenして保存
# または (1500, 48, 48, 1) - 4次元のまま保存（自動でflatten）

import numpy as np
images = np.random.rand(1500, 48, 48, 1)  # 例: ランダム画像
np.savez('data/your_dataset/images.npz', y=images)
```

**時系列データの例**:
```python
# 元データ: (1000, 10) - 1000時点の10次元時系列
timeseries = np.random.rand(1000, 10)
np.savez('data/your_dataset/timeseries.npz', y=timeseries)
```

#### **ターゲット予測モード**
```python
# 必須キー: 'y', 'target'
np.savez(
    'data/your_dataset/your_data.npz',
    y=observation_data,  # shape: (N, D)
    target=target_data   # shape: (N, T)
)

# T: ターゲット次元数
```

**例: 画像からの状態予測**:
```python
images = np.random.rand(1500, 48, 48, 1)     # 観測画像
states = np.random.rand(1500, 8)             # 制御状態（8次元）
np.savez('data/your_dataset/control.npz', y=images, target=states)
```

### データ読み込み自動処理
```python
# src/utils/data_loader.pyが自動実行
data_dict = load_experimental_data_with_architecture(
    data_path="data/your_dataset/your_data.npz",
    config=config
)

# 出力形式:
# {
#   'train': torch.Tensor (N_train, D),
#   'val': torch.Tensor (N_val, D),
#   'test': torch.Tensor (N_test, D),
#   'metadata': DataMetadata,
#   'targets': {  # ターゲット予測モード時のみ
#       'train': torch.Tensor (N_train, T),
#       'val': torch.Tensor (N_val, T),
#       'test': torch.Tensor (N_test, T)
#   }
# }
```

**自動処理内容**:
1. `.npz`読み込み
2. 4次元画像データは自動flatten: (N, H, W, C) → (N, H*W*C)
3. train/val/test分割（設定ファイルの比率に従う）
4. PyTorch Tensor変換

---

## 🔄 学習フロー

### 全体フロー
```
データ読み込み → トレーナー初期化 → 統合学習 → 評価
```

### Phase-1: DF層学習（各エポック）

#### **ステップ1: データ準備**
```
_prepare_data(Y_train)
  ↓
1. エンコード
   encoder: (N, D_obs) → (N, d_m)
   例: (1050, 2304) → (1050, 50)

2. 状態推定
   realization.estimate_states(): (N, d_m) → (N-h, r)
   例: (1050, 50) → (1030, 30)
   ※ h=過去窓長, r=状態次元
```

**使用モジュール**:
- `src/models/encoder.py`: エンコーダFactory
- `src/models/architectures/rkn.py`: RKN CNN実装
- `src/ssm/realization.py`: 確率的実現

**確率的実現の詳細**:
```python
# src/ssm/realization.py::StochasticRealizationWithEncoder
fit(Y, encoder)
  ↓ M = encoder(Y): (N, D_obs) → (N, d_m)
  ↓ ラグ共分散推定: Cov(M_t, M_{t-τ})
  ↓ 特徴写像: averaging/linear/mlp
  ↓ SVD分解: A, C行列取得

estimate_states(Y)
  ↓ M = encoder(Y)
  ↓ 過去窓でHankel行列構成: (N, h*d_m)
  ↓ X = hankel @ realization_matrix
  ↓ 出力: (N-h, r)
```

#### **ステップ2: DF-A学習（状態層）**
```
_train_df_a_epoch(X_states)
  ↓
for iter in T1_iterations:
    # Stage-1: V_A推定 + φ_θ更新
    Φ = φ_θ(X): (N, r) → (N, d_A)
    V_A^{(-k)} = Ridge(Φ_{-k}^+, Φ_{-k}^-)  # K個推定
    loss = ||Φ^+ - V_A^{(-k)} Φ^-||² + λ_A ||V_A||²
    φ_θ勾配更新

for iter in T2_iterations:
    # Stage-2: U_A推定 + φ_θ更新
    同様の処理
```

**使用モジュール**:
- `src/ssm/df_state_layer.py`: DF-A実装
- `src/ssm/cross_fitting.py`: クロスフィッティング

**クロスフィッティングの詳細**:
```python
# src/ssm/cross_fitting.py
CrossFittingManager.create_blocks(data, n_blocks=5)
  ↓ 時系列を5個の連続ブロックに分割

TwoStageCrossFitter.cross_fit_stage1(phi_minus, phi_plus)
  ↓ for k in range(K):
      訓練: blocks[≠k] でV_k推定
      予測: block[k] でH_cf_k = V_k @ Φ_k^-
  ↓ 出力: V_A_list=[V_1,...,V_K], H_cf
```

#### **ステップ3: DF-B学習（観測層）**
```
_train_df_b_epoch(X_states, M_features)
  ↓
for iter in T1_iterations:
    # Stage-1: V_B推定 + φ_θ更新
    Ψ = ψ_ω(M): (N, d_m) → (N, d_B)
    V_B^{(-k)} = Ridge(Ψ_{-k}^+, Ψ_{-k}^-)
    loss = ||Ψ^+ - V_B^{(-k)} Ψ^-||² + λ_B ||V_B||²
    φ_θ勾配更新

for iter in T2_iterations:
    # Stage-2: U_B推定 + ψ_ω更新
    同様の処理（ψ_ω更新）
```

**使用モジュール**:
- `src/ssm/df_observation_layer.py`: DF-B実装

---

### Phase-2: End-to-end学習（warmup後）

```
_train_integrated_phase2_epoch(Y_train)
  ↓
1. 推論パス
   M = encoder(Y): (N, D_obs) → (N, d_m)
   X_hat = realization.estimate_states(M): (N, d_m) → (N-h, r)
   X_hat_pred = U_A^T V_A φ_θ(X_hat): (N-h, r) → (N-h-1, r)
   M_hat = U_B^T V_B ψ_ω(X_hat_pred): (N-h-1, r) → (N-h-1, d_m)
   Y_hat = decoder(M_hat): (N-h-1, d_m) → (N-h-1, D_obs)

2. 損失計算
   L_rec = ||Y[h:] - Y_hat||²
   L_cca = -Σ_i ρ_i  # 正準相関係数
   L_total = L_rec + λ_cca * L_cca

3. 勾配更新
   encoder, decoderのみ更新
   DF層（V_A, U_A, V_B, U_B）は固定
```

**使用モジュール**:
- `src/models/decoder.py`: デコーダFactory
- `src/loss.py`: CCA損失

---

## ⚙️ 設定ファイル

### 基本構造
```yaml
experiment:
  mode: "reconstruction"  # または "target_prediction"

data:
  train_ratio: 0.7        # 学習:検証:テスト = 7:2:1
  val_ratio: 0.2
  test_ratio: 0.1
  image_shape: [48, 48, 1]  # 画像の場合のみ

model:
  encoder:
    type: "rkn"           # rknEncoder使用
    feature_dim: 50       # d_m: エンコーダ出力次元
    conv_channels: [32, 64]

  decoder:
    type: "rkn"           # rknDecoder使用
    feature_dim: 50       # d_m: デコーダ入力次元
    conv_channels: [64, 32, 1]

training:
  epochs: 10
  T1_iterations: 5        # DF Stage-1反復数
  T2_iterations: 5        # DF Stage-2反復数
  phase1_warmup_epochs: 5 # Phase-2開始エポック

  lr_phi: 1e-3            # φ_θ学習率
  lr_psi: 1e-3            # ψ_ω学習率
  lr_encoder: 1e-3        # encoder学習率
  lr_decoder: 1e-3        # decoder学習率

ssm:
  realization:
    past_horizon: 20      # h: 過去窓長
    rank: 30              # r: 状態次元数
    encoder_output_dim: 50  # d_m（model.encoder.feature_dimと一致）
    feature_mapping:
      type: "mlp"         # averaging/linear/mlp
      hidden_dims: [32]

  df_state:
    feature_dim: 50       # d_A: 状態特徴次元
    lambda_A: 1e-3        # V_A正則化
    lambda_B: 1e-3        # U_A正則化
    cross_fitting:
      n_blocks: 5         # K: ブロック数

  df_observation:
    obs_feature_dim: 25         # d_B: 観測特徴次元
    multivariate_feature_dim: 50  # d_m（encoder出力次元）
    lambda_B: 1e-3        # V_B正則化
    lambda_dB: 1e-3       # U_B正則化
```

### ターゲット予測モード
```yaml
experiment:
  mode: "target_prediction"

model:
  target_decoder:
    type: "rkn"
    feature_dim: 50       # d_m
    state_dim: 8          # T: ターゲット次元
    activation: "relu"
    output_activation: "tanh"

training:
  experiment_mode: "target_prediction"
  target_loss_weight: 1.0
  reconstruction_loss_weight: 0.0
```

---

## 🚀 基本的な使用方法

### コード例
```python
import torch
import numpy as np
from src.utils.data_loader import load_experimental_data_with_architecture
from src.training.two_stage_trainer import TwoStageTrainer
from src.utils.config import load_cfg

# 1. データ準備（事前に.npzファイル作成）
# data/your_dataset/your_data.npz を配置

# 2. 設定読み込み
config = load_cfg("configs/quad_image_reconstruction_config.yaml")

# 3. データ読み込み
data_dict = load_experimental_data_with_architecture(
    "data/your_dataset/your_data.npz",
    config
)

# 4. トレーナー初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = TwoStageTrainer(
    config=config,
    device=device,
    output_dir="results/your_experiment"
)

# 5. 学習実行
trainer.train_integrated(
    data_dict['train'],
    data_dict['val']
)

# 6. 評価
from src.evaluation.metrics import compute_all_metrics
metrics = trainer.evaluate(data_dict['test'])
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"PSNR: {metrics['psnr']:.4f}")
```

---

## 📈 出力例

### 学習ログ（コンソール出力）
```
=====
Epoch 1/10
=====
Phase-1学習
  DF-A Stage-1: loss=0.0234
  DF-A Stage-2: loss=0.0198
  DF-B Stage-1: loss=0.0156
  DF-B Stage-2: loss=0.0142
Phase-2学習
  Rec loss=0.0876, CCA loss=0.0023, Total=0.0899
Validation RMSE: 0.1234

=====
Epoch 6/10
=====
Phase-1学習
  DF-A Stage-1: loss=0.0089
  DF-A Stage-2: loss=0.0076
  DF-B Stage-1: loss=0.0062
  DF-B Stage-2: loss=0.0055
Phase-2学習
  Rec loss=0.0245, CCA loss=0.0008, Total=0.0253
Validation RMSE: 0.0567
```

### 保存ファイル構造
```
results/your_experiment/
├── final_model.pth              # 学習済みモデル
├── training_history.json        # 学習履歴
├── evaluation_results.json      # 評価結果
└── plots/
    ├── training_curves.png      # 学習曲線
    ├── reconstruction_samples.png  # 再構成サンプル
    └── spectrum_analysis.png    # スペクトル分析
```

### モデルファイル内容
```python
# final_model.pth の内容
checkpoint = torch.load('results/your_experiment/final_model.pth')

checkpoint.keys():
# ['config', 'model_state_dict', 'training_state']

checkpoint['model_state_dict'].keys():
# ['encoder', 'decoder', 'df_state', 'df_obs']

checkpoint['config']:
# {
#   'ssm': {'realization': {...}, 'df_state': {...}, 'df_observation': {...}},
#   'model': {'encoder': {...}, 'decoder': {...}}
# }
```

### 評価結果（JSON）
```json
{
  "reconstruction_metrics": {
    "rmse": 0.0567,
    "psnr": 28.34,
    "temporal_correlation": 0.956
  },
  "mode_decomposition": {
    "spectral_radius": 0.892,
    "n_stable_modes": 28,
    "n_dominant_modes": 5
  }
}
```

---

## 🔧 オプション機能

### Kalmanフィルタリング
```python
trainer = TwoStageTrainer(
    config=config,
    device=device,
    use_kalman_filtering=True  # 不確実性定量化
)
```

**使用モジュール**:
- `src/inference/kalman_filter.py`
- `src/inference/noise_covariance.py`
- `src/inference/state_estimator.py`

**出力**: 状態推定値の平均・共分散

---

## 📝 主要な数式

### DF-A Stage-1損失
```
L_Stage1 = Σ_k ||Φ_k^+ - V_A^{(-k)} Φ_k^-||² + λ_A ||V_A^{(-k)}||²
```

### DF-B Stage-1損失
```
L_Stage1 = Σ_k ||Ψ_k^+ - V_B^{(-k)} Ψ_k^-||² + λ_B ||V_B^{(-k)}||²
```

### Phase-2総合損失
```
L_total = L_reconstruction + λ_cca * L_cca
```

### 推論パス
```
Y → encoder → M → realization → X̂
X̂ → φ_θ → Φ → (U_A^T V_A) → Φ_pred → X̂_pred
X̂_pred → ψ_ω → Ψ → (U_B^T V_B) → Ψ_pred → M̂
M̂ → decoder → Ŷ
```

---

## 🎓 実装の特徴

1. **クロスフィッティング**: 時系列連続ブロック分割でout-of-fold推定
2. **勾配伝播**: Ridge解（V_A, U_A等）を通じた勾配伝播実装
3. **統合学習**: Phase-1/Phase-2を各エポックで連続実行
4. **多変量対応**: DF-B多変量観測対応
5. **モジュール設計**: Factoryパターンで拡張性確保

---

## 📚 参考資料

- 詳細定式化: `_prompt/Theory_and_Formulation.md`
- 実装ガイド: `_prompt/Code_Implementation_Guide.md`
- コードフロー: `_prompt/CODE_FLOW_ANALYSIS.md`
