# 提案手法の2段階学習実装 - 使用ガイド

## 概要

本実装は提案手法の2段階学習戦略を完全に実装したものです：

- **Phase-1**: DF-A/DF-B の Stage-1/Stage-2 交互学習
- **Phase-2**: End-to-end 微調整

## ファイル構成

```
src/
├── training/
│   └── two_stage_trainer.py        # メイン学習クラス
├── ssm/
│   ├── df_state_layer.py           # 修正版DF-A
│   ├── df_observation_layer.py     # 修正版DF-B  
│   ├── realization.py              # 確率的実現
│   └── cross_fitting.py            # クロスフィッティング
├── models/
│   ├── encoder.py                  # 修正版エンコーダファクトリ
│   ├── decoder.py                  # 修正版デコーダファクトリ
│   └── architectures/
│       └── tcn.py                  # 修正版TCN実装
└── utils/
    └── gpu_utils.py                # GPU選択ユーティリティ

main_two_stage.py                   # メイン実行スクリプト
configs/
└── config_two_stage_experiment.yaml # サンプル設定ファイル
```

## 使用方法

### 1. 基本的な実験実行

```bash
python main_two_stage.py \
    --config configs/config_two_stage_experiment.yaml \
    --data data/sim_complex.npz \
    --output results/experiment_001 \
    --device cuda
```

### 2. 設定ファイルのカスタマイズ

`configs/config_two_stage_experiment.yaml`を編集して実験設定を調整：

```yaml
# データに合わせて入力・出力次元を調整
model:
  encoder:
    input_dim: 7  # 観測次元
  decoder:
    output_dim: 7 # 観測次元

# 学習ハイパーパラメータの調整
training:
  phase1_epochs: 30     # Phase-1エポック数
  T1_iterations: 8      # Stage-1反復数
  T2_iterations: 3      # Stage-2反復数
  lr_phi: 1e-3         # φ_θ学習率
  lr_psi: 1e-3         # ψ_ω学習率
```

### 3. データ形式

データファイル（.npz）には以下のキーが必要：
- `'Y'` または `'arr_0'`: 観測時系列 (T, d) の NumPy配列

シミュレーションデータの生成：
```bash
python scripts/generate_sim_data.py  # sim_complex.npz などを生成
```

### 4. 結果の確認

実験結果は指定した出力ディレクトリに保存されます：

```
results/experiment_001/
├── final_model.pth              # 最終学習済みモデル
├── phase1_training.csv          # Phase-1詳細ログ
├── phase2_training.csv          # Phase-2詳細ログ
├── training_summary.json        # 学習サマリ
├── phase1_losses.png           # Phase-1損失推移
├── phase2_losses.png           # Phase-2損失推移
├── prediction_results.png       # 予測結果
└── config_used.yaml            # 使用した設定ファイル
```

## 学習戦略の詳細

### Phase-1: DF学習

```python
# DF-A学習
for epoch in range(phase1_epochs):
    # Stage-1: V_A推定 + φ_θ勾配更新
    for t in range(T1_iterations):
        V_A = 閉形式解(Φ_minus, Φ_plus)
        φ_θ.backward()  # φ_θ更新
    
    # Stage-2: U_A推定（閉形式解のみ）
    for t in range(T2_iterations):
        U_A = 閉形式解(H_cf, X_plus)  # 勾配なし

# DF-B学習（DF-Aのウォームアップ後）
for epoch in range(df_a_warmup_epochs, phase1_epochs):
    # Stage-1: V_B推定 + φ_θ更新（ψ_ω固定）
    for t in range(T1_iterations):
        V_B = 閉形式解(Φ_prev, Ψ_curr)
        φ_θ.backward()  # ψ_ω固定
    
    # Stage-2: u_B推定 + ψ_ω更新（φ_θ固定）
    for t in range(T2_iterations):
        u_B = 閉形式解(H_cf, m)
        ψ_ω.backward()  # φ_θ固定
```

### Phase-2: End-to-end微調整

```python
# 固定推論パス
x̂_t = U_A^T V_A φ_θ(x_{t-1})
m̂_t = u_B^T V_B φ_θ(x̂_t)
ŷ_t = g_α(m̂_t)

# 損失・逆伝播
L_total = L_rec + λ_c L_cca
(u_η, g_α, φ_θ, ψ_ω).backward()
```

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```yaml
   # バッチサイズや特徴次元を削減
   training:
     T1_iterations: 5  # デフォルト: 8
     T2_iterations: 2  # デフォルト: 3
   
   ssm:
     df_state:
       feature_dim: 16  # デフォルト: 32
   ```

2. **数値不安定**
   ```yaml
   # 正則化パラメータを増加
   ssm:
     realization:
       jitter: 1e-2     # デフォルト: 1e-3
     df_state:
       lambda_A: 1e-2   # デフォルト: 1e-3
   ```

3. **学習が収束しない**
   ```yaml
   # 学習率を調整
   training:
     lr_phi: 5e-4       # デフォルト: 1e-3
     lr_psi: 5e-4       # デフォルト: 1e-3
   ```

### ログの確認

- Phase-1の進捗: `phase1_training.csv`
- Phase-2の進捗: `phase2_training.csv`
- エラーログ: `error_log.json`（エラー時）

### デバッグモード

```yaml
debug:
  save_intermediate: true    # 中間結果保存
  profile_training: true     # 学習プロファイリング
  continue_on_error: true    # エラー時継続
```

## 高度な使用法

### カスタム損失関数

`TwoStageTrainer`クラスを継承して独自の損失関数を実装：

```python
class CustomTwoStageTrainer(TwoStageTrainer):
    def _compute_custom_loss(self, Y_pred, Y_target):
        # カスタム損失の実装
        pass
```

### 分散学習

複数GPUでの学習（実装が必要）：

```python
trainer = TwoStageTrainer(...)
trainer = nn.DataParallel(trainer)
```

## 参考

- 元論文の定式化：document_content の Section 1.4, 2.6
- 学習戦略の詳細：上記のコメント部分を参照
- 実装の詳細：各クラスのdocstring

## サポート

問題がある場合は、以下を確認してください：
1. 設定ファイルの形式
2. データファイルの形式と次元
3. CUDA/PyTorchのバージョン互換性
4. メモリ使用量
