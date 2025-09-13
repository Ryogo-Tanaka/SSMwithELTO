# フィルタリング状態推定・評価

## 📋 概要

学習済みDFIV Kalman Filterモデルを使用して状態推定を行い、その性能を包括的に評価します。

### 🎯 主要機能

- **フィルタリング性能評価**: バッチ・オンライン両方式での状態推定性能測定
- **推定手法比較**: Kalman実現 vs 決定的実現の詳細比較
- **不確実性定量化評価**: 信頼区間品質・キャリブレーション分析
- **結果の完全出力**: CSV・JSON・NPZ形式での数値保存
- **ターミナル詳細出力**: 実行中のリアルタイム性能表示

## 🚀 クイックスタート

### 基本実行（推奨）

```bash
# 基本的なフィルタリング性能評価
python scripts/run_filtering_evaluation.py \
    --model results/trained_model.pth \
    --data data/test.npz \
    --output results/filtering_evaluation
```

### 包括的評価

```bash
# 詳細な分析を含む包括的評価
python scripts/run_filtering_evaluation.py \
    --model results/trained_model.pth \
    --data data/test.npz \
    --output results/comprehensive_eval \
    --config configs/evaluation_config.yaml \
    --mode comprehensive
```

### クイックテスト

```bash
# 開発・デバッグ用の高速評価
python scripts/run_filtering_evaluation.py \
    --model results/trained_model.pth \
    --data data/test.npz \
    --output results/quick_test \
    --mode quick
```

## 📁 ファイル構成

```
├── src/evaluation/
│   ├── metrics.py                    # 評価指標計算
│   ├── filtering_analysis.py         # フィルタリング分析
│   └── uncertainty_evaluation.py     # 不確実性評価
├── scripts/
│   ├── run_filtering_evaluation.py      # 統合実行スクリプト
│   ├── evaluate_filtering_performance.py  # フィルタリング評価
│   └── compare_estimation_methods.py      # 手法比較
├── configs/
│   └── evaluation_config.yaml        # 評価設定
└── README_Task4.md                   # このファイル
```

## ⚙️ 設定ファイル

### evaluation_config.yaml の主要設定

```yaml
evaluation:
  experiment_name: "filtering_performance_evaluation"
  save_detailed_results: true
  create_visualizations: true
  
  data:
    test_split: "test"              # 評価データ分割
    calibration_ratio: 0.25         # キャリブレーション用データ比率
    
  metrics:
    accuracy: ["mse", "mae", "rmse", "correlation"]
    uncertainty: ["coverage_95", "interval_width", "calibration_error"]
    efficiency: ["processing_time", "memory_usage"]

uncertainty_analysis:
  enabled: true
  confidence_intervals:
    levels: [0.68, 0.95, 0.99]      # 信頼水準
  calibration:
    n_bins: 10                      # キャリブレーション評価ビン数

visualization:
  enabled: true
  plots:
    - "uncertainty_distribution"
    - "temporal_uncertainty"
    - "error_vs_uncertainty"
    - "confidence_intervals"
    - "calibration_curve"
```

## 🔧 個別実行

各評価を個別に実行することも可能です：

### 1. フィルタリング性能評価のみ

```bash
python scripts/evaluate_filtering_performance.py \
    --model_path results/trained_model.pth \
    --data_path data/test.npz \
    --output_dir results/filtering_only \
    --config configs/evaluation_config.yaml
```

### 2. 推定手法比較のみ

```bash
python scripts/compare_estimation_methods.py \
    --model_path results/trained_model.pth \
    --data_path data/test.npz \
    --output_dir results/comparison_only \
    --config configs/evaluation_config.yaml
```

## 📊 出力結果の構成

統合実行後の出力ディレクトリ構成：

```
results/filtering_evaluation/
├── filtering_performance/           # フィルタリング性能詳細
│   ├── *_analysis.json             # 詳細分析結果
│   ├── *_metrics.csv               # 性能指標CSV
│   └── *_data.npz                  # 数値データ
├── method_comparison/               # 手法比較詳細
│   ├── *_comparison.json           # 比較分析結果
│   └── *_comparison.csv            # 比較サマリCSV
├── uncertainty_analysis/            # 不確実性分析詳細
│   └── uncertainty_analysis.png    # 不確実性可視化
├── summary/                         # 統合サマリ
│   ├── filtering_evaluation_comprehensive_*.json  # 統合結果
│   └── filtering_evaluation_summary_*.csv         # 統合サマリCSV
└── visualizations/                  # 可視化結果
    └── *.png                       # 各種プロット
```

## 📈 出力される評価指標

### 精度指標
- **MSE**: 平均二乗誤差
- **MAE**: 平均絶対誤差
- **RMSE**: 二乗平均平方根誤差
- **相関係数**: 推定値と真値の相関

### 不確実性指標
- **カバレッジ率**: 68%, 95%, 99%信頼区間のカバレッジ
- **区間幅**: 信頼区間の平均幅
- **キャリブレーション誤差**: 不確実性の校正度

### 計算効率指標
- **処理時間**: バッチ・オンライン処理時間
- **メモリ使用量**: 推論時のメモリ消費
- **スループット**: 単位時間あたりの処理数

### 手法比較指標
- **改善率**: Kalman vs 決定的手法の性能向上率
- **速度比**: 処理速度の比較
- **不確実性付与**: 不確実性定量化の有無

## 💾 結果の活用

### CSV形式での数値保存

```python
import pandas as pd

# サマリ結果の読み込み
df = pd.read_csv('results/filtering_evaluation/summary/filtering_evaluation_summary_20250913_143022.csv')
print(df[['method', 'mse', 'mae', 'coverage_95']])
```

### JSON形式での詳細分析

```python
import json

# 詳細結果の読み込み
with open('results/filtering_evaluation/summary/filtering_evaluation_comprehensive_20250913_143022.json', 'r') as f:
    results = json.load(f)
    
# 主要発見事項の確認
findings = results['evaluation_results']['integrated_analysis']['key_findings']
for finding in findings:
    print(f"- {finding}")
```

### NPZ形式での数値データ

```python
import numpy as np

# 状態推定結果の読み込み
data = np.load('results/filtering_evaluation/filtering_performance/*_data.npz')
batch_states = data['batch_states']      # バッチフィルタリング結果
online_states = data['online_states']    # オンラインフィルタリング結果
print(f"推定状態形状: {batch_states.shape}")
```

## 🔍 トラブルシューティング

### よくあるエラーと対処法

#### 1. モデル読み込みエラー

```
❌ モデル読み込みエラー: ...
```

**対処法:**
- モデルファイルのパスを確認
- 推論設定ファイル（configs/inference_config.yaml）の存在確認
- モデルと設定ファイルの整合性確認

#### 2. データ形式エラー

```
❌ データ読み込みエラー: ...
```

**対処法:**
- データファイルが (T, d) の2次元形状であることを確認
- .npzファイル内のキー名確認（'Y', 'arr_0', 'X'のいずれか）
- データファイルの破損確認

#### 3. メモリ不足

```
❌ CUDA out of memory
```

**対処法:**
- CPUモードで実行: `--device cpu`
- データサイズ制限: 設定ファイルで `max_evaluation_length: 1000`
- クイックモード使用: `--mode quick`

#### 4. 推論環境セットアップエラー

```
❌ 推論環境セットアップエラー: ...
```

**対処法:**
- 学習済みモデルがKalman対応で学習されているか確認
- キャリブレーションデータのサイズ確認
- 数値安定性パラメータの調整

### デバッグモード

詳細なエラー情報が必要な場合：

```bash
# デバッグ情報付きで実行
python -v scripts/run_filtering_evaluation.py \
    --model results/trained_model.pth \
    --data data/test.npz \
    --output results/debug_eval \
    --mode quick
```

## 📚 実装詳細

### 主要クラス

- **FilteringPerformanceEvaluator**: フィルタリング性能の包括的評価
- **EstimationMethodComparator**: 推定手法の比較分析
- **StateEstimationMetrics**: 状態推定指標の計算
- **UncertaintyEvaluator**: 不確実性品質評価
- **FilteringAnalyzer**: フィルタリング分析の統合

### 評価フロー

1. **データ準備**: 統一データローダーによる前処理
2. **推論環境セットアップ**: キャリブレーションデータによる初期化
3. **バッチフィルタリング**: 全時点一括状態推定
4. **オンラインフィルタリング**: 逐次状態推定
5. **手法比較**: Kalman vs 決定的手法
6. **不確実性分析**: 信頼区間・キャリブレーション評価
7. **結果統合**: サマリ作成・可視化・保存

## 🔬 評価モード詳細

### Quick Mode
- **実行時間**: 1-2分
- **対象データ**: 100サンプル
- **評価指標**: MSE, MAE, 95%カバレッジ
- **可視化**: なし
- **用途**: 開発・デバッグ

### Standard Mode
- **実行時間**: 5-15分
- **対象データ**: 全データ
- **評価指標**: 精度・不確実性・効率の基本指標
- **可視化**: 基本プロット
- **用途**: 通常の性能確認

### Comprehensive Mode
- **実行時間**: 15-60分
- **対象データ**: 全データ
- **評価指標**: 全指標 + 詳細分析
- **可視化**: 全プロット + 詳細分析図
- **用途**: 論文・詳細分析

## 📖 参考資料

- **Algorithm 1**: 作用素ベースKalman更新（論文参照）
- **DFIV手法**: 深層学習状態空間モデル実装
- **評価指標**: 状態推定・不確実性定量化の標準指標
- **統計的検定**: 手法比較の有意性評価

## 🤝 次のステップ

タスク4完了後の推奨アクション：

1. **結果分析**: CSV・JSONファイルの詳細確認
2. **可視化確認**: 生成されたプロットの分析
3. **タスク5準備**: 提案手法詳細実験への移行
4. **外部比較**: 他手法との比較準備（ユーザー実装）

---

**作成日**: 2025年9月13日  
**対応プロジェクト**: DFIV Kalman Filter フィルタリング評価  
**実装状況**: 完了
