#!/usr/bin/env python3
"""
推定手法比較スクリプト

DFIV内の決定的実現とKalman実現を比較し、
Kalman Filteringの効果を定量的に評価する。

Usage:
    python scripts/compare_estimation_methods.py \
        --model_path results/trained_model.pth \
        --data_path data/test_data.npz \
        --output_dir results/method_comparison \
        --config configs/inference_config.yaml
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import csv
from typing import Dict, List, Any

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent.parent))

from src.models.inference_model import InferenceModel
from src.training.two_stage_trainer import TwoStageTrainer
from src.evaluation.metrics import StateEstimationMetrics, print_comparison_summary
from src.utils.data_loader import load_experimental_data


class EstimationMethodComparator:
    """推定手法比較クラス"""
    
    def __init__(
        self, 
        model_path: str, 
        config_path: str, 
        output_dir: str,
        device: str = 'auto'
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # デバイス設定
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🖥️  使用デバイス: {self.device}")
        
        # 評価器初期化
        self.metrics_evaluator = StateEstimationMetrics(device=self.device)
        
        # モデル読み込み・準備
        self.models = {}
        self._prepare_models()
        
    def _prepare_models(self):
        """比較用モデルの準備"""
        print(f"\n📂 比較用モデル準備中...")
        
        try:
            # 1. Kalman推論モデル
            print("  📊 Kalman推論モデル読み込み...")
            self.models['kalman'] = InferenceModel(
                str(self.model_path), str(self.config_path)
            )
            
            # 2. 決定的推論用にトレーナーも読み込み（決定的実現用）
            print("  📈 決定的推論用トレーナー読み込み...")

            # 訓練時の設定ファイルを使用してモデル構造を正しく復元
            self.deterministic_trainer = TwoStageTrainer.from_trained_model(
                str(self.model_path),
                config_path=str(self.config_path),  # 追加: YAML設定を渡す
                device=self.device,
                output_dir=str(self.output_dir / 'temp')
            )
            
            print("✅ モデル準備完了")
            
        except Exception as e:
            print(f"❌ モデル準備エラー: {e}")
            raise
    
    
    def compare_methods(
        self,
        data_path: str,
        experiment_name: str = None,
        data_split: str = 'test',
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        推定手法の包括的比較
        
        Args:
            data_path: 評価データパス
            experiment_name: 実験名
            data_split: データ分割
            save_results: 結果保存するか
            
        Returns:
            Dict: 比較結果
        """
        if experiment_name is None:
            experiment_name = f"method_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        print(f"\n🔍 推定手法比較開始")
        print(f"📊 実験名: {experiment_name}")
        print(f"📁 データ: {data_path}")
        print("="*70)
        
        # 1. データ読み込み
        comparison_data = self._load_comparison_data(data_path, data_split)
        test_data = comparison_data['observations']
        true_states = comparison_data.get('true_states', None)
        
        print(f"📏 評価データ形状: {test_data.shape}")
        
        # 2. 各手法で推定実行
        method_results = {}
        
        # 2.1 Kalman推定
        kalman_results = self._run_kalman_estimation(test_data, true_states)
        method_results['kalman'] = kalman_results
        
        # 2.2 決定的推定
        deterministic_results = self._run_deterministic_estimation(test_data, true_states)
        method_results['deterministic'] = deterministic_results
        
        # 3. 手法比較分析
        comparison_analysis = self._analyze_method_comparison(method_results, true_states)
        
        # 4. 結果統合
        complete_comparison = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'data_path': data_path,
                'data_split': data_split,
                'data_shape': list(test_data.shape),
                'has_true_states': true_states is not None
            },
            'method_results': method_results,
            'comparison_analysis': comparison_analysis,
            'summary': self._create_comparison_summary(method_results, comparison_analysis)
        }
        
        # 5. 結果出力
        self._print_comparison_results(complete_comparison)
        
        # 6. 結果保存
        if save_results:
            self._save_comparison_results(complete_comparison, experiment_name)
        
        return complete_comparison
    
    def _load_comparison_data(self, data_path: str, data_split: str) -> dict:
        """比較用データの読み込み"""
        print(f"\n📂 データ読み込み中...")
        
        try:
            data_dict = load_experimental_data(data_path)
            
            # 指定された分割を取得
            if data_split in data_dict:
                observations = data_dict[data_split]
            else:
                observations = data_dict[list(data_dict.keys())[0]]
                print(f"⚠️  指定分割 '{data_split}' が見つからない。'{list(data_dict.keys())[0]}' を使用。")
            
            observations = observations.to(self.device)
            
            # 真値状態
            true_states = None
            if 'true_states' in data_dict:
                true_states = data_dict['true_states'].to(self.device)
            elif f'{data_split}_states' in data_dict:
                true_states = data_dict[f'{data_split}_states'].to(self.device)
                
            print(f"✅ データ読み込み完了")
            return {
                'observations': observations,
                'true_states': true_states,
                'metadata': {'available_keys': list(data_dict.keys())}
            }
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def _run_kalman_estimation(self, test_data: torch.Tensor, true_states: torch.Tensor) -> dict:
        """Kalman推定の実行"""
        print(f"\n🎲 Kalman推定実行中...")
        
        try:
            # 推論環境セットアップ
            # past_horizonを考慮した安全なキャリブレーションサイズ計算
            past_horizon = 10  # デフォルト値（設定から取得すべきだが一時的に固定）
            min_required = 2 * past_horizon + 1
            total_samples = test_data.size(0)

            print(f"🔍 Stage 2 - キャリブレーション分析:")
            print(f"   観測データ総数: {total_samples}")
            print(f"   past_horizon: {past_horizon}")
            print(f"   必要最小サンプル: {min_required}")

            if total_samples >= min_required:
                calibration_size = min(50, max(min_required, total_samples // 4))
                print(f"✅ 十分なデータ: キャリブレーション{calibration_size}サンプル使用")
            else:
                calibration_size = total_samples
                print(f"❌ データ不足: 全{total_samples}サンプル使用、数値不安定の可能性")

            calibration_data = test_data[:calibration_size]
            
            self.models['kalman'].setup_inference(
                calibration_data=calibration_data,
                method='data_driven'
            )
            
            # フィルタリング実行
            start_time = datetime.now()
            filtering_result = self.models['kalman'].filter_sequence(
                test_data, return_likelihood=True
            )
            
            if len(filtering_result) == 3:
                X_means, X_covariances, likelihoods = filtering_result
            else:
                X_means, X_covariances = filtering_result
                likelihoods = None
                
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 性能評価
            metrics = self.metrics_evaluator.compute_all_metrics(
                X_means, true_states, X_covariances, test_data, likelihoods, verbose=False
            )
            
            print(f"  ✅ Kalman推定完了 ({processing_time:.4f}秒)")
            print(f"  📏 推定状態形状: {X_means.shape}")
            
            return {
                'success': True,
                'method_name': 'Kalman Filtering',
                'estimated_states': X_means,
                'covariances': X_covariances,
                'likelihoods': likelihoods,
                'processing_time': processing_time,
                'metrics': metrics,
                'has_uncertainty': True
            }
            
        except Exception as e:
            import traceback
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'full_traceback': traceback.format_exc(),
                'model_type': type(self.models['kalman']).__name__,
                'available_methods': [m for m in dir(self.models['kalman']) if not m.startswith('_')],
                'filter_methods': [m for m in dir(self.models['kalman']) if not m.startswith('_') and 'filter' in m.lower()],
                'test_data_shape': list(test_data.shape),
                'model_setup_status': getattr(self.models['kalman'], 'is_setup', 'unknown')
            }

            print(f"  ❌ Kalman推定エラー: {e}")
            print(f"  🔍 エラータイプ: {type(e).__name__}")
            print(f"  📊 データ形状: {test_data.shape}")
            print(f"  🎯 モデルタイプ: {type(self.models['kalman']).__name__}")
            print(f"  🔧 利用可能なfilterメソッド: {error_details['filter_methods']}")
            print(f"  ⚙️  モデルセットアップ状況: {error_details['model_setup_status']}")
            print(f"  📝 詳細トレース:\n{traceback.format_exc()}")

            return {
                'success': False,
                'method_name': 'Kalman Filtering',
                'error': str(e),
                'error_details': error_details
            }
    
    def _run_deterministic_estimation(self, test_data: torch.Tensor, true_states: torch.Tensor) -> dict:
        """決定的推定の実行"""
        print(f"\n📈 決定的推定実行中...")
        
        try:
            # 決定的実現を使用した状態推定
            start_time = datetime.now()
            
            # エンコーディング
            with torch.no_grad():
                self.deterministic_trainer.encoder.eval()
                encoded = self.deterministic_trainer.encoder(test_data.unsqueeze(0)).squeeze(0)
                
                # 実現化による状態推定（形状調整版）
                # realization用の2次元形状調整: [T, feature_dim] → [T, d]
                if hasattr(self.deterministic_trainer, 'realization'):
                    # encodedが[T, feature_dim]の場合、適切に2次元に調整
                    if encoded.dim() == 2:
                        if encoded.shape[1] == 1:
                            encoded_2d = encoded  # [T, 1] ← 既に正しい
                        else:
                            # feature_dimが複数の場合、1次元に調整（最初の特徴量を使用）
                            encoded_2d = encoded[:, :1]  # [T, 1]
                    elif encoded.dim() == 1:
                        encoded_2d = encoded.unsqueeze(1)  # [T, 1]
                    else:
                        raise ValueError(f"Unexpected encoded dimension: {encoded.dim()}, shape: {encoded.shape}")

                    self.deterministic_trainer.realization.fit(encoded_2d)
                    X_estimated = self.deterministic_trainer.realization.filter(encoded_2d)
                else:
                    # フォールバック：エンコード結果をそのまま状態として使用
                    X_estimated = encoded
                    
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 性能評価
            metrics = self.metrics_evaluator.compute_all_metrics(
                X_estimated, true_states, None, test_data, None, verbose=False
            )
            
            print(f"  ✅ 決定的推定完了 ({processing_time:.4f}秒)")
            print(f"  📏 推定状態形状: {X_estimated.shape}")
            
            return {
                'success': True,
                'method_name': 'Deterministic Realization',
                'estimated_states': X_estimated,
                'covariances': None,
                'likelihoods': None,
                'processing_time': processing_time,
                'metrics': metrics,
                'has_uncertainty': False
            }
            
        except Exception as e:
            import traceback
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'full_traceback': traceback.format_exc(),
                'trainer_type': type(self.deterministic_trainer).__name__,
                'available_methods': [m for m in dir(self.deterministic_trainer) if not m.startswith('_')],
                'realization_methods': [m for m in dir(self.deterministic_trainer) if not m.startswith('_') and 'realization' in m.lower()],
                'test_data_shape': list(test_data.shape),
                'has_realization': hasattr(self.deterministic_trainer, 'realization')
            }

            print(f"  ❌ 決定的推定エラー: {e}")
            print(f"  🔍 エラータイプ: {type(e).__name__}")
            print(f"  📊 データ形状: {test_data.shape}")
            print(f"  🎯 トレーナータイプ: {type(self.deterministic_trainer).__name__}")
            print(f"  🔧 利用可能なrealizationメソッド: {error_details['realization_methods']}")
            print(f"  ⚙️  realization存在: {error_details['has_realization']}")
            print(f"  📝 詳細トレース:\n{traceback.format_exc()}")

            return {
                'success': False,
                'method_name': 'Deterministic Realization',
                'error': str(e),
                'error_details': error_details
            }
    
    def _analyze_method_comparison(self, method_results: dict, true_states: torch.Tensor) -> dict:
        """手法比較の詳細分析"""
        print(f"\n🔍 手法比較分析中...")
        
        analysis = {
            'methods_compared': list(method_results.keys()),
            'comparison_available': True
        }
        
        # 成功した手法のみを比較
        successful_methods = {k: v for k, v in method_results.items() if v.get('success', False)}
        
        if len(successful_methods) < 2:
            analysis['comparison_available'] = False
            analysis['error'] = 'Not enough successful methods for comparison'
            return analysis
        
        # 精度比較
        if all('accuracy' in method['metrics'] for method in successful_methods.values()):
            analysis['accuracy_comparison'] = self._compare_accuracy_metrics(successful_methods)
        
        # 計算効率比較
        analysis['efficiency_comparison'] = self._compare_efficiency_metrics(successful_methods)
        
        # Kalman特有の分析（不確実性定量化）
        if 'kalman' in successful_methods and successful_methods['kalman']['has_uncertainty']:
            analysis['uncertainty_analysis'] = self._analyze_kalman_uncertainty(
                successful_methods['kalman'], true_states
            )
        
        return analysis
    
    def _compare_accuracy_metrics(self, successful_methods: dict) -> dict:
        """精度指標の比較"""
        accuracy_comparison = {}
        
        # 各指標を比較
        metrics_to_compare = ['mse', 'mae', 'rmse', 'correlation']
        
        for metric in metrics_to_compare:
            metric_values = {}
            for method_name, method_result in successful_methods.items():
                if metric in method_result['metrics']['accuracy']:
                    metric_values[method_name] = method_result['metrics']['accuracy'][metric]
            
            if len(metric_values) >= 2:
                # 最良・最悪値
                best_method = min(metric_values, key=metric_values.get) if metric != 'correlation' else max(metric_values, key=metric_values.get)
                worst_method = max(metric_values, key=metric_values.get) if metric != 'correlation' else min(metric_values, key=metric_values.get)
                
                accuracy_comparison[metric] = {
                    'values': metric_values,
                    'best_method': best_method,
                    'worst_method': worst_method,
                    'improvement': self._calculate_improvement(metric_values, metric)
                }
        
        return accuracy_comparison
    
    def _compare_efficiency_metrics(self, successful_methods: dict) -> dict:
        """計算効率の比較"""
        efficiency_comparison = {}
        
        processing_times = {}
        for method_name, method_result in successful_methods.items():
            processing_times[method_name] = method_result['processing_time']
        
        if len(processing_times) >= 2:
            fastest_method = min(processing_times, key=processing_times.get)
            slowest_method = max(processing_times, key=processing_times.get)
            
            efficiency_comparison['processing_time'] = {
                'values': processing_times,
                'fastest_method': fastest_method,
                'slowest_method': slowest_method,
                'speed_ratio': max(processing_times.values()) / min(processing_times.values())
            }
        
        return efficiency_comparison
    
    def _analyze_kalman_uncertainty(self, kalman_result: dict, true_states: torch.Tensor) -> dict:
        """Kalman手法の不確実性分析"""
        if not kalman_result['has_uncertainty'] or true_states is None:
            return {'analysis_available': False}
        
        uncertainty_analysis = {}
        
        # 不確実性の基本統計
        covariances = kalman_result['covariances']
        uncertainties = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))
        
        uncertainty_analysis['basic_stats'] = {
            'mean_uncertainty': uncertainties.mean().item(),
            'std_uncertainty': uncertainties.std().item(),
            'min_uncertainty': uncertainties.min().item(),
            'max_uncertainty': uncertainties.max().item()
        }
        
        # カバレッジ率（簡易版）
        if 'uncertainty' in kalman_result['metrics']:
            unc_metrics = kalman_result['metrics']['uncertainty']
            uncertainty_analysis['coverage_rates'] = {
                key: value for key, value in unc_metrics.items() 
                if key.startswith('coverage_')
            }
        
        return uncertainty_analysis
    
    def _calculate_improvement(self, metric_values: dict, metric_name: str) -> dict:
        """改善率の計算"""
        if len(metric_values) != 2:
            return {}
        
        methods = list(metric_values.keys())
        if 'kalman' in methods and 'deterministic' in methods:
            kalman_value = metric_values['kalman']
            det_value = metric_values['deterministic']
            
            if metric_name == 'correlation':
                # 相関は高い方が良い
                improvement = (kalman_value - det_value) / abs(det_value) * 100
            else:
                # MSE, MAE, RMSEは低い方が良い
                improvement = (det_value - kalman_value) / det_value * 100
            
            return {
                'kalman_vs_deterministic': improvement,
                'interpretation': 'positive means Kalman is better'
            }
        
        return {}
    
    def _create_comparison_summary(self, method_results: dict, comparison_analysis: dict) -> dict:
        """比較サマリの作成"""
        summary = {}
        
        # 各手法の主要指標
        for method_name, method_result in method_results.items():
            if method_result.get('success', False) and 'accuracy' in method_result['metrics']:
                acc = method_result['metrics']['accuracy']
                summary[method_name] = {
                    'mse': acc['mse'],
                    'mae': acc['mae'],
                    'rmse': acc['rmse'],
                    'correlation': acc['correlation'],
                    'processing_time': method_result['processing_time'],
                    'has_uncertainty': method_result['has_uncertainty']
                }
        
        # 比較結果サマリ
        if comparison_analysis.get('comparison_available', False):
            summary['comparison_summary'] = {}
            
            if 'accuracy_comparison' in comparison_analysis:
                acc_comp = comparison_analysis['accuracy_comparison']
                summary['comparison_summary']['accuracy'] = {
                    metric: {
                        'best_method': result['best_method'],
                        'improvement': result.get('improvement', {})
                    }
                    for metric, result in acc_comp.items()
                }
            
            if 'efficiency_comparison' in comparison_analysis:
                eff_comp = comparison_analysis['efficiency_comparison']
                if 'processing_time' in eff_comp:
                    summary['comparison_summary']['efficiency'] = {
                        'fastest_method': eff_comp['processing_time']['fastest_method'],
                        'speed_ratio': eff_comp['processing_time']['speed_ratio']
                    }
        
        return summary
    
    def _print_comparison_results(self, comparison: dict):
        """比較結果の出力"""
        print(f"\n" + "="*70)
        print(f"🔍 推定手法比較結果")
        print(f"🏷️  実験名: {comparison['experiment_info']['name']}")
        print("="*70)
        
        summary = comparison['summary']
        
        # 各手法の結果
        print(f"\n📊 手法別性能:")
        for method_name, method_summary in summary.items():
            if method_name != 'comparison_summary':
                print(f"\n  🎯 {method_name}:")
                print(f"     MSE:          {method_summary['mse']:.6f}")
                print(f"     MAE:          {method_summary['mae']:.6f}")
                print(f"     RMSE:         {method_summary['rmse']:.6f}")
                print(f"     相関係数:     {method_summary['correlation']:.4f}")
                print(f"     処理時間:     {method_summary['processing_time']:.4f}秒")
                print(f"     不確実性:     {'あり' if method_summary['has_uncertainty'] else 'なし'}")
        
        # 比較サマリ
        if 'comparison_summary' in summary:
            comp_summary = summary['comparison_summary']
            print(f"\n🔍 手法比較サマリ:")
            
            if 'accuracy' in comp_summary:
                print(f"  📈 精度比較:")
                for metric, result in comp_summary['accuracy'].items():
                    best_method = result['best_method']
                    print(f"     {metric.upper()}: {best_method} が最良")
                    
                    # 改善率表示
                    if 'improvement' in result and 'kalman_vs_deterministic' in result['improvement']:
                        improvement = result['improvement']['kalman_vs_deterministic']
                        print(f"              Kalman改善率: {improvement:+.2f}%")
            
            if 'efficiency' in comp_summary:
                eff = comp_summary['efficiency']
                print(f"  ⚡ 効率比較:")
                print(f"     最高速: {eff['fastest_method']}")
                print(f"     速度比: {eff['speed_ratio']:.2f}x")
        
        # Kalman特有の分析
        if 'uncertainty_analysis' in comparison['comparison_analysis']:
            unc_analysis = comparison['comparison_analysis']['uncertainty_analysis']
            print(f"\n🎲 Kalman不確実性分析:")
            
            if 'basic_stats' in unc_analysis:
                stats = unc_analysis['basic_stats']
                print(f"     平均不確実性: {stats['mean_uncertainty']:.6f}")
                print(f"     不確実性範囲: [{stats['min_uncertainty']:.6f}, {stats['max_uncertainty']:.6f}]")
            
            if 'coverage_rates' in unc_analysis:
                coverage = unc_analysis['coverage_rates']
                for level, rate in coverage.items():
                    if isinstance(rate, (int, float)):
                        level_num = level.split('_')[1] if '_' in level else level
                        print(f"     {level_num}カバレッジ: {rate:.4f}")
        
        print("="*70)
        print(f"✅ 比較完了")
    
    def _save_comparison_results(self, comparison: dict, experiment_name: str):
        """比較結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON詳細結果
        json_path = self.output_dir / f"{experiment_name}_{timestamp}_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(self._make_json_serializable(comparison), f, indent=2)
        
        # CSV サマリ
        self._save_comparison_csv(comparison, experiment_name, timestamp)
        
        print(f"\n📁 比較結果保存完了:")
        print(f"   詳細結果: {json_path}")
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def _save_comparison_csv(self, comparison: dict, experiment_name: str, timestamp: str):
        """比較結果をCSV形式で保存"""
        csv_path = self.output_dir / f"{experiment_name}_{timestamp}_comparison.csv"
        summary = comparison['summary']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'experiment_name', 'method', 'mse', 'mae', 'rmse', 'correlation',
                'processing_time', 'has_uncertainty'
            ])
            
            # 各手法の結果
            for method_name, method_summary in summary.items():
                if method_name != 'comparison_summary':
                    writer.writerow([
                        experiment_name, method_name,
                        method_summary['mse'], method_summary['mae'], 
                        method_summary['rmse'], method_summary['correlation'],
                        method_summary['processing_time'], method_summary['has_uncertainty']
                    ])
        
        print(f"   比較CSV: {csv_path}")
    
    def _make_json_serializable(self, obj):
        """JSON対応形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="DFIV推定手法比較")
    
    parser.add_argument('--model_path', required=True, help='学習済みモデルパス')
    parser.add_argument('--data_path', required=True, help='評価データパス')
    parser.add_argument('--output_dir', required=True, help='結果出力ディレクトリ')
    parser.add_argument('--config', default='configs/inference_config.yaml', help='設定ファイル')
    parser.add_argument('--experiment_name', default=None, help='実験名')
    parser.add_argument('--data_split', default='test', choices=['test', 'val', 'all'], help='データ分割')
    parser.add_argument('--device', default='auto', help='計算デバイス')
    parser.add_argument('--no_save', action='store_true', help='結果保存をスキップ')
    
    args = parser.parse_args()
    
    # 引数検証
    for path_arg, path_value in [('model_path', args.model_path), ('data_path', args.data_path), ('config', args.config)]:
        if not Path(path_value).exists():
            print(f"❌ {path_arg}が見つかりません: {path_value}")
            return
    
    try:
        # 比較実行
        comparator = EstimationMethodComparator(
            model_path=args.model_path,
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device
        )
        
        results = comparator.compare_methods(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            data_split=args.data_split,
            save_results=not args.no_save
        )

        print(f"\n🎉 推定手法比較完了！")
        print(f"📊 比較結果: {len(results.get('method_results', {}))}個の手法を比較")

        # 簡潔なサマリー表示
        if 'method_results' in results:
            for method_name, result in results['method_results'].items():
                status = "✅ 成功" if result.get('success', False) else "❌ 失敗"
                print(f"  • {method_name}: {status}")
                if not result.get('success', False) and 'error' in result:
                    print(f"    エラー: {result['error'][:100]}...")
        
    except Exception as e:
        print(f"\n❌ 比較中にエラーが発生: {e}")
        raise


if __name__ == "__main__":
    main()