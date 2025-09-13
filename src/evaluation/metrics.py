"""
評価指標計算モジュール

DFIV Kalman Filterの状態推定性能を包括的に評価するための指標を提供。
- 推定精度（MSE, MAE, RMSE）
- 不確実性定量化品質（カバレッジ率、区間幅）
- 予測性能（対数尤度、キャリブレーション）
- 計算効率（時間、メモリ）
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings


class StateEstimationMetrics:
    """状態推定性能評価クラス"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
    def compute_all_metrics(
        self,
        X_estimated: torch.Tensor,
        X_true: Optional[torch.Tensor] = None,
        X_covariances: Optional[torch.Tensor] = None,
        observations: Optional[torch.Tensor] = None,
        likelihoods: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict[str, Union[float, Dict]]:
        """
        包括的評価指標の計算
        
        Args:
            X_estimated: 推定状態 (T, r)
            X_true: 真の状態 (T, r) [optional]
            X_covariances: 状態共分散 (T, r, r) [optional]
            observations: 観測データ (T, n) [optional]
            likelihoods: 観測尤度 (T,) [optional]
            verbose: ターミナル出力するかどうか
            
        Returns:
            Dict: 全評価指標
        """
        metrics = {}
        
        # 基本統計
        metrics['basic_stats'] = self._compute_basic_stats(X_estimated)
        
        # 精度評価（真値がある場合）
        if X_true is not None:
            metrics['accuracy'] = self._compute_accuracy_metrics(X_estimated, X_true)
            
        # 不確実性評価（共分散がある場合）
        if X_covariances is not None:
            metrics['uncertainty'] = self._compute_uncertainty_metrics(
                X_estimated, X_covariances, X_true
            )
            
        # 尤度評価
        if likelihoods is not None:
            metrics['likelihood'] = self._compute_likelihood_metrics(likelihoods)
            
        # 予測性能（観測データがある場合）
        if observations is not None:
            metrics['prediction'] = self._compute_prediction_metrics(
                X_estimated, observations
            )
            
        # ターミナル出力
        if verbose:
            self._print_metrics_summary(metrics)
            
        return metrics
    
    def _compute_basic_stats(self, X_estimated: torch.Tensor) -> Dict[str, float]:
        """基本統計情報"""
        with torch.no_grad():
            return {
                'sequence_length': X_estimated.size(0),
                'state_dimension': X_estimated.size(1),
                'mean_state_norm': torch.norm(X_estimated, dim=1).mean().item(),
                'std_state_norm': torch.norm(X_estimated, dim=1).std().item(),
                'max_state_value': X_estimated.max().item(),
                'min_state_value': X_estimated.min().item()
            }
    
    def _compute_accuracy_metrics(
        self, 
        X_estimated: torch.Tensor, 
        X_true: torch.Tensor
    ) -> Dict[str, float]:
        """推定精度指標"""
        with torch.no_grad():
            # エラー計算
            errors = X_estimated - X_true
            squared_errors = errors ** 2
            abs_errors = torch.abs(errors)
            
            # 次元ごとの指標
            mse_per_dim = squared_errors.mean(dim=0)
            mae_per_dim = abs_errors.mean(dim=0)
            
            return {
                'mse': squared_errors.mean().item(),
                'mae': abs_errors.mean().item(),
                'rmse': torch.sqrt(squared_errors.mean()).item(),
                'mse_per_dimension': mse_per_dim.tolist(),
                'mae_per_dimension': mae_per_dim.tolist(),
                'relative_error': (torch.norm(errors, dim=1) / torch.norm(X_true, dim=1)).mean().item(),
                'correlation': self._compute_correlation(X_estimated, X_true).item()
            }
    
    def _compute_uncertainty_metrics(
        self,
        X_estimated: torch.Tensor,
        X_covariances: torch.Tensor,
        X_true: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List]]:
        """不確実性定量化品質"""
        with torch.no_grad():
            # 標準偏差抽出
            std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))
            
            metrics = {
                'mean_uncertainty': std_devs.mean().item(),
                'std_uncertainty': std_devs.std().item(),
                'uncertainty_per_dimension': std_devs.mean(dim=0).tolist(),
                'determinant_mean': torch.det(X_covariances).mean().item(),
                'trace_mean': torch.trace(X_covariances.view(-1, X_covariances.size(-1), X_covariances.size(-1))).mean().item()
            }
            
            # カバレッジ率計算（真値がある場合）
            if X_true is not None:
                coverage_results = self._compute_coverage_rates(
                    X_estimated, std_devs, X_true
                )
                metrics.update(coverage_results)
                
            return metrics
    
    def _compute_coverage_rates(
        self,
        X_estimated: torch.Tensor,
        std_devs: torch.Tensor,
        X_true: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, float]:
        """信頼区間カバレッジ率"""
        coverage_results = {}
        
        for conf_level in confidence_levels:
            # Z値計算
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            # 信頼区間
            lower = X_estimated - z_score * std_devs
            upper = X_estimated + z_score * std_devs
            
            # カバレッジ判定
            covered = (X_true >= lower) & (X_true <= upper)
            coverage_rate = covered.all(dim=1).float().mean().item()
            
            coverage_results[f'coverage_{int(conf_level*100)}'] = coverage_rate
            coverage_results[f'coverage_error_{int(conf_level*100)}'] = abs(coverage_rate - conf_level)
            
        return coverage_results
    
    def _compute_likelihood_metrics(self, likelihoods: torch.Tensor) -> Dict[str, float]:
        """尤度関連指標"""
        with torch.no_grad():
            return {
                'total_log_likelihood': likelihoods.sum().item(),
                'mean_log_likelihood': likelihoods.mean().item(),
                'std_log_likelihood': likelihoods.std().item(),
                'perplexity': torch.exp(-likelihoods.mean()).item(),
                'likelihood_trend': self._compute_likelihood_trend(likelihoods)
            }
    
    def _compute_prediction_metrics(
        self, 
        X_estimated: torch.Tensor, 
        observations: torch.Tensor
    ) -> Dict[str, float]:
        """予測性能指標"""
        # 一期先予測誤差（簡略版）
        if X_estimated.size(0) > 1:
            pred_errors = []
            for t in range(1, X_estimated.size(0)):
                # 簡単な線形予測
                if t >= 2:
                    predicted = X_estimated[t-1] + (X_estimated[t-1] - X_estimated[t-2])
                else:
                    predicted = X_estimated[t-1]
                actual = X_estimated[t]
                pred_errors.append(torch.norm(actual - predicted).item())
                
            return {
                'one_step_prediction_error': np.mean(pred_errors),
                'prediction_error_std': np.std(pred_errors),
                'prediction_stability': 1.0 / (1.0 + np.std(pred_errors))
            }
        else:
            return {'prediction_error': 0.0}
    
    def _compute_correlation(self, X_estimated: torch.Tensor, X_true: torch.Tensor) -> torch.Tensor:
        """状態推定の相関係数"""
        # 各次元での相関計算
        correlations = []
        for dim in range(X_estimated.size(1)):
            est_dim = X_estimated[:, dim]
            true_dim = X_true[:, dim]
            
            # 標準化
            est_norm = (est_dim - est_dim.mean()) / est_dim.std()
            true_norm = (true_dim - true_dim.mean()) / true_dim.std()
            
            # 相関計算
            corr = (est_norm * true_norm).mean()
            correlations.append(corr)
            
        return torch.stack(correlations).mean()
    
    def _compute_likelihood_trend(self, likelihoods: torch.Tensor) -> float:
        """尤度の傾向分析"""
        if len(likelihoods) < 3:
            return 0.0
            
        # 線形回帰による傾向
        x = torch.arange(len(likelihoods), dtype=torch.float32)
        y = likelihoods
        
        # 傾きを計算
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        
        return slope.item()
    
    def _print_metrics_summary(self, metrics: Dict) -> None:
        """メトリクス結果のターミナル出力"""
        print("\n" + "="*50)
        print("📊 フィルタリング性能評価結果")
        print("="*50)
        
        # 基本統計
        if 'basic_stats' in metrics:
            stats = metrics['basic_stats']
            print(f"\n📈 基本統計:")
            print(f"  系列長: {stats['sequence_length']}")
            print(f"  状態次元: {stats['state_dimension']}")
            print(f"  平均状態ノルム: {stats['mean_state_norm']:.4f}")
            print(f"  状態ノルム標準偏差: {stats['std_state_norm']:.4f}")
        
        # 精度指標
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"\n🎯 推定精度:")
            print(f"  MSE: {acc['mse']:.6f}")
            print(f"  MAE: {acc['mae']:.6f}")
            print(f"  RMSE: {acc['rmse']:.6f}")
            print(f"  相関係数: {acc['correlation']:.4f}")
            print(f"  相対誤差: {acc['relative_error']:.4f}")
        
        # 不確実性指標
        if 'uncertainty' in metrics:
            unc = metrics['uncertainty']
            print(f"\n🎲 不確実性定量化:")
            print(f"  平均不確実性: {unc['mean_uncertainty']:.6f}")
            
            # カバレッジ率
            for key, value in unc.items():
                if key.startswith('coverage_') and not key.endswith('_error'):
                    conf_level = key.split('_')[1]
                    error_key = f'coverage_error_{conf_level}'
                    error = unc.get(error_key, 0.0)
                    print(f"  {conf_level}%信頼区間カバレッジ: {value:.4f} (誤差: {error:.4f})")
        
        # 尤度指標
        if 'likelihood' in metrics:
            like = metrics['likelihood']
            print(f"\n📈 尤度評価:")
            print(f"  総対数尤度: {like['total_log_likelihood']:.2f}")
            print(f"  平均対数尤度: {like['mean_log_likelihood']:.4f}")
            print(f"  パープレキシティ: {like['perplexity']:.4f}")
            
        print("\n" + "="*50)


class ComputationalMetrics:
    """計算効率評価クラス"""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
        
    def measure_inference_time(
        self, 
        inference_func, 
        *args, 
        n_trials: int = 5,
        warmup: int = 2
    ) -> Dict[str, float]:
        """推論時間測定"""
        times = []
        
        # ウォームアップ
        for _ in range(warmup):
            _ = inference_func(*args)
            
        # 測定
        for trial in range(n_trials):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            result = inference_func(*args)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'trials': n_trials
        }
    
    def measure_memory_usage(self, function_to_measure, *args) -> Dict[str, float]:
        """メモリ使用量測定"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            initial_memory = torch.cuda.memory_allocated()
            
            result = function_to_measure(*args)
            
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            return {
                'initial_memory_mb': initial_memory / (1024**2),
                'peak_memory_mb': peak_memory / (1024**2),
                'final_memory_mb': final_memory / (1024**2),
                'memory_increase_mb': (final_memory - initial_memory) / (1024**2),
                'peak_increase_mb': (peak_memory - initial_memory) / (1024**2)
            }
        else:
            return {'message': 'CUDA not available, memory measurement skipped'}


class CalibrationMetrics:
    """キャリブレーション評価クラス"""
    
    @staticmethod
    def compute_calibration_error(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """キャリブレーション誤差の計算"""
        # 不確実性を確率に変換（正規分布仮定）
        probabilities = torch.sigmoid(uncertainties)
        
        # ビンごとの分析
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # ビン内のサンプル
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            if in_bin.sum() == 0:
                continue
                
            # 期待信頼度と実際の精度
            expected_confidence = probabilities[in_bin].mean()
            actual_accuracy = ((predictions[in_bin] - true_values[in_bin]).abs() < uncertainties[in_bin]).float().mean()
            
            # ビンの重み
            bin_weight = in_bin.sum().float() / len(probabilities)
            
            # 誤差累積
            calibration_error += bin_weight * abs(expected_confidence - actual_accuracy)
            
        return calibration_error.item()


def create_metrics_evaluator(device: str = 'cpu') -> StateEstimationMetrics:
    """メトリクス評価器の作成"""
    return StateEstimationMetrics(device=device)


def print_comparison_summary(
    method1_metrics: Dict, 
    method2_metrics: Dict, 
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> None:
    """2手法の比較結果を出力"""
    print(f"\n🔍 手法比較: {method1_name} vs {method2_name}")
    print("="*60)
    
    # 精度比較
    if 'accuracy' in method1_metrics and 'accuracy' in method2_metrics:
        acc1 = method1_metrics['accuracy']
        acc2 = method2_metrics['accuracy']
        
        print(f"\n📊 精度比較:")
        print(f"  MSE:  {method1_name}: {acc1['mse']:.6f}  |  {method2_name}: {acc2['mse']:.6f}")
        print(f"  MAE:  {method1_name}: {acc1['mae']:.6f}  |  {method2_name}: {acc2['mae']:.6f}")
        print(f"  RMSE: {method1_name}: {acc1['rmse']:.6f}  |  {method2_name}: {acc2['rmse']:.6f}")
        
        # 改善率計算
        mse_improvement = (acc1['mse'] - acc2['mse']) / acc1['mse'] * 100
        mae_improvement = (acc1['mae'] - acc2['mae']) / acc1['mae'] * 100
        
        print(f"\n💡 改善率 ({method2_name} vs {method1_name}):")
        print(f"  MSE改善: {mse_improvement:+.2f}%")
        print(f"  MAE改善: {mae_improvement:+.2f}%")
    
    print("="*60)