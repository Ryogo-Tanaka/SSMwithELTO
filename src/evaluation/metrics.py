"""
Evaluation metrics module.

Provides comprehensive metrics for DFIV Kalman Filter state estimation:
- Estimation accuracy (MSE, MAE, RMSE)
- Uncertainty quantification quality (coverage rate, interval width)
- Prediction performance (log-likelihood, calibration)
- Computational efficiency (time, memory)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import r2_score
import warnings


class StateEstimationMetrics:
    """State estimation performance evaluation."""
    
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
        Compute comprehensive evaluation metrics.

        Args:
            X_estimated: Estimated states (T, r)
            X_true: True states (T, r) [optional]
            X_covariances: State covariances (T, r, r) [optional]
            observations: Observation data (T, n) [optional]
            likelihoods: Observation likelihoods (T,) [optional]
            verbose: Whether to print results

        Returns:
            Dict of all evaluation metrics.
        """
        metrics = {}

        metrics['basic_stats'] = self._compute_basic_stats(X_estimated)

        if X_true is not None:
            metrics['accuracy'] = self._compute_accuracy_metrics(X_estimated, X_true)

        if X_covariances is not None:
            metrics['uncertainty'] = self._compute_uncertainty_metrics(
                X_estimated, X_covariances, X_true
            )

        if likelihoods is not None:
            metrics['likelihood'] = self._compute_likelihood_metrics(likelihoods)

        if observations is not None:
            metrics['prediction'] = self._compute_prediction_metrics(
                X_estimated, observations
            )

        if verbose:
            self._print_metrics_summary(metrics)
            
        return metrics
    
    def _compute_basic_stats(self, X_estimated: torch.Tensor) -> Dict[str, float]:
        """Basic statistics."""
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
        """Estimation accuracy metrics."""
        with torch.no_grad():
            errors = X_estimated - X_true
            squared_errors = errors ** 2
            abs_errors = torch.abs(errors)

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
        """Uncertainty quantification quality."""
        with torch.no_grad():
            std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))
            
            metrics = {
                'mean_uncertainty': std_devs.mean().item(),
                'std_uncertainty': std_devs.std().item(),
                'uncertainty_per_dimension': std_devs.mean(dim=0).tolist(),
                'determinant_mean': torch.det(X_covariances).mean().item(),
                'trace_mean': torch.trace(X_covariances.view(-1, X_covariances.size(-1), X_covariances.size(-1))).mean().item()
            }
            
            # Coverage rate (when ground truth is available)
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
        """Confidence interval coverage rates."""
        coverage_results = {}
        
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)

            lower = X_estimated - z_score * std_devs
            upper = X_estimated + z_score * std_devs

            covered = (X_true >= lower) & (X_true <= upper)
            coverage_rate = covered.all(dim=1).float().mean().item()
            
            coverage_results[f'coverage_{int(conf_level*100)}'] = coverage_rate
            coverage_results[f'coverage_error_{int(conf_level*100)}'] = abs(coverage_rate - conf_level)
            
        return coverage_results
    
    def _compute_likelihood_metrics(self, likelihoods: torch.Tensor) -> Dict[str, float]:
        """Likelihood-related metrics."""
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
        """Prediction performance metrics."""
        # One-step-ahead prediction error (simplified)
        if X_estimated.size(0) > 1:
            pred_errors = []
            for t in range(1, X_estimated.size(0)):
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
        """Correlation coefficient of state estimation."""
        correlations = []
        for dim in range(X_estimated.size(1)):
            est_dim = X_estimated[:, dim]
            true_dim = X_true[:, dim]

            est_norm = (est_dim - est_dim.mean()) / est_dim.std()
            true_norm = (true_dim - true_dim.mean()) / true_dim.std()

            corr = (est_norm * true_norm).mean()
            correlations.append(corr)
            
        return torch.stack(correlations).mean()
    
    def _compute_likelihood_trend(self, likelihoods: torch.Tensor) -> float:
        """Likelihood trend analysis via linear regression."""
        if len(likelihoods) < 3:
            return 0.0

        x = torch.arange(len(likelihoods), dtype=torch.float32)
        y = likelihoods
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        
        return slope.item()
    
    def _print_metrics_summary(self, metrics: Dict) -> None:
        """Print metrics summary."""
        print("\n" + "="*50)
        print("Filtering Performance Evaluation")
        print("="*50)

        if 'basic_stats' in metrics:
            stats = metrics['basic_stats']
            print(f"\nBasic Statistics:")
            print(f"  Sequence length: {stats['sequence_length']}")
            print(f"  State dimension: {stats['state_dimension']}")
            print(f"  Mean state norm: {stats['mean_state_norm']:.4f}")
            print(f"  Std state norm: {stats['std_state_norm']:.4f}")

        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"\nEstimation Accuracy:")
            print(f"  MSE: {acc['mse']:.6f}")
            print(f"  MAE: {acc['mae']:.6f}")
            print(f"  RMSE: {acc['rmse']:.6f}")
            print(f"  Correlation: {acc['correlation']:.4f}")
            print(f"  Relative error: {acc['relative_error']:.4f}")

        if 'uncertainty' in metrics:
            unc = metrics['uncertainty']
            print(f"\nUncertainty Quantification:")
            print(f"  Mean uncertainty: {unc['mean_uncertainty']:.6f}")

            for key, value in unc.items():
                if key.startswith('coverage_') and not key.endswith('_error'):
                    conf_level = key.split('_')[1]
                    error_key = f'coverage_error_{conf_level}'
                    error = unc.get(error_key, 0.0)
                    print(f"  {conf_level}% CI coverage: {value:.4f} (error: {error:.4f})")

        if 'likelihood' in metrics:
            like = metrics['likelihood']
            print(f"\nLikelihood Evaluation:")
            print(f"  Total log-likelihood: {like['total_log_likelihood']:.2f}")
            print(f"  Mean log-likelihood: {like['mean_log_likelihood']:.4f}")
            print(f"  Perplexity: {like['perplexity']:.4f}")

        print("\n" + "="*50)


class ComputationalMetrics:
    """Computational efficiency evaluation."""
    
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
        """Measure inference time."""
        times = []

        for _ in range(warmup):
            _ = inference_func(*args)

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
        """Measure memory usage."""
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
    """Calibration evaluation."""
    
    @staticmethod
    def compute_calibration_error(
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        true_values: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Compute calibration error."""
        # Convert uncertainties to probabilities (Gaussian assumption)
        probabilities = torch.sigmoid(uncertainties)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            if in_bin.sum() == 0:
                continue

            expected_confidence = probabilities[in_bin].mean()
            actual_accuracy = ((predictions[in_bin] - true_values[in_bin]).abs() < uncertainties[in_bin]).float().mean()

            bin_weight = in_bin.sum().float() / len(probabilities)
            calibration_error += bin_weight * abs(expected_confidence - actual_accuracy)
            
        return calibration_error.item()


def create_metrics_evaluator(device: str = 'cpu') -> StateEstimationMetrics:
    """Create a metrics evaluator."""
    return StateEstimationMetrics(device=device)


def print_comparison_summary(
    method1_metrics: Dict,
    method2_metrics: Dict,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> None:
    """Print comparison summary of two methods."""
    print(f"\nMethod Comparison: {method1_name} vs {method2_name}")
    print("="*60)

    if 'accuracy' in method1_metrics and 'accuracy' in method2_metrics:
        acc1 = method1_metrics['accuracy']
        acc2 = method2_metrics['accuracy']

        print(f"\nAccuracy Comparison:")
        print(f"  MSE:  {method1_name}: {acc1['mse']:.6f}  |  {method2_name}: {acc2['mse']:.6f}")
        print(f"  MAE:  {method1_name}: {acc1['mae']:.6f}  |  {method2_name}: {acc2['mae']:.6f}")
        print(f"  RMSE: {method1_name}: {acc1['rmse']:.6f}  |  {method2_name}: {acc2['rmse']:.6f}")

        mse_improvement = (acc1['mse'] - acc2['mse']) / acc1['mse'] * 100
        mae_improvement = (acc1['mae'] - acc2['mae']) / acc1['mae'] * 100

        print(f"\nImprovement ({method2_name} vs {method1_name}):")
        print(f"  MSE improvement: {mse_improvement:+.2f}%")
        print(f"  MAE improvement: {mae_improvement:+.2f}%")

    print("="*60)


class TargetPredictionMetrics:
    """
    Target prediction evaluation (unified interface with StateEstimationMetrics).

    Provides evaluation metrics for target prediction experiments:
    - Selectable metrics (RMSE, MAE, R^2, per-dimension R^2)
    - Unified terminal output format
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def compute_target_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['rmse'],
        verbose: bool = True
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute target prediction evaluation metrics.

        Args:
            y_true: Ground truth tensor (T, d)
            y_pred: Predicted tensor (T, d)
            metrics: List of metrics to compute ['mse', 'rmse', 'mae', 'r2', 'r2_per_dim']
            verbose: Whether to print results

        Returns:
            Dict of evaluation metric results.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        available_metrics = {
            'mse': lambda: F.mse_loss(y_pred, y_true).item(),
            'rmse': lambda: torch.sqrt(F.mse_loss(y_pred, y_true)).item(),
            'mae': lambda: F.l1_loss(y_pred, y_true).item(),
            'r2': lambda: self._compute_r2_score(y_true, y_pred),
            'r2_per_dim': lambda: self._compute_r2_per_dimension(y_true, y_pred)
        }

        results = {}
        for metric in metrics:
            if metric in available_metrics:
                try:
                    results[metric] = available_metrics[metric]()
                except Exception as e:
                    print(f"Error computing metric '{metric}': {e}")
                    results[metric] = None
            else:
                print(f"Unknown metric: '{metric}'. Available: {list(available_metrics.keys())}")
        if verbose:
            self._print_target_metrics_summary(results)

        return results

    def save_target_metrics_results(
        self,
        results: Dict[str, Union[float, List[float]]],
        output_dir: str,
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save target prediction evaluation results to JSON.

        Args:
            results: Output of compute_target_metrics()
            output_dir: Output directory
            experiment_info: Additional experiment info (optional)

        Returns:
            Path of the saved file.
        """
        import json
        from datetime import datetime
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'target_prediction',
            'metrics': results,
            'experiment_info': experiment_info or {}
        }

        save_file = output_path / 'target_prediction_metrics.json'
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"Target prediction results saved: {save_file}")
        return str(save_file)

    def create_target_visualizations(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['rmse'],
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate target prediction visualizations (placeholder).

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            metrics: List of metrics to visualize
            output_dir: Output directory (None to skip saving)

        Returns:
            List of generated file paths.
        """
        # Visualization not yet implemented; use numerical output instead
        generated_files = []
        return generated_files

    def _compute_r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute overall R^2 score."""
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            return float(r2_score(y_true_np, y_pred_np, multioutput='uniform_average'))
        except Exception as e:
            print(f"R^2 computation error: {e}")
            return 0.0

    def _compute_r2_per_dimension(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> List[float]:
        """Compute per-dimension R^2 scores."""
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            r2_values = r2_score(y_true_np, y_pred_np, multioutput='raw_values')
            return r2_values.tolist()
        except Exception as e:
            print(f"Per-dimension R^2 computation error: {e}")
            return [0.0]

    def _plot_individual_metric(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metric: str,
        save_path: Optional[Path] = None,
        max_samples: int = 200
    ) -> None:
        """
        Visualization for individual metrics (placeholder).

        Supported metrics: rmse, mae, r2, r2_per_dim.
        Currently disabled; use numerical output instead.
        """
        pass

    def _print_target_metrics_summary(self, metrics: Dict[str, Union[float, List[float]]]) -> None:
        """Print target prediction metrics summary."""
        print("\n" + "="*50)
        print("Target Prediction Evaluation")
        print("="*50)

        for metric_name, value in metrics.items():
            if value is None:
                print(f"  {metric_name.upper()}: computation error")
            elif isinstance(value, list):
                if metric_name == 'r2_per_dim':
                    print(f"  R^2 PER DIMENSION:")
                    for i, dim_r2 in enumerate(value):
                        print(f"    Dim {i}: {dim_r2:.4f}")
                    print(f"    Average: {np.mean(value):.4f}")
                else:
                    print(f"  {metric_name.upper()}: {value}")
            else:
                print(f"  {metric_name.upper()}: {value:.4f}")

        print("="*50)


class ReconstructionMetrics:
    """
    Data reconstruction evaluation (unified interface with TargetPredictionMetrics).

    Provides metrics for reconstruction experiments on arbitrary data types
    (images: T,H,W,C / time series: T,d / etc.):
    - reconstruction_rmse, psnr, temporal_correlation
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def compute_reconstruction_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['reconstruction_rmse'],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute reconstruction evaluation metrics.

        Args:
            y_true: Ground truth tensor (arbitrary shape: T,H,W,C / T,d / etc.)
            y_pred: Predicted tensor (same shape as y_true)
            metrics: List of metrics ['reconstruction_rmse', 'psnr', 'temporal_correlation']
            verbose: Whether to print results

        Returns:
            Dict of reconstruction evaluation metrics.
        """
        y_true = y_true.to(self.device).detach()
        y_pred = y_pred.to(self.device).detach()

        if y_true.shape != y_pred.shape:
            print(f"Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")
            return {'error': 1.0}

        results = {}

        for metric in metrics:
            try:
                if metric == 'reconstruction_rmse':
                    results[metric] = self._compute_reconstruction_rmse(y_true, y_pred)
                elif metric == 'psnr':
                    results[metric] = self._compute_psnr(y_true, y_pred)
                elif metric == 'temporal_correlation':
                    results[metric] = self._compute_temporal_correlation(y_true, y_pred)
                else:
                    print(f"Unknown metric: '{metric}'")
                    results[metric] = 0.0

            except Exception as e:
                print(f"Error computing metric '{metric}': {e}")
                results[metric] = 0.0

        if verbose:
            self._print_reconstruction_metrics_summary(results)

        return results

    def create_reconstruction_visualizations(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        metrics: List[str] = ['reconstruction_rmse'],
        output_dir: str = None
    ) -> List[str]:
        """
        Generate reconstruction visualizations (placeholder).

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            metrics: List of metrics to visualize
            output_dir: Output directory

        Returns:
            List of generated file paths.
        """
        # Visualization not yet implemented; use numerical output instead
        generated_files = []
        return generated_files

    def _compute_reconstruction_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Reconstruction RMSE for arbitrary data types."""
        mse = torch.mean((y_true - y_pred) ** 2).item()
        rmse = mse ** 0.5
        return float(rmse)

    def _compute_psnr(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """PSNR (Peak Signal-to-Noise Ratio) for arbitrary data types."""
        try:
            mse = torch.mean((y_true - y_pred) ** 2).item()
            if mse == 0:
                return float('inf')

            # Auto-detect data range
            data_range = torch.max(y_true).item() - torch.min(y_true).item()
            if data_range <= 0:
                return float('inf')

            psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(torch.tensor(mse))
            return float(psnr.item())

        except Exception as e:
            print(f"PSNR computation error: {e}")
            return 0.0

    def _compute_temporal_correlation(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Temporal reconstruction correlation for arbitrary data types."""
        try:
            if len(y_true.shape) < 2:
                if torch.std(y_true) > 1e-8 and torch.std(y_pred) > 1e-8:
                    corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()
                    return float(corr) if not torch.isnan(torch.tensor(corr)) else 0.0
                return 0.0

            # Multi-dimensional: compute per-timestep correlation and average
            correlations = []
            for t in range(y_true.shape[0]):
                true_t = y_true[t].flatten()
                pred_t = y_pred[t].flatten()

                # Pearson correlation
                if torch.std(true_t) > 1e-8 and torch.std(pred_t) > 1e-8:
                    corr = torch.corrcoef(torch.stack([true_t, pred_t]))[0, 1].item()
                    if not torch.isnan(torch.tensor(corr)):
                        correlations.append(corr)

            if correlations:
                return float(sum(correlations) / len(correlations))
            else:
                return 0.0

        except Exception as e:
            print(f"Temporal correlation computation error: {e}")
            return 0.0

    def _print_reconstruction_metrics_summary(self, results: Dict[str, float]):
        """Print reconstruction evaluation summary."""
        print("\n" + "="*50)
        print("Reconstruction Metrics")
        print("="*50)

        for metric, value in results.items():
            if metric == 'reconstruction_rmse':
                print(f"  Reconstruction RMSE: {value:.6f}")
            elif metric == 'psnr':
                if value == float('inf'):
                    print(f"  PSNR: inf dB (Perfect Match)")
                else:
                    print(f"  PSNR: {value:.2f} dB")
            elif metric == 'temporal_correlation':
                print(f"  Temporal Correlation: {value:.6f}")
            else:
                print(f"  {metric}: {value:.6f}")

        print("="*50)

    def save_reconstruction_metrics_results(
        self,
        results: Dict[str, float],
        output_dir: str,
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save reconstruction evaluation results to JSON.

        Args:
            results: Output of compute_reconstruction_metrics()
            output_dir: Output directory
            experiment_info: Additional experiment info (optional)

        Returns:
            Path of the saved file.
        """
        import json
        from datetime import datetime
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'reconstruction',
            'metrics': results,
        }

        if experiment_info:
            save_data['experiment_info'] = experiment_info

        save_file = output_path / 'reconstruction_metrics.json'
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Reconstruction results saved: {save_file}")
        return str(save_file)


def create_target_prediction_evaluator(device: str = 'cpu') -> TargetPredictionMetrics:
    """Create a target prediction evaluator."""
    return TargetPredictionMetrics(device=device)


def create_reconstruction_evaluator(device: str = 'cpu') -> ReconstructionMetrics:
    """Create a reconstruction evaluator."""
    return ReconstructionMetrics(device=device)