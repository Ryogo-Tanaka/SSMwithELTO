"""
Reconstruction evaluation metrics.

The public experiment pipeline uses this module to evaluate image and
time-series reconstructions and save a compact JSON summary.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class ReconstructionMetrics:
    """
    Metrics for reconstruction experiments on arbitrary data types
    (images: T,H,W,C / time series: T,d / etc.).
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
            y_true: Ground truth tensor (same shape as y_pred)
            y_pred: Predicted tensor
            metrics: Metric names: reconstruction_rmse, psnr, temporal_correlation
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

    def _compute_reconstruction_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Reconstruction RMSE."""
        mse = torch.mean((y_true - y_pred) ** 2).item()
        return float(mse ** 0.5)

    def _compute_psnr(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """PSNR (Peak Signal-to-Noise Ratio)."""
        try:
            mse = torch.mean((y_true - y_pred) ** 2).item()
            if mse == 0:
                return float('inf')

            data_range = torch.max(y_true).item() - torch.min(y_true).item()
            if data_range <= 0:
                return float('inf')

            psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(torch.tensor(mse))
            return float(psnr.item())
        except Exception as e:
            print(f"PSNR computation error: {e}")
            return 0.0

    def _compute_temporal_correlation(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Temporal reconstruction correlation."""
        try:
            if len(y_true.shape) < 2:
                if torch.std(y_true) > 1e-8 and torch.std(y_pred) > 1e-8:
                    corr = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1].item()
                    return float(corr) if not torch.isnan(torch.tensor(corr)) else 0.0
                return 0.0

            correlations = []
            for t in range(y_true.shape[0]):
                true_t = y_true[t].flatten()
                pred_t = y_pred[t].flatten()

                if torch.std(true_t) > 1e-8 and torch.std(pred_t) > 1e-8:
                    corr = torch.corrcoef(torch.stack([true_t, pred_t]))[0, 1].item()
                    if not torch.isnan(torch.tensor(corr)):
                        correlations.append(corr)

            return float(sum(correlations) / len(correlations)) if correlations else 0.0
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
                    print("  PSNR: inf dB (Perfect Match)")
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
            experiment_info: Additional experiment info

        Returns:
            Path of the saved file.
        """
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
