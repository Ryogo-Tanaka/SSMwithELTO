# src/inference/streaming.py
"""
ストリーミング処理: リアルタイム状態推定

StateEstimatorをラップしてストリーミング用の追加機能を提供:
- バッファ管理
- 異常検知
- 予測機能
- パフォーマンスモニタリング
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Generator, Union
from collections import deque
import time
import warnings
from dataclasses import dataclass

from .state_estimator import StateEstimator


@dataclass
class StreamingMetrics:
    """ストリーミング処理の性能指標"""
    total_observations: int = 0
    processing_times: List[float] = None
    likelihood_history: List[float] = None
    anomaly_count: int = 0
    buffer_utilization: float = 0.0
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = []
        if self.likelihood_history is None:
            self.likelihood_history = []


class StreamingEstimator:
    """
    StateEstimatorをラップしたストリーミング推論エンジン
    
    リアルタイム処理に特化した機能:
    - 逐次データ処理
    - バッファ管理  
    - 異常検知
    - パフォーマンス監視
    """
    
    def __init__(
        self,
        state_estimator: StateEstimator,
        buffer_size: int = 100,
        anomaly_threshold: float = 3.0,
        enable_metrics: bool = True
    ):
        """
        Args:
            state_estimator: 基底StateEstimatorインスタンス
            buffer_size: 内部バッファサイズ
            anomaly_threshold: 異常検知閾値（標準偏差倍数）
            enable_metrics: メトリクス収集有効化
        """
        self.estimator = state_estimator
        self.buffer_size = buffer_size
        self.anomaly_threshold = anomaly_threshold
        self.enable_metrics = enable_metrics
        
        # バッファ（dequeで効率的なFIFO）
        self.observation_buffer = deque(maxlen=buffer_size)
        self.state_buffer = deque(maxlen=buffer_size)
        self.covariance_buffer = deque(maxlen=buffer_size)
        self.likelihood_buffer = deque(maxlen=buffer_size)
        
        # メトリクス
        self.metrics = StreamingMetrics()
        
        # 異常検知用統計
        self.likelihood_mean = 0.0
        self.likelihood_std = 1.0
        self.likelihood_update_count = 0
        
        # 状態
        self.is_streaming = False
        self.last_update_time = None

    def start_streaming(self, initial_data: Optional[torch.Tensor] = None):
        """
        ストリーミング開始
        
        Args:
            initial_data: 初期化用データ (N0, n)
        """
        if not self.estimator.is_initialized:
            if initial_data is not None:
                self.estimator.initialize_filtering(initial_data)
            else:
                raise RuntimeError("Estimator not initialized and no initial data provided")
                
        self.is_streaming = True
        self.last_update_time = time.time()
        
        print("Streaming estimation started")

    def stop_streaming(self):
        """ストリーミング停止"""
        self.is_streaming = False
        print("Streaming estimation stopped")

    def add_observation(
        self,
        observation: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        新しい観測の追加
        
        バッファ管理と状態更新を実行
        
        Args:
            observation: 新しい観測 (n,)
            timestamp: タイムスタンプ（Noneの場合は現在時刻）
            
        Returns:
            x_hat: 推定状態 (r,)
            Sigma_x: 状態共分散 (r, r)
            info: 処理情報辞書
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
            
        start_time = time.time()
        if timestamp is None:
            timestamp = start_time
            
        # 状態更新
        x_hat, Sigma_x, likelihood = self.estimator.filter_online(observation)
        
        # バッファ更新
        self.observation_buffer.append(observation.cpu())
        self.state_buffer.append(x_hat.cpu())
        self.covariance_buffer.append(Sigma_x.cpu())
        self.likelihood_buffer.append(likelihood)
        
        # 異常検知
        is_anomaly = self._detect_anomaly(likelihood)
        
        # メトリクス更新
        processing_time = time.time() - start_time
        info = self._update_metrics(processing_time, likelihood, is_anomaly, timestamp)
        
        self.last_update_time = timestamp
        
        return x_hat, Sigma_x, info

    def process_stream(
        self,
        data_stream: Generator[torch.Tensor, None, None]
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], None, None]:
        """
        データストリームの処理
        
        yielding形式で逐次結果を返す
        
        Args:
            data_stream: 観測データのジェネレータ
            
        Yields:
            tuple: (x_hat, Sigma_x, info) for each observation
        """
        for observation in data_stream:
            try:
                result = self.add_observation(observation)
                yield result
            except Exception as e:
                warnings.warn(f"Streaming processing error: {e}")
                # エラー時はダミー結果を返す
                dummy_state = torch.zeros(self.estimator.kalman_filter.r)
                dummy_cov = torch.eye(self.estimator.kalman_filter.r)
                dummy_info = {"error": str(e), "timestamp": time.time()}
                yield dummy_state, dummy_cov, dummy_info

    def get_predictions(
        self,
        n_steps: int = 1,
        include_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        n期先予測
        
        現在状態から将来予測
        
        Args:
            n_steps: 予測ステップ数
            include_uncertainty: 不確実性も含めるか
            
        Returns:
            Dict: 予測結果
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming not active")
            
        # 予測実行
        x_pred, Sigma_pred = self.estimator.predict_ahead(n_steps)
        
        results = {
            "predictions": x_pred,       # (n_steps, r)
            "prediction_steps": n_steps
        }
        
        if include_uncertainty:
            # 予測区間計算（±2σ）
            std_devs = torch.sqrt(torch.diagonal(Sigma_pred, dim1=1, dim2=2))  # (n_steps, r)
            results.update({
                "covariances": Sigma_pred,
                "lower_bounds": x_pred - 2 * std_devs,
                "upper_bounds": x_pred + 2 * std_devs
            })
            
        return results

    def detect_anomalies(
        self,
        method: str = "likelihood",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        異常検知（イノベーション基準）
        
        統計的外れ値検出
        
        Args:
            method: 検知手法 ("likelihood" | "innovation")
            threshold: 閾値（Noneの場合はデフォルト使用）
            
        Returns:
            Dict: 異常検知結果
        """
        if threshold is None:
            threshold = self.anomaly_threshold
            
        if method == "likelihood":
            return self._detect_likelihood_anomalies(threshold)
        elif method == "innovation":
            return self._detect_innovation_anomalies(threshold)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

    def _detect_anomaly(self, likelihood: float) -> bool:
        """
        単一観測の異常検知
        
        Args:
            likelihood: 観測尤度
            
        Returns:
            bool: 異常かどうか
        """
        # 尤度統計の更新（オンライン平均・分散）
        self.likelihood_update_count += 1
        delta = likelihood - self.likelihood_mean
        self.likelihood_mean += delta / self.likelihood_update_count
        delta2 = likelihood - self.likelihood_mean
        
        if self.likelihood_update_count > 1:
            variance_update = delta * delta2 / (self.likelihood_update_count - 1)
            self.likelihood_std = np.sqrt(max(0, variance_update))
        
        # 異常判定（尤度が平均から大きく外れている）
        if self.likelihood_update_count > 10:  # 最低10サンプル必要
            z_score = abs(likelihood - self.likelihood_mean) / (self.likelihood_std + 1e-8)
            return z_score > self.anomaly_threshold
            
        return False

    def _detect_likelihood_anomalies(self, threshold: float) -> Dict[str, Any]:
        """尤度ベース異常検知"""
        if len(self.likelihood_buffer) < 10:
            return {"method": "likelihood", "anomalies": [], "message": "Insufficient data"}
            
        likelihoods = np.array(list(self.likelihood_buffer))
        mean_ll = np.mean(likelihoods)
        std_ll = np.std(likelihoods)
        
        # Z-score異常検知
        z_scores = np.abs(likelihoods - mean_ll) / (std_ll + 1e-8)
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        return {
            "method": "likelihood",
            "threshold": threshold,
            "anomalies": anomaly_indices.tolist(),
            "z_scores": z_scores.tolist(),
            "statistics": {
                "mean_likelihood": mean_ll,
                "std_likelihood": std_ll
            }
        }

    def _detect_innovation_anomalies(self, threshold: float) -> Dict[str, Any]:
        """イノベーションベース異常検知"""
        if len(self.state_buffer) < 2:
            return {"method": "innovation", "anomalies": [], "message": "Insufficient data"}
            
        # 状態変化の分析
        states = torch.stack(list(self.state_buffer))  # (buffer_size, r)
        state_diffs = torch.diff(states, dim=0)        # (buffer_size-1, r)
        
        # 状態変化の大きさ
        change_magnitudes = torch.norm(state_diffs, dim=1)  # (buffer_size-1,)
        
        mean_change = torch.mean(change_magnitudes)
        std_change = torch.std(change_magnitudes)
        
        # 異常検知
        z_scores = torch.abs(change_magnitudes - mean_change) / (std_change + 1e-8)
        anomaly_indices = torch.where(z_scores > threshold)[0]
        
        return {
            "method": "innovation",
            "threshold": threshold,
            "anomalies": anomaly_indices.tolist(),
            "z_scores": z_scores.tolist(),
            "statistics": {
                "mean_change": mean_change.item(),
                "std_change": std_change.item()
            }
        }

    def _update_metrics(
        self,
        processing_time: float,
        likelihood: float,
        is_anomaly: bool,
        timestamp: float
    ) -> Dict[str, Any]:
        """メトリクス更新"""
        if not self.enable_metrics:
            return {}
            
        self.metrics.total_observations += 1
        self.metrics.processing_times.append(processing_time)
        self.metrics.likelihood_history.append(likelihood)
        
        if is_anomaly:
            self.metrics.anomaly_count += 1
            
        self.metrics.buffer_utilization = len(self.observation_buffer) / self.buffer_size
        
        # 処理情報
        info = {
            "timestamp": timestamp,
            "processing_time": processing_time,
            "likelihood": likelihood,
            "is_anomaly": is_anomaly,
            "buffer_utilization": self.metrics.buffer_utilization,
            "total_processed": self.metrics.total_observations
        }
        
        return info

    def get_buffer_statistics(self) -> Dict[str, Any]:
        """バッファ統計取得"""
        if len(self.state_buffer) == 0:
            return {"status": "empty"}
            
        states = torch.stack(list(self.state_buffer))
        likelihoods = np.array(list(self.likelihood_buffer))
        
        return {
            "buffer_size": len(self.state_buffer),
            "utilization": len(self.state_buffer) / self.buffer_size,
            "state_statistics": {
                "mean": torch.mean(states, dim=0).tolist(),
                "std": torch.std(states, dim=0).tolist(),
                "min": torch.min(states, dim=0)[0].tolist(),
                "max": torch.max(states, dim=0)[0].tolist()
            },
            "likelihood_statistics": {
                "mean": np.mean(likelihoods),
                "std": np.std(likelihoods),
                "min": np.min(likelihoods),
                "max": np.max(likelihoods)
            }
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標取得"""
        if not self.enable_metrics or len(self.metrics.processing_times) == 0:
            return {"status": "no_metrics"}
            
        processing_times = np.array(self.metrics.processing_times)
        
        return {
            "total_observations": self.metrics.total_observations,
            "processing_performance": {
                "mean_time": np.mean(processing_times),
                "std_time": np.std(processing_times),
                "min_time": np.min(processing_times),
                "max_time": np.max(processing_times),
                "throughput_hz": 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0
            },
            "anomaly_rate": self.metrics.anomaly_count / self.metrics.total_observations,
            "buffer_utilization": self.metrics.buffer_utilization
        }

    def get_current_state_summary(self) -> Dict[str, Any]:
        """現在状態の要約"""
        if not self.is_streaming:
            return {"status": "not_streaming"}
            
        try:
            current_state, current_cov = self.estimator.get_current_state()
            
            # 不確実性指標
            uncertainty = torch.sqrt(torch.diagonal(current_cov))
            confidence = 1.0 / (1.0 + uncertainty.mean())  # 簡易信頼度
            
            return {
                "timestamp": self.last_update_time,
                "current_state": current_state.tolist(),
                "uncertainty": uncertainty.tolist(),
                "confidence_score": confidence.item(),
                "covariance_trace": torch.trace(current_cov).item(),
                "covariance_determinant": torch.det(current_cov).item()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def save_streaming_results(self, filepath: str):
        """ストリーミング結果の保存"""
        if len(self.state_buffer) == 0:
            warnings.warn("No streaming data to save")
            return
            
        # バッファからデータ抽出
        states = torch.stack(list(self.state_buffer))
        covariances = torch.stack(list(self.covariance_buffer))
        likelihoods = torch.tensor(list(self.likelihood_buffer))
        
        # 保存データ作成
        results = {
            "states": states,
            "covariances": covariances,
            "likelihoods": likelihoods,
            "metrics": self.metrics,
            "config": {
                "buffer_size": self.buffer_size,
                "anomaly_threshold": self.anomaly_threshold
            }
        }
        
        torch.save(results, filepath)
        print(f"Streaming results saved to {filepath}")

    def clear_buffers(self):
        """バッファクリア"""
        self.observation_buffer.clear()
        self.state_buffer.clear()
        self.covariance_buffer.clear()
        self.likelihood_buffer.clear()
        
        # メトリクスリセット
        self.metrics = StreamingMetrics()
        self.likelihood_mean = 0.0
        self.likelihood_std = 1.0
        self.likelihood_update_count = 0
        
        print("Buffers and metrics cleared")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.is_streaming:
            self.stop_streaming()