# src/models/inference_model.py
"""
統合インターフェース: InferenceModel
学習済みモデルと推論設定から初期化し、バッチ・ストリーミング推論を統一的に提供
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union, Generator
from pathlib import Path
import yaml
import json
import warnings
from dataclasses import dataclass

from ..inference.state_estimator import StateEstimator
from ..inference.streaming import StreamingEstimator
from ..inference.utils import format_filter_results


@dataclass
class InferenceConfig:
    """推論設定"""
    device: str = 'auto'
    noise_estimation_method: str = 'residual_based'
    gamma_Q: float = 1e-6
    gamma_R: float = 1e-6
    use_calibration: bool = True
    initialization_method: str = 'data_driven'
    n_init_samples: int = 50
    buffer_size: int = 100
    batch_processing: bool = False
    anomaly_detection: bool = True
    anomaly_threshold: float = 3.0
    condition_threshold: float = 1e12
    min_eigenvalue: float = 1e-8
    jitter: float = 1e-6
    save_states: bool = True
    save_covariances: bool = False
    save_likelihoods: bool = True


class InferenceModel:
    """
    学習済みモデルと推論設定から初期化される統合推論システム
    バッチ推論・ストリーミング推論両対応
    """

    def __init__(
        self,
        trained_model_path: Union[str, Path],
        inference_config: Union[Dict[str, Any], InferenceConfig, str, Path]
    ):
        """
        Args:
            trained_model_path: 学習済みモデルパス
            inference_config: 推論設定 (辞書/InferenceConfig/YAMLパス)
        """
        self.model_path = Path(trained_model_path)

        # 設定解析 (複数ドキュメント対応)
        if isinstance(inference_config, (str, Path)):
            with open(inference_config, 'r') as f:
                documents = list(yaml.safe_load_all(f))
                config_dict = documents[0] if documents else {}
            inference_settings = config_dict.get('inference', {})
            if not inference_settings:
                print("'inference'セクションが見つかりません。デフォルト設定を使用します。")
            self.config = InferenceConfig(**inference_settings)
        elif isinstance(inference_config, dict):
            self.config = InferenceConfig(**inference_config)
        elif isinstance(inference_config, InferenceConfig):
            self.config = inference_config
        else:
            raise ValueError("Invalid inference_config type")

        # デバイス設定
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        print(f"InferenceModel initialized on device: {self.device}")

        # コンポーネント
        self.state_estimator: Optional[StateEstimator] = None
        self.streaming_estimator: Optional[StreamingEstimator] = None

        # 状態
        self.is_setup = False
        self.calibration_data: Optional[torch.Tensor] = None

    def setup_inference(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        config_override: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None
    ):
        """
        推論環境セットアップ (ノイズ推定、初期化実行)
        Args:
            calibration_data: キャリブレーション用データ (T_cal, n)
            config_override: 設定上書き辞書
            method: 初期化方法動的オーバーライド
        """
        print("Setting up inference environment...")

        # 設定上書き
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        if method is not None:
            self.config.initialization_method = method

        # StateEstimator作成
        estimator_config = {
            'device': str(self.device),
            'model': {
                'df_state': {'state_dim': 5, 'feature_dim': 16},
                'df_obs': {'obs_feature_dim': 8},
                'encoder': {'input_dim': 7}
            },
            'noise_estimation': {
                'method': self.config.noise_estimation_method,
                'gamma_Q': self.config.gamma_Q,
                'gamma_R': self.config.gamma_R
            },
            'initialization': {
                'method': self.config.initialization_method,
                'n_init_samples': self.config.n_init_samples
            },
            'numerical': {
                'condition_threshold': self.config.condition_threshold,
                'min_eigenvalue': self.config.min_eigenvalue,
                'jitter': self.config.jitter
            }
        }

        self.state_estimator = StateEstimator(estimator_config)
        self._load_trained_model()

        # ノイズ共分散推定
        if calibration_data is not None and self.config.use_calibration:
            self.calibration_data = calibration_data.to(self.device)
            self.state_estimator.estimate_noise_covariances(
                self.calibration_data, self.config.noise_estimation_method
            )
        elif calibration_data is not None:
            self.calibration_data = calibration_data.to(self.device)

        # フィルタリング初期化
        self.state_estimator.initialize_filtering(self.calibration_data)

        # StreamingEstimator作成
        if not self.config.batch_processing:
            self.streaming_estimator = StreamingEstimator(
                state_estimator=self.state_estimator,
                buffer_size=self.config.buffer_size,
                anomaly_threshold=self.config.anomaly_threshold,
                enable_metrics=True
            )

        self.is_setup = True
        print("Inference environment setup completed")

    def _load_trained_model(self):
        """学習済みモデル読込"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'config' in checkpoint:
                train_config = checkpoint['config']
                self._update_dimensions_from_training_config(train_config)
            self.state_estimator.load_components(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")

    def _update_dimensions_from_training_config(self, train_config: Dict[str, Any]):
        """学習時設定から次元情報更新"""
        # DF-A設定
        if 'ssm' in train_config and 'df_state' in train_config['ssm']:
            df_state_config = train_config['ssm']['df_state']
            if hasattr(self.state_estimator, 'config'):
                df_state_update = {
                    'feature_dim': df_state_config.get('feature_dim', 16),
                    'state_dim': train_config['ssm']['realization'].get('rank', 5)
                }
                if 'feature_net' in df_state_config:
                    df_state_update['feature_net'] = df_state_config['feature_net']
                self.state_estimator.config['model']['df_state'].update(df_state_update)

        # DF-B設定
        if 'ssm' in train_config and 'df_observation' in train_config['ssm']:
            df_obs_config = train_config['ssm']['df_observation']
            if hasattr(self.state_estimator, 'config'):
                df_obs_update = {
                    'obs_feature_dim': df_obs_config.get('obs_feature_dim', 8)
                }
                if 'obs_net' in df_obs_config:
                    df_obs_update['obs_net'] = df_obs_config['obs_net']
                self.state_estimator.config['model']['df_obs'].update(df_obs_update)

        # エンコーダ設定
        if 'model' in train_config and 'encoder' in train_config['model']:
            encoder_config = train_config['model']['encoder']
            if hasattr(self.state_estimator, 'config'):
                self.state_estimator.config['model']['encoder'].update({
                    'input_dim': encoder_config.get('input_dim', 7)
                })

    def inference_batch(
        self,
        observations: torch.Tensor,
        return_format: str = 'dict'
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        バッチ推論
        Args:
            observations: 観測系列 (T, n)
            return_format: 返却形式 ('tuple' | 'dict')
        Returns:
            推論結果
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        observations = observations.to(self.device)
        T, n = observations.shape
        print(f"Performing batch inference on {T} observations...")

        # バッチフィルタリング実行
        if self.config.save_likelihoods:
            X_means, X_covariances, likelihoods = self.state_estimator.filter_sequence(
                observations, return_likelihood=True
            )
        else:
            X_means, X_covariances = self.state_estimator.filter_sequence(
                observations, return_likelihood=False
            )
            likelihoods = None

        # 返却形式に応じて整形
        if return_format == 'tuple':
            if likelihoods is not None:
                return X_means, X_covariances, likelihoods
            return X_means, X_covariances

        elif return_format == 'dict':
            results = format_filter_results(X_means, X_covariances, likelihoods)
            results['inference_info'] = {
                'sequence_length': T,
                'observation_dimension': n,
                'model_path': str(self.model_path),
                'device': str(self.device),
                'config': self.config.__dict__
            }
            return results
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def inference_streaming(
        self,
        initial_data: Optional[torch.Tensor] = None
    ) -> StreamingEstimator:
        """
        ストリーミング推論ジェネレータ
        Args:
            initial_data: 初期化用データ (N0, n)
        Returns:
            StreamingEstimator インスタンス
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        if self.streaming_estimator is None:
            raise RuntimeError("Streaming estimator not available. Set batch_processing=False.")

        # ストリーミング開始
        if initial_data is not None:
            initial_data = initial_data.to(self.device)
            self.streaming_estimator.start_streaming(initial_data)
        else:
            self.streaming_estimator.start_streaming(self.calibration_data)

        print("Streaming inference started. Use returned estimator to process data.")
        return self.streaming_estimator

    def predict_future(
        self,
        n_steps: int,
        current_observations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        将来予測
        Args:
            n_steps: 予測ステップ数
            current_observations: 現在観測 (状態更新用)
        Returns:
            予測結果辞書
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        # 状態更新
        if current_observations is not None:
            current_observations = current_observations.to(self.device)
            if current_observations.dim() == 1:
                self.state_estimator.filter_online(current_observations)
            else:
                for obs in current_observations:
                    self.state_estimator.filter_online(obs)

        # 予測実行
        predictions = self.state_estimator.predict_ahead(n_steps)

        return {
            'predictions': predictions[0],
            'covariances': predictions[1],
            'prediction_steps': n_steps,
            'prediction_time': torch.arange(1, n_steps + 1, dtype=torch.float32)
        }

    def evaluate_model(
        self,
        test_data: torch.Tensor,
        metrics: List[str] = ['mse', 'likelihood', 'coverage']
    ) -> Dict[str, Any]:
        """
        モデル評価
        Args:
            test_data: テストデータ (T, n)
            metrics: 評価指標リスト
        Returns:
            評価結果辞書
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        test_data = test_data.to(self.device)
        T = test_data.size(0)
        print(f"Evaluating model on {T} test samples...")

        # 推論実行
        results = self.inference_batch(test_data, return_format='dict')
        X_means = torch.tensor(results['summary']['mean_trajectory'])
        X_covariances = torch.tensor(results['summary']['covariance_trajectory'])

        if 'likelihood' in results['statistics']:
            likelihoods = torch.tensor(results['statistics']['likelihood']['likelihood_trajectory'])
        else:
            likelihoods = None

        evaluation = {'test_length': T, 'metrics': {}}

        # MSE
        if 'mse' in metrics:
            pred_errors = []
            for t in range(1, T):
                x_pred = X_means[t-1]
                x_actual = X_means[t]
                pred_errors.append(torch.norm(x_pred - x_actual).item())
            evaluation['metrics']['mse'] = {
                'mean_prediction_error': np.mean(pred_errors),
                'std_prediction_error': np.std(pred_errors)
            }

        # 尤度評価
        if 'likelihood' in metrics and likelihoods is not None:
            evaluation['metrics']['likelihood'] = {
                'total_log_likelihood': torch.sum(likelihoods).item(),
                'mean_log_likelihood': torch.mean(likelihoods).item(),
                'likelihood_std': torch.std(likelihoods).item()
            }

        # カバレッジ
        if 'coverage' in metrics:
            std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))
            lower_bounds = X_means - 1.96 * std_devs
            upper_bounds = X_means + 1.96 * std_devs
            coverage_count = 0
            for t in range(T):
                if torch.all(X_means[t] >= lower_bounds[t]) and torch.all(X_means[t] <= upper_bounds[t]):
                    coverage_count += 1
            evaluation['metrics']['coverage'] = {
                'coverage_rate': coverage_count / T,
                'expected_coverage': 0.95,
                'coverage_difference': coverage_count / T - 0.95
            }

        return evaluation

    def export_for_deployment(
        self,
        export_path: Union[str, Path],
        include_calibration: bool = True
    ):
        """
        デプロイ用ファイル出力
        Args:
            export_path: 出力パス
            include_calibration: キャリブレーションデータ含めるか
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # StateEstimatorエクスポート
        self.state_estimator.export_for_deployment(export_path)

        # 推論設定保存
        config_dict = {
            'inference_config': self.config.__dict__,
            'model_info': {
                'original_model_path': str(self.model_path),
                'export_timestamp': torch.tensor(time.time()).item(),
                'device_used': str(self.device)
            }
        }

        with open(export_path / 'deployment_config.yaml', 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # キャリブレーションデータ
        if include_calibration and self.calibration_data is not None:
            torch.save(self.calibration_data.cpu(), export_path / 'calibration_data.pth')

        # デプロイ用スクリプト生成
        self._generate_deployment_script(export_path)
        print(f"Deployment package created at {export_path}")

    def _generate_deployment_script(self, export_path: Path):
        """デプロイ用スクリプト生成"""
        script_content = '''#!/usr/bin/env python3
"""Auto-generated deployment script for DFIV Kalman Filter inference"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from src.inference.state_estimator import StateEstimator
from src.inference.kalman_filter import OperatorBasedKalmanFilter

def load_inference_model(model_dir):
    """Load exported inference model"""
    model_dir = Path(model_dir)
    params = torch.load(model_dir / 'inference_params.pth', map_location='cpu')
    encoder_state = torch.load(model_dir / 'encoder.pth', map_location='cpu')
    with open(model_dir / 'deployment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    estimator = StateEstimator(config['inference_config'])
    return estimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Exported model directory')
    parser.add_argument('--input', required=True, help='Input data file (.npz)')
    parser.add_argument('--output', required=True, help='Output results file (.json)')
    args = parser.parse_args()

    estimator = load_inference_model(args.model_dir)
    data = np.load(args.input)
    observations = torch.from_numpy(data['observations']).float()
    results = estimator.filter_sequence(observations)
    print(f"Inference completed. Results saved to {args.output}")

if __name__ == '__main__':
    main()
'''
        with open(export_path / 'deploy_inference.py', 'w') as f:
            f.write(script_content)

        try:
            import os
            os.chmod(export_path / 'deploy_inference.py', 0o755)
        except:
            pass

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """システム診断情報"""
        diagnostics = {
            'setup_status': self.is_setup,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'config': self.config.__dict__
        }

        if self.is_setup and self.state_estimator:
            diagnostics['filter_diagnostics'] = self.state_estimator.get_filter_diagnostics()

        if self.streaming_estimator:
            diagnostics['streaming_metrics'] = self.streaming_estimator.get_performance_metrics()

        return diagnostics

    def filter_sequence(
        self,
        observations: torch.Tensor,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        系列フィルタリング (評価スクリプト用インターフェース)
        Note: inference_batch()と戻り値形式が異なる
        Args:
            observations: 観測系列 (T, d)
            return_likelihood: 尤度も返すか
        Returns:
            (X_means, X_covariances) or (X_means, X_covariances, likelihoods)
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
        result = self.state_estimator.filter_sequence(observations, return_likelihood=return_likelihood)
        return result

    def filter_online(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        オンライン単一ステップフィルタリング (評価スクリプト用)
        Args:
            observation: 単一観測値 (d,)
        Returns:
            (x_hat, Sigma_x, likelihood)
        """
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")
        result = self.state_estimator.filter_online(observation)
        return result

    def reset_state(self):
        """フィルタ状態リセット (オンライン処理用)"""
        if not self.is_setup:
            raise RuntimeError("Inference not setup. Call setup_inference() first.")

        if hasattr(self.state_estimator, 'reset_filter_state'):
            self.state_estimator.reset_filter_state()

        if self.streaming_estimator and hasattr(self.streaming_estimator, 'reset'):
            self.streaming_estimator.reset()

    def __repr__(self) -> str:
        status = "setup" if self.is_setup else "not setup"
        return f"InferenceModel(model={self.model_path.name}, device={self.device}, status={status})"
