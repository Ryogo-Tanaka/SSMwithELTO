# src/inference/state_estimator.py
"""
統合推論クラス: StateEstimator

学習済みDFIVモデルからKalman Filtering推論エンジンを構築。
DF-A/DF-B コンポーネントとの統合、ノイズ推定、推論実行を管理。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import warnings
import yaml

from .kalman_filter import OperatorBasedKalmanFilter
from .utils import (
    estimate_noise_covariances,
    compute_residuals_from_operators,
    validate_kalman_inputs,
    initialize_state_data_driven,
    format_filter_results
)


class StateEstimator:
    """
    統合推論クラス
    
    学習済みDFIVモデル（DF-A + DF-B）から転送作用素を抽出し、
    Algorithm 1による逐次状態推定を実行。
    
    機能:
    - 学習済みモデル読み込み
    - ノイズ共分散推定
    - Kalman Filter初期化・実行
    - バッチ・オンライン推論対応
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        設定ファイルから初期化
        
        Args:
            config: 推論設定辞書
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # コンポーネント
        self.df_state_layer = None      # DF-A
        self.df_obs_layer = None        # DF-B  
        self.encoder = None             # エンコーダ
        self.kalman_filter = None       # Kalman Filter
        
        # 学習済みパラメータ
        self.V_A: Optional[torch.Tensor] = None
        self.V_B: Optional[torch.Tensor] = None
        self.U_A: Optional[torch.Tensor] = None
        self.u_B: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[Union[torch.Tensor, float]] = None
        
        # 状態
        self.is_initialized = False
        self.calibration_data: Optional[torch.Tensor] = None

    @classmethod
    def from_trained_model(
        cls,
        model_path: Union[str, Path],
        config_path: Union[str, Path]
    ) -> 'StateEstimator':
        """
        学習済みモデルから初期化
        
        V_A, V_B, φ_θ, ψ_ω, エンコーダを読み込み
        
        Args:
            model_path: 学習済みモデルパス
            config_path: 設定ファイルパス
            
        Returns:
            StateEstimator: 初期化済みインスタンス
        """
        # 設定読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        estimator = cls(config['inference'])
        estimator.load_components(model_path)
        
        return estimator

    def load_components(self, model_path: Union[str, Path]):
        """
        個別コンポーネントの読み込み
        
        DF-A, DF-B, エンコーダから演算子を抽出
        
        Args:
            model_path: 学習済みモデルパス
        """
        model_path = Path(model_path)
        
        try:
            # モデル状態読み込み
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 各コンポーネントの状態辞書取得
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # DF-A コンポーネント
            if 'df_state' in state_dict:
                self._load_df_state_component(state_dict['df_state'])
            else:
                raise KeyError("DF-A component not found in model")
                
            # DF-B コンポーネント  
            if 'df_obs' in state_dict:
                self._load_df_obs_component(state_dict['df_obs'])
            else:
                raise KeyError("DF-B component not found in model")
                
            # エンコーダ
            if 'encoder' in state_dict:
                self._load_encoder_component(state_dict['encoder'])
            else:
                raise KeyError("Encoder component not found in model")
                
            # 転送作用素抽出
            self._extract_operators()
            
            print(f"Successfully loaded components from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}")

    def _load_df_state_component(self, df_state_dict: Dict[str, Any]):
        """DF-A コンポーネント読み込み"""
        from ..ssm.df_state_layer import DFStateLayer
        
        # 設定から次元取得
        state_config = self.config.get('model', {}).get('df_state', {})
        
        self.df_state_layer = DFStateLayer(
            state_dim=state_config.get('state_dim', 5),
            feature_dim=state_config.get('feature_dim', 16),
            lambda_A=state_config.get('lambda_A', 1e-3),
            lambda_B=state_config.get('lambda_B', 1e-3)
        ).to(self.device)
        
        self.df_state_layer.load_state_dict(df_state_dict)
        self.df_state_layer.eval()

    def _load_df_obs_component(self, df_obs_dict: Dict[str, Any]):
        """DF-B コンポーネント読み込み"""
        from ..ssm.df_observation_layer import DFObservationLayer
        
        obs_config = self.config.get('model', {}).get('df_obs', {})
        
        self.df_obs_layer = DFObservationLayer(
            df_state_layer=self.df_state_layer,  # DF-A参照
            obs_feature_dim=obs_config.get('obs_feature_dim', 8),
            lambda_B=obs_config.get('lambda_B', 1e-3),
            lambda_dB=obs_config.get('lambda_dB', 1e-3)
        ).to(self.device)
        
        self.df_obs_layer.load_state_dict(df_obs_dict)
        self.df_obs_layer.eval()

    def _load_encoder_component(self, encoder_dict: Dict[str, Any]):
        """エンコーダ読み込み"""
        from ..models.architectures.tcn import tcnEncoder
        
        encoder_config = self.config.get('model', {}).get('encoder', {})
        
        self.encoder = tcnEncoder(
            input_dim=encoder_config.get('input_dim', 7),
            output_dim=1,  # スカラー特徴量
            channels=encoder_config.get('channels', 64),
            layers=encoder_config.get('layers', 6)
        ).to(self.device)
        
        self.encoder.load_state_dict(encoder_dict)
        self.encoder.eval()
        
        # 推論時は勾配なし
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _extract_operators(self):
        """
        学習済みコンポーネントから転送作用素を抽出
        
        DF-A: V_A, U_A
        DF-B: V_B, u_B
        """
        if not all([self.df_state_layer, self.df_obs_layer]):
            raise RuntimeError("DF components not loaded")
            
        # DF-A から V_A, U_A 抽出
        if hasattr(self.df_state_layer, 'V_A') and self.df_state_layer.V_A is not None:
            self.V_A = self.df_state_layer.V_A.clone().detach()
        else:
            raise RuntimeError("V_A not found in DF-A component")
            
        if hasattr(self.df_state_layer, 'U_A') and self.df_state_layer.U_A is not None:
            self.U_A = self.df_state_layer.U_A.clone().detach()
        else:
            raise RuntimeError("U_A not found in DF-A component")
            
        # DF-B から V_B, u_B 抽出
        if hasattr(self.df_obs_layer, 'V_B') and self.df_obs_layer.V_B is not None:
            self.V_B = self.df_obs_layer.V_B.clone().detach()
        else:
            raise RuntimeError("V_B not found in DF-B component")
            
        if hasattr(self.df_obs_layer, 'u_B') and self.df_obs_layer.u_B is not None:
            self.u_B = self.df_obs_layer.u_B.clone().detach()
        else:
            raise RuntimeError("u_B not found in DF-B component")
            
        print("Operators extracted successfully:")
        print(f"  V_A: {self.V_A.shape}")
        print(f"  V_B: {self.V_B.shape}")
        print(f"  U_A: {self.U_A.shape}")
        print(f"  u_B: {self.u_B.shape}")

    def estimate_noise_covariances(
        self,
        calibration_data: torch.Tensor,
        method: str = "residual_based"
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        キャリブレーション用データでQ, R推定
        
        式45-46の実装
        
        Args:
            calibration_data: キャリブレーション観測 (T_cal, n)
            method: ノイズ推定法 ("residual_based")
            
        Returns:
            Q: 状態ノイズ共分散 (dA, dA)
            R: 観測ノイズ分散
        """
        if method != "residual_based":
            raise ValueError(f"Unknown noise estimation method: {method}")
            
        self.calibration_data = calibration_data
        T_cal = calibration_data.size(0)
        
        print(f"Estimating noise covariances from {T_cal} calibration samples...")
        
        with torch.no_grad():
            # 1. エンコード: {y_t} → {m_t}
            m_series = self.encoder(calibration_data.unsqueeze(0)).squeeze()  # (T_cal,)
            
            # 2. 特徴写像適用: {m_t} → {φ_t}, {ψ_t}
            # 注意: 実際には学習済みφ_θ, ψ_ωを使用
            phi_sequence = self._apply_state_feature_mapping(m_series)      # (T_cal+1, dA)
            psi_sequence = self._apply_obs_feature_mapping(m_series)        # (T_cal+1, dB)
            
            # 3. 残差計算
            residuals_state, residuals_obs = compute_residuals_from_operators(
                phi_sequence, psi_sequence, self.V_A, self.V_B
            )
            
            # 4. 共分散推定
            regularization = self.config.get('noise_estimation', {})
            Q, R = estimate_noise_covariances(
                residuals_state, residuals_obs, regularization
            )
            
        self.Q = Q
        self.R = R
        
        print(f"Noise covariances estimated:")
        print(f"  Q condition number: {torch.linalg.cond(Q).item():.2e}")
        if isinstance(R, torch.Tensor):
            print(f"  R condition number: {torch.linalg.cond(R).item():.2e}")
        else:
            print(f"  R (scalar): {R:.6f}")
            
        return Q, R

    def _apply_state_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        状態特徴写像 φ_θ の適用
        
        注意: 簡易実装。実際は学習済みφ_θを使用する必要がある。
        
        Args:
            m_series: スカラー特徴系列 (T,)
            
        Returns:
            torch.Tensor: 状態特徴系列 (T+1, dA)
        """
        T = m_series.size(0)
        dA = self.V_A.size(0)
        
        # 簡易実装: 線形変換 + 遅延埋め込み
        features = []
        for t in range(T + 1):
            if t < T:
                # 遅延埋め込み
                delays = []
                for d in range(min(5, dA)):
                    if t - d >= 0:
                        delays.append(m_series[t - d])
                    else:
                        delays.append(torch.zeros_like(m_series[0]))
                feature = torch.stack(delays)
                
                # パディング
                if len(delays) < dA:
                    padding = torch.zeros(dA - len(delays), device=self.device)
                    feature = torch.cat([feature, padding])
                elif len(delays) > dA:
                    feature = feature[:dA]
            else:
                # 最後のサンプル（予測用）
                feature = features[-1]  # 前のサンプルをコピー
                
            features.append(feature[:dA])
            
        return torch.stack(features)  # (T+1, dA)

    def _apply_obs_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        観測特徴写像 ψ_ω の適用
        
        Args:
            m_series: スカラー特徴系列 (T,)
            
        Returns:
            torch.Tensor: 観測特徴系列 (T+1, dB)
        """
        # 実際の実装では学習済みψ_ωを使用
        T = m_series.size(0)
        dB = self.V_B.size(0)
        
        # 簡易実装
        if hasattr(self.df_obs_layer, 'psi_omega'):
            with torch.no_grad():
                psi_sequence = []
                for t in range(T + 1):
                    if t < T:
                        psi_t = self.df_obs_layer.psi_omega(m_series[t:t+1])  # (dB,)
                    else:
                        psi_t = self.df_obs_layer.psi_omega(m_series[-1:])   # 最後の値
                    psi_sequence.append(psi_t)
                return torch.stack(psi_sequence)  # (T+1, dB)
        else:
            # フォールバック: 線形変換
            features = []
            for t in range(T + 1):
                if t < T:
                    base = m_series[t]
                else:
                    base = m_series[-1]
                feature = torch.cat([
                    base.unsqueeze(0),
                    torch.randn(dB - 1, device=self.device) * 0.1
                ])
                features.append(feature)
            return torch.stack(features)

    def initialize_filtering(
        self,
        initial_data: Optional[torch.Tensor] = None,
        method: str = "data_driven"
    ):
        """
        フィルタリング開始前の初期化
        
        KalmanFilterの作成、初期状態設定
        
        Args:
            initial_data: 初期化用データ (N0, n) or None
            method: 初期化方法 ("data_driven" | "zero")
        """
        if not all([self.V_A, self.V_B, self.U_A, self.u_B]):
            raise RuntimeError("Operators not extracted. Call load_components() first.")
            
        if self.Q is None or self.R is None:
            # デフォルトノイズ設定
            warnings.warn("Noise covariances not estimated. Using defaults.")
            dA = int(self.V_A.size(0))
            self.Q = 0.01 * torch.eye(dA, device=self.device)
            self.R = 0.1
            
        # 入力検証
        validation = validate_kalman_inputs(
            self.V_A, self.V_B, self.U_A, self.u_B, self.Q, self.R
        )
        
        if not validation["valid"]:
            raise RuntimeError(f"Invalid Kalman inputs: {validation['errors']}")
            
        if validation["warnings"]:
            for warning in validation["warnings"]:
                warnings.warn(warning)
                
        # Kalman Filter作成
        self.kalman_filter = OperatorBasedKalmanFilter(
            V_A=self.V_A,
            V_B=self.V_B,
            U_A=self.U_A,
            u_B=self.u_B,
            Q=self.Q,
            R=self.R,
            encoder=self.encoder,
            device=str(self.device)
        )
        
        # 初期状態設定
        if initial_data is not None:
            self.kalman_filter.initialize_state(initial_data, method)
        elif self.calibration_data is not None:
            # キャリブレーションデータから初期化
            n_init = min(10, self.calibration_data.size(0))
            self.kalman_filter.initialize_state(
                self.calibration_data[:n_init], method
            )
        else:
            # ゼロ初期化
            self.kalman_filter.initialize_state(
                torch.zeros(1, self.encoder.input_dim, device=self.device), "zero"
            )
            
        self.is_initialized = True
        print("Kalman Filter initialized successfully")

    def filter_sequence(
        self,
        observations: torch.Tensor,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        バッチフィルタリング（従来型）
        
        観測系列全体を一度に処理
        
        Args:
            observations: 観測系列 (T, n)
            return_likelihood: 尤度も返すかどうか
            
        Returns:
            X_means: 状態平均系列 (T, r)
            X_covariances: 状態共分散系列 (T, r, r)
            likelihoods: 観測尤度系列 (T,) [optional]
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_sequence(observations, None, return_likelihood)

    def filter_online(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        オンラインフィルタリング（逐次型）
        
        1観測ずつ処理、内部状態保持
        
        Args:
            observation: 現在観測 (n,)
            
        Returns:
            x_hat: 推定状態 (r,)
            Sigma_x: 状態共分散 (r, r)  
            likelihood: 観測尤度
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_step(observation)

    def reset_state(self):
        """
        内部状態リセット（新しい系列開始時）
        """
        if self.kalman_filter is not None:
            self.kalman_filter.reset_state()

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        現在の状態推定値と信頼区間を取得
        
        Returns:
            x_hat: 現在状態推定 (r,)
            Sigma_x: 現在状態共分散 (r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self.kalman_filter.get_current_state()

    def predict_ahead(
        self,
        n_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        n期先予測
        
        現在状態から将来予測
        
        Args:
            n_steps: 予測ステップ数
            
        Returns:
            x_pred: 予測状態 (n_steps, r)
            Sigma_pred: 予測共分散 (n_steps, r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        # 現在状態取得
        mu_current, Sigma_current = self.kalman_filter.mu, self.kalman_filter.Sigma
        
        # 逐次予測
        predictions = []
        covariances = []
        
        mu, Sigma = mu_current.clone(), Sigma_current.clone()
        
        for step in range(n_steps):
            # 時間更新のみ（観測なし）
            mu, Sigma = self.kalman_filter.predict_step(mu, Sigma)
            
            # 元状態空間での復元
            x_pred, Sigma_x_pred = self.kalman_filter._recover_original_state(mu, Sigma)
            
            predictions.append(x_pred)
            covariances.append(Sigma_x_pred)
            
        return torch.stack(predictions), torch.stack(covariances)

    def get_filter_diagnostics(self) -> Dict[str, Any]:
        """
        フィルタ診断情報取得
        
        Returns:
            Dict: 診断結果
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        diagnostics = {
            "initialization_status": self.is_initialized,
            "operator_shapes": {
                "V_A": self.V_A.shape,
                "V_B": self.V_B.shape,
                "U_A": self.U_A.shape,
                "u_B": self.u_B.shape
            },
            "numerical_stability": self.kalman_filter.check_numerical_stability()
        }
        
        if self.Q is not None:
            diagnostics["noise_covariances"] = {
                "Q_condition": torch.linalg.cond(self.Q).item(),
                "Q_trace": torch.trace(self.Q).item()
            }
            
        if isinstance(self.R, torch.Tensor):
            diagnostics["noise_covariances"]["R_condition"] = torch.linalg.cond(self.R).item()
        else:
            diagnostics["noise_covariances"]["R_scalar"] = self.R
            
        return diagnostics

    def export_for_deployment(self, export_path: Union[str, Path]):
        """
        デプロイ用ファイル出力
        
        推論のみ必要なコンポーネント抽出
        
        Args:
            export_path: 出力パス
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Cannot export.")
            
        # 推論用パラメータ
        inference_params = {
            "operators": {
                "V_A": self.V_A.cpu(),
                "V_B": self.V_B.cpu(),
                "U_A": self.U_A.cpu(),
                "u_B": self.u_B.cpu()
            },
            "noise_covariances": {
                "Q": self.Q.cpu(),
                "R": self.R if isinstance(self.R, (int, float)) else self.R.cpu()
            },
            "config": self.config
        }
        
        # エンコーダ状態
        encoder_state = self.encoder.state_dict()
        
        # 保存
        torch.save(inference_params, export_path / "inference_params.pth")
        torch.save(encoder_state, export_path / "encoder.pth")
        
        with open(export_path / "inference_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        print(f"Inference components exported to {export_path}")