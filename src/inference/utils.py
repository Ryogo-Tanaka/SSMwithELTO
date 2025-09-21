# src/inference/utils.py
"""
推論用ユーティリティ関数

式45-46のノイズ共分散推定、数値安定性チェック、
フィルタ性能評価などの支援機能を提供。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings


def estimate_noise_covariances(
    residuals_state: torch.Tensor,
    residuals_obs: torch.Tensor,
    regularization: Dict[str, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    式45-46のQ, R推定
    
    残差系列から状態・観測ノイズ共分散を推定。
    正定値性保証付き。
    
    Args:
        residuals_state: 状態残差 ε_t := φ_{t+1} - V_A φ_t ∈ R^dA (T, dA)
        residuals_obs: 観測残差 ρ_t := ψ_t - V_B φ_t ∈ R^dB (T, dB)  
        regularization: 正則化パラメータ {"gamma_Q": float, "gamma_R": float}
        
    Returns:
        Q: 状態ノイズ共分散 (dA, dA)
        R: 観測ノイズ共分散 (dB, dB) または スカラー
    """
    T_state, dA = residuals_state.shape
    T_obs, dB = residuals_obs.shape
    
    # 明示的な型変換
    dA = int(dA)
    dB = int(dB)
    gamma_Q = regularization.get("gamma_Q", 1e-6)
    gamma_R = regularization.get("gamma_R", 1e-6)
    
    # 式45: Q = (1/T) Σ_{t=0}^{T-1} ε_t ε_t^T + γ_Q I_{dA}
    Q = torch.mean(
        torch.einsum('ti,tj->tij', residuals_state, residuals_state), 
        dim=0
    )  # (dA, dA)
    Q += gamma_Q * torch.eye(dA, device=residuals_state.device)
    
    # 式46: R = (1/(T+1)) Σ_{t=0}^T ρ_t ρ_t^T + γ_R I_{dB}
    R = torch.mean(
        torch.einsum('ti,tj->tij', residuals_obs, residuals_obs),
        dim=0
    )  # (dB, dB)
    R += gamma_R * torch.eye(dB, device=residuals_obs.device)
    
    # 正定値性確保
    Q = regularize_covariance(Q)
    R = regularize_covariance(R)

    # R は常にテンソル形式を保持（多変量観測対応）
    # スカラー観測の場合でも (1,1) 行列として扱う

    return Q, R


def compute_residuals_from_operators(
    phi_sequence: torch.Tensor,
    psi_sequence: torch.Tensor,
    V_A: torch.Tensor,
    V_B: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    転送作用素を用いた残差計算
    
    学習済み V_A, V_B から状態・観測残差を計算。
    ノイズ共分散推定の前処理として使用。
    
    Args:
        phi_sequence: 状態特徴系列 φ_t (T+1, dA)
        psi_sequence: 観測特徴系列 ψ_t (T+1, dB)
        V_A: 状態転送作用素 (dA, dA)
        V_B: 観測転送作用素 (dB, dA)
        
    Returns:
        residuals_state: 状態残差 ε_t (T, dA)
        residuals_obs: 観測残差 ρ_t (T+1, dB)
    """
    T = phi_sequence.size(0) - 1  # T+1 → T
    
    # 状態残差: ε_t := φ_{t+1} - V_A φ_t
    phi_pred = phi_sequence[:-1] @ V_A.T  # (T, dA)
    residuals_state = phi_sequence[1:] - phi_pred  # (T, dA)
    
    # 観測残差: ρ_t := ψ_t - V_B φ_t  
    psi_pred = phi_sequence @ V_B.T  # (T+1, dB)
    residuals_obs = psi_sequence - psi_pred  # (T+1, dB)
    
    return residuals_state, residuals_obs


def compute_innovation_residuals(
    predictions: torch.Tensor,
    observations: torch.Tensor
) -> torch.Tensor:
    """
    イノベーション残差計算
    
    フィルタ性能評価用。予測と観測の差分を計算。
    
    Args:
        predictions: 予測値系列 (T,)
        observations: 観測値系列 (T,)
        
    Returns:
        torch.Tensor: イノベーション残差 (T,)
    """
    return observations - predictions


def check_numerical_stability(
    matrix: torch.Tensor,
    name: str = "Matrix",
    condition_threshold: float = 1e12,
    min_eigenvalue: float = 1e-8
) -> Dict[str, Any]:
    """
    行列の数値的安定性チェック
    
    条件数、固有値確認により数値計算の安定性を診断。
    
    Args:
        matrix: チェック対象行列 (d, d)
        name: 行列名（ログ用）
        condition_threshold: 条件数閾値
        min_eigenvalue: 最小固有値閾値
        
    Returns:
        Dict: 診断結果
    """
    try:
        # 条件数計算
        condition_number = torch.linalg.cond(matrix).item()
        
        # 固有値計算
        eigenvalues = torch.linalg.eigvals(matrix).real
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        
        # 対称性チェック
        is_symmetric = torch.allclose(matrix, matrix.T, atol=1e-6)
        
        # 正定値性チェック
        is_positive_definite = min_eig > 0
        
        # 安定性判定
        is_stable = (
            condition_number < condition_threshold and
            min_eig > min_eigenvalue and
            is_symmetric and
            is_positive_definite
        )
        
        return {
            "matrix_name": name,
            "shape": matrix.shape,
            "condition_number": condition_number,
            "eigenvalues": {
                "min": min_eig,
                "max": max_eig,
                "range": max_eig - min_eig
            },
            "properties": {
                "symmetric": is_symmetric,
                "positive_definite": is_positive_definite
            },
            "stability": {
                "is_stable": is_stable,
                "condition_ok": condition_number < condition_threshold,
                "eigenvals_ok": min_eig > min_eigenvalue
            }
        }
        
    except Exception as e:
        return {
            "matrix_name": name,
            "error": str(e),
            "is_stable": False
        }


def regularize_covariance(
    cov_matrix: torch.Tensor,
    min_eigenvalue: float = 1e-8,
    jitter: float = 1e-6
) -> torch.Tensor:
    """
    共分散行列の正則化
    
    数値安定性確保のため正定値性を保証。
    
    Args:
        cov_matrix: 共分散行列 (d, d)
        min_eigenvalue: 最小固有値
        jitter: ジッター項
        
    Returns:
        torch.Tensor: 正則化済み共分散行列 (d, d)
    """
    # 対称性確保
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
    try:
        # 固有値分解
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # 負の固有値をクリップ
        eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
        
        # 再構成
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        
    except Exception:
        # 失敗時はジッター追加
        warnings.warn(f"Eigendecomposition failed, adding jitter: {jitter}")
        cov_matrix += jitter * torch.eye(cov_matrix.size(0), device=cov_matrix.device)
        
    return cov_matrix


def compute_log_likelihood(
    innovations: torch.Tensor,
    innovation_covariances: torch.Tensor
) -> torch.Tensor:
    """
    対数尤度計算
    
    モデル評価・比較用。イノベーション系列から尤度を計算。
    
    Args:
        innovations: イノベーション系列 (T,)
        innovation_covariances: イノベーション共分散系列 (T,)
        
    Returns:
        torch.Tensor: 対数尤度系列 (T,)
    """
    # 正規分布の対数確率密度
    log_likelihoods = -0.5 * (
        torch.log(2 * torch.pi * innovation_covariances) +
        (innovations ** 2) / innovation_covariances
    )
    
    return log_likelihoods


def initialize_state_data_driven(
    feature_samples: torch.Tensor,
    method: str = "empirical"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    データ駆動状態初期化
    
    式47-48の実装。初期観測サンプルから µ₀, Σ₀ を推定。
    
    Args:
        feature_samples: 特徴サンプル (N0, dA)
        method: 初期化方法 ("empirical" | "robust")
        
    Returns:
        mu_0: 初期状態平均 (dA,)
        Sigma_0: 初期状態共分散 (dA, dA)
    """
    N0, dA = feature_samples.shape
    
    if method == "empirical":
        # 式47: µ₀ = (1/N₀) Σ_{i=1}^{N₀} φ_θ(x₀^{(i)})
        mu_0 = torch.mean(feature_samples, dim=0)  # (dA,)
        
        # 式48: Σ₀ = (1/(N₀-1)) Σ_{i=1}^{N₀} (φ_θ(x₀^{(i)}) - µ₀)(φ_θ(x₀^{(i)}) - µ₀)^T
        centered = feature_samples - mu_0.unsqueeze(0)  # (N0, dA)
        Sigma_0 = (centered.T @ centered) / (N0 - 1)  # (dA, dA)
        
    elif method == "robust":
        # ロバスト推定（中央値ベース）
        mu_0 = torch.median(feature_samples, dim=0)[0]  # (dA,)
        
        # MAD (Median Absolute Deviation) ベース共分散
        centered = feature_samples - mu_0.unsqueeze(0)
        mad = torch.median(torch.abs(centered), dim=0)[0]  # (dA,)
        Sigma_0 = torch.diag(mad ** 2)  # (dA, dA)
        
    else:
        raise ValueError(f"Unknown initialization method: {method}")
        
    # 正定値性確保
    Sigma_0 = regularize_covariance(Sigma_0)
    
    return mu_0, Sigma_0


def validate_kalman_inputs(
    V_A: torch.Tensor,
    V_B: torch.Tensor,
    U_A: torch.Tensor,
    u_B: torch.Tensor,
    Q: torch.Tensor,
    R: Union[torch.Tensor, float]
) -> Dict[str, Any]:
    """
    Kalman Filter入力パラメータの検証
    
    次元整合性、数値的性質をチェック。
    
    Args:
        V_A: 状態転送作用素 (dA, dA)
        V_B: 観測転送作用素 (dB, dA)
        U_A: 状態読み出し行列 (dA, r)
        u_B: 観測読み出しベクトル (dB,)
        Q: 状態ノイズ共分散 (dA, dA)
        R: 観測ノイズ分散
        
    Returns:
        Dict: 検証結果
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "dimension_check": {},
        "numerical_check": {}
    }
    
    try:
        # 次元チェック
        dA_A, dA_A2 = V_A.shape
        dB, dA_B = V_B.shape
        dA_U, r = U_A.shape
        dB_u = u_B.shape[0]
        
        validation_results["dimension_check"] = {
            "V_A_square": dA_A == dA_A2,
            "dimension_consistency": dA_A == dA_B == dA_U,
            "observation_consistency": dB == dB_u,
            "dimensions": {
                "dA": dA_A,
                "dB": dB,
                "r": r
            }
        }
        
        # 次元整合性チェック
        if dA_A != dA_A2:
            validation_results["errors"].append("V_A is not square")
            validation_results["valid"] = False
            
        if not (dA_A == dA_B == dA_U):
            validation_results["errors"].append("Feature dimension mismatch")
            validation_results["valid"] = False
            
        if dB != dB_u:
            validation_results["errors"].append("Observation dimension mismatch")
            validation_results["valid"] = False
            
        # 数値的性質チェック
        if validation_results["valid"]:
            # Q の正定値性
            Q_check = check_numerical_stability(Q, "Q")
            validation_results["numerical_check"]["Q"] = Q_check
            if not Q_check.get("stability", {}).get("is_stable", False):
                validation_results["warnings"].append("Q matrix numerically unstable")
                
            # V_A の安定性（固有値が単位円内）
            try:
                eigenvals_A = torch.linalg.eigvals(V_A)
                max_eigenval_A = torch.abs(eigenvals_A).max().item()
                
                validation_results["numerical_check"]["V_A"] = {
                    "max_eigenvalue_magnitude": max_eigenval_A,
                    "stable": max_eigenval_A <= 1.0
                }
                
                if max_eigenval_A > 1.0:
                    validation_results["warnings"].append(
                        f"V_A may be unstable (max eigenvalue: {max_eigenval_A:.3f})"
                    )
                    
            except Exception as e:
                validation_results["warnings"].append(f"V_A eigenvalue check failed: {e}")
                
            # R の正定値性
            if isinstance(R, torch.Tensor):
                if R.dim() == 0:  # scalar
                    R_positive = R.item() > 0
                elif R.dim() == 2:  # matrix
                    R_check = check_numerical_stability(R, "R")
                    validation_results["numerical_check"]["R"] = R_check
                    R_positive = R_check.get("stability", {}).get("is_stable", False)
                else:
                    R_positive = False
            else:  # float/int
                R_positive = R > 0
                
            validation_results["numerical_check"]["R_positive"] = R_positive
            if not R_positive:
                validation_results["errors"].append("R is not positive")
                validation_results["valid"] = False
                
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation failed: {e}")
        
    return validation_results


def create_test_operators(
    dA: int = 10,
    dB: int = 5,
    r: int = 3,
    m: int = 8,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, float]]:
    """
    テスト用演算子生成（多変量対応）

    Kalman Filter のテスト・デバッグ用に安定な演算子を生成。

    Args:
        dA: 特徴空間状態次元
        dB: 特徴空間観測次元
        r: 元状態次元
        m: 多変量特徴量次元
        device: 計算デバイス

    Returns:
        V_A: 状態転送作用素 (dA, dA)
        V_B: 観測転送作用素 (dB, dA)
        U_A: 状態読み出し行列 (dA, r)
        U_B: 観測読み出し行列 (dB, m)
        Q: 状態ノイズ共分散 (dA, dA)
        R: 観測ノイズ共分散 (m, m) or scalar
    """
    device = torch.device(device)
    
    # 安定なV_A生成（固有値を単位円内に制限）
    V_A = torch.randn(dA, dA, device=device) * 0.1
    eigenvals, eigenvecs = torch.linalg.eig(V_A)
    eigenvals = eigenvals * 0.9 / torch.abs(eigenvals).max()  # 最大固有値を0.9に制限
    V_A = torch.real(eigenvecs @ torch.diag(eigenvals) @ torch.linalg.inv(eigenvecs))
    
    # V_B: ランダム行列
    V_B = torch.randn(dB, dA, device=device) * 0.5
    
    # U_A: ランダム読み出し行列
    U_A = torch.randn(dA, r, device=device) * 0.7
    
    # U_B: ランダム読み出し行列（多変量対応）
    U_B = torch.randn(dB, m, device=device) * 0.5

    # Q: 正定値共分散行列
    Q_half = torch.randn(dA, dA, device=device) * 0.1
    Q = Q_half @ Q_half.T + 0.01 * torch.eye(dA, device=device)

    # R: 観測ノイズ共分散（多変量対応）
    if m == 1:
        R = 0.1  # スカラー（後方互換性）
    else:
        R_half = torch.randn(m, m, device=device) * 0.05
        R = R_half @ R_half.T + 0.01 * torch.eye(m, device=device)

    return V_A, V_B, U_A, U_B, Q, R


def format_filter_results(
    X_means: torch.Tensor,
    X_covariances: torch.Tensor,
    likelihoods: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    フィルタ結果の整形
    
    結果の可視化・保存用フォーマット。
    
    Args:
        X_means: 状態平均系列 (T, r)
        X_covariances: 状態共分散系列 (T, r, r)
        likelihoods: 観測尤度系列 (T,) [optional]
        
    Returns:
        Dict: 整形済み結果
    """
    T, r = X_means.shape
    
    # 基本統計
    results = {
        "summary": {
            "sequence_length": T,
            "state_dimension": r,
            "mean_trajectory": X_means.cpu().numpy(),
            "covariance_trajectory": X_covariances.cpu().numpy()
        },
        "statistics": {
            "state_means": {
                "temporal_mean": torch.mean(X_means, dim=0).cpu().numpy(),
                "temporal_std": torch.std(X_means, dim=0).cpu().numpy()
            },
            "uncertainty": {
                "mean_trace": torch.mean(torch.diagonal(X_covariances, dim1=1, dim2=2), dim=0).cpu().numpy(),
                "mean_determinant": torch.mean(torch.det(X_covariances)).item()
            }
        }
    }
    
    # 尤度統計
    if likelihoods is not None:
        results["statistics"]["likelihood"] = {
            "total_log_likelihood": torch.sum(likelihoods).item(),
            "mean_log_likelihood": torch.mean(likelihoods).item(),
            "likelihood_trajectory": likelihoods.cpu().numpy()
        }
        
    # 信頼区間（±2σ）
    std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))  # (T, r)
    results["confidence_intervals"] = {
        "lower_2sigma": (X_means - 2 * std_devs).cpu().numpy(),
        "upper_2sigma": (X_means + 2 * std_devs).cpu().numpy()
    }
    
    return results


# 多変量対応の新しいユーティリティ関数

def create_multivariate_kalman_filter(
    df_state_layer,
    df_obs_layer,
    encoder,
    initial_R: Union[float, torch.Tensor] = 1e-2,
    Q_regularization: float = 1e-3,
    device: str = 'cpu'
):
    """
    多変量対応Kalmanフィルターのファクトリ関数

    Args:
        df_state_layer: 学習済みDF-A層
        df_obs_layer: 学習済みDF-B層
        encoder: エンコーダー
        initial_R: 初期観測ノイズ分散
        Q_regularization: 状態ノイズ正則化
        device: 計算デバイス

    Returns:
        OperatorBasedKalmanFilter: 設定済みフィルター
    """
    from .kalman_filter import OperatorBasedKalmanFilter

    # 学習済みパラメータを取得
    V_A = df_state_layer.get_state_operator()  # (d_A, d_A)
    U_A = df_state_layer.get_readout_matrix()  # (d_A, r)
    V_B = df_obs_layer.get_observation_operator()  # (d_B, d_A)
    U_B = df_obs_layer.get_readout_matrix()  # (d_B, m)

    # 次元取得
    d_A = V_A.size(0)
    d_B = V_B.size(0)

    # 状態ノイズ共分散（対角行列）
    Q = Q_regularization * torch.eye(d_A, device=device)

    # 観測ノイズ共分散（多変量対応）
    if isinstance(initial_R, (int, float)):
        R = initial_R * torch.eye(d_B, device=device)
    else:
        R = initial_R.to(device)

    # Kalmanフィルター初期化
    kalman_filter = OperatorBasedKalmanFilter(
        V_A=V_A,
        V_B=V_B,
        U_A=U_A,
        U_B=U_B,
        Q=Q,
        R=R,
        encoder=encoder,
        df_obs_layer=df_obs_layer,
        device=device
    )

    return kalman_filter


def estimate_multivariate_observation_noise_covariance(
    kalman_filter,
    observations: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    regularization: float = 1e-3
) -> torch.Tensor:
    """
    観測系列から多変量観測ノイズ共分散を推定

    Args:
        kalman_filter: Kalmanフィルター（学習済み）
        observations: 観測系列 (T, n)
        states: 状態系列 (T, r) - Noneの場合はフィルタリングで推定
        regularization: 正則化パラメータ

    Returns:
        R_estimated: 推定観測ノイズ共分散 (d_B, d_B)
    """
    from .noise_covariance import ObservationNoiseCovarianceEstimator

    T = observations.size(0)

    # 状態系列の取得または推定
    if states is None:
        # Kalmanフィルタリングで状態を推定
        with torch.no_grad():
            # 初期化
            kalman_filter.initialize_state(observations[:min(10, T)])

            # フィルタリング実行
            X_means, _, _ = kalman_filter.filter_sequence(observations)
            states = X_means  # (T, r)

    # 観測特徴量の計算
    with torch.no_grad():
        # エンコーダーで多変量特徴量を取得
        m_features = []
        for t in range(T):
            obs_t = observations[t].unsqueeze(0).unsqueeze(0)  # (1, 1, n)
            m_t = kalman_filter.encoder(obs_t).squeeze()  # (m,)
            m_features.append(m_t)
        m_features = torch.stack(m_features)  # (T, m)

        # 観測特徴量の生成
        psi_obs = []
        for t in range(T):
            psi_t = kalman_filter._generate_obs_features_from_multivariate(m_features[t])
            psi_obs.append(psi_t)
        psi_obs = torch.stack(psi_obs)  # (T, d_B)

        # 予測特徴量の計算
        psi_pred = []
        for t in range(T):
            if t == 0:
                # 初期予測
                x_hat_prev = states[0]  # 初期状態を使用
            else:
                x_hat_prev = states[t-1]

            # 状態特徴量
            phi_prev = kalman_filter.df_obs_layer.phi_theta(x_hat_prev.unsqueeze(0)).squeeze()  # (d_A,)

            # 観測予測
            psi_pred_t = kalman_filter.V_B @ phi_prev  # (d_B,)
            psi_pred.append(psi_pred_t)
        psi_pred = torch.stack(psi_pred)  # (T, d_B)

    # 共分散推定器を使用
    d_B = psi_obs.size(1)
    estimator = ObservationNoiseCovarianceEstimator(
        d_B=d_B,
        regularization=regularization
    )

    R_estimated = estimator.estimate_from_sequences(psi_obs, psi_pred)
    return R_estimated


def create_adaptive_multivariate_kalman_filter(
    df_state_layer,
    df_obs_layer,
    encoder,
    initial_observations: torch.Tensor,
    device: str = 'cpu',
    **kwargs
):
    """
    観測データから適応的に観測ノイズ共分散を推定してKalmanフィルターを作成

    Args:
        df_state_layer: 学習済みDF-A層
        df_obs_layer: 学習済みDF-B層
        encoder: エンコーダー
        initial_observations: 初期観測データ (T_init, n)
        device: 計算デバイス
        **kwargs: その他のパラメータ

    Returns:
        OperatorBasedKalmanFilter: 適応的に調整されたフィルター
    """
    # 初期Kalmanフィルターを作成
    kalman_filter = create_multivariate_kalman_filter(
        df_state_layer=df_state_layer,
        df_obs_layer=df_obs_layer,
        encoder=encoder,
        device=device,
        **kwargs
    )

    # 初期観測から観測ノイズ共分散を推定
    regularization = kwargs.get('noise_regularization', 1e-3)
    R_estimated = estimate_multivariate_observation_noise_covariance(
        kalman_filter=kalman_filter,
        observations=initial_observations,
        regularization=regularization
    )

    # 推定した共分散でフィルターを更新
    kalman_filter.R = R_estimated

    return kalman_filter


def batch_multivariate_filtering(
    df_state_layer,
    df_obs_layer,
    encoder,
    observation_sequences: torch.Tensor,
    adaptive_noise: bool = True,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    バッチ観測系列に対する多変量Kalmanフィルタリング

    Args:
        df_state_layer: 学習済みDF-A層
        df_obs_layer: 学習済みDF-B層
        encoder: エンコーダー
        observation_sequences: 観測系列 (batch_size, T, n)
        adaptive_noise: 適応的ノイズ推定を使用するか
        device: 計算デバイス

    Returns:
        Dict containing:
            - filtered_states: (batch_size, T, r)
            - state_covariances: (batch_size, T, r, r)
            - estimated_noise_covariances: (batch_size, d_B, d_B) if adaptive_noise
    """
    batch_size, T, n = observation_sequences.shape

    # 次元情報を取得
    r = df_state_layer.get_readout_matrix().size(1)  # 元状態次元
    d_B = df_obs_layer.get_observation_operator().size(0)  # 観測特徴次元

    # 結果格納用
    results = {
        'filtered_states': torch.zeros(batch_size, T, r, device=device),
        'state_covariances': torch.zeros(batch_size, T, r, r, device=device)
    }

    if adaptive_noise:
        results['estimated_noise_covariances'] = torch.zeros(batch_size, d_B, d_B, device=device)

    # 各系列に対してフィルタリング実行
    for b in range(batch_size):
        obs_seq = observation_sequences[b]  # (T, n)

        if adaptive_noise:
            # 適応的フィルター作成
            kalman_filter = create_adaptive_multivariate_kalman_filter(
                df_state_layer=df_state_layer,
                df_obs_layer=df_obs_layer,
                encoder=encoder,
                initial_observations=obs_seq,
                device=device
            )
            results['estimated_noise_covariances'][b] = kalman_filter.R
        else:
            # 標準フィルター作成
            kalman_filter = create_multivariate_kalman_filter(
                df_state_layer=df_state_layer,
                df_obs_layer=df_obs_layer,
                encoder=encoder,
                device=device
            )

        # フィルタリング実行
        X_means, X_covariances, _ = kalman_filter.filter_sequence(obs_seq)

        results['filtered_states'][b] = X_means
        results['state_covariances'][b] = X_covariances

    return results


def validate_multivariate_kalman_setup(
    df_state_layer,
    df_obs_layer,
    encoder,
    sample_observation: torch.Tensor
) -> Dict[str, Any]:
    """
    多変量Kalmanフィルター設定の検証

    Args:
        df_state_layer: DF-A層
        df_obs_layer: DF-B層
        encoder: エンコーダー
        sample_observation: サンプル観測 (n,)

    Returns:
        Dict: 検証結果
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "dimension_info": {},
        "numerical_stability": {}
    }

    try:
        # 次元取得
        V_A = df_state_layer.get_state_operator()
        V_B = df_obs_layer.get_observation_operator()
        U_A = df_state_layer.get_readout_matrix()
        U_B = df_obs_layer.get_readout_matrix()

        d_A = V_A.size(0)
        d_B = V_B.size(0)
        r = U_A.size(1)
        m = U_B.size(1)

        validation_results["dimension_info"] = {
            "d_A": d_A,
            "d_B": d_B,
            "r": r,
            "m": m,
            "observation_dim": sample_observation.numel()
        }

        # 次元整合性チェック
        if V_A.shape != (d_A, d_A):
            validation_results["errors"].append(f"V_A is not square: {V_A.shape}")
            validation_results["valid"] = False

        if V_B.shape != (d_B, d_A):
            validation_results["errors"].append(f"V_B dimension mismatch: expected ({d_B}, {d_A}), got {V_B.shape}")
            validation_results["valid"] = False

        if U_A.shape != (d_A, r):
            validation_results["errors"].append(f"U_A dimension mismatch: expected ({d_A}, {r}), got {U_A.shape}")
            validation_results["valid"] = False

        if U_B.shape != (d_B, m):
            validation_results["errors"].append(f"U_B dimension mismatch: expected ({d_B}, {m}), got {U_B.shape}")
            validation_results["valid"] = False

        # エンコーダー出力次元チェック
        with torch.no_grad():
            sample_obs_batch = sample_observation.unsqueeze(0).unsqueeze(0)  # (1, 1, n)
            encoder_output = encoder(sample_obs_batch).squeeze()
            actual_m = encoder_output.numel()

            if actual_m != m:
                validation_results["errors"].append(f"Encoder output dimension mismatch: expected {m}, got {actual_m}")
                validation_results["valid"] = False

        # DF-B層のpsi_omega存在チェック
        if not hasattr(df_obs_layer, 'psi_omega'):
            validation_results["errors"].append("DF-B layer missing psi_omega network")
            validation_results["valid"] = False
        else:
            # psi_omega入出力次元チェック
            try:
                with torch.no_grad():
                    dummy_m = torch.randn(m)
                    psi_output = df_obs_layer.psi_omega(dummy_m)
                    actual_d_B = psi_output.numel()

                    if actual_d_B != d_B:
                        validation_results["errors"].append(f"psi_omega output dimension mismatch: expected {d_B}, got {actual_d_B}")
                        validation_results["valid"] = False
            except Exception as e:
                validation_results["errors"].append(f"psi_omega test failed: {e}")
                validation_results["valid"] = False

        # 数値安定性チェック（エラーでなく警告）
        if validation_results["valid"]:
            # V_A の固有値チェック
            try:
                eigenvals_A = torch.linalg.eigvals(V_A)
                max_eigenval = torch.abs(eigenvals_A).max().item()
                validation_results["numerical_stability"]["V_A_max_eigenvalue"] = max_eigenval

                if max_eigenval > 1.0:
                    validation_results["warnings"].append(f"V_A potentially unstable: max eigenvalue {max_eigenval:.3f}")
            except Exception as e:
                validation_results["warnings"].append(f"V_A eigenvalue check failed: {e}")

            # 行列条件数チェック
            for name, matrix in [("V_A", V_A), ("V_B", V_B), ("U_A", U_A), ("U_B", U_B)]:
                try:
                    cond_num = torch.linalg.cond(matrix).item()
                    validation_results["numerical_stability"][f"{name}_condition_number"] = cond_num

                    if cond_num > 1e8:
                        validation_results["warnings"].append(f"{name} ill-conditioned: condition number {cond_num:.2e}")
                except Exception as e:
                    validation_results["warnings"].append(f"{name} condition number check failed: {e}")

    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation failed: {e}")

    return validation_results