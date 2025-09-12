import torch
from torch.linalg import cholesky, solve_triangular, eigvalsh, svd
import warnings
from typing import Optional, Dict, Any, Tuple

class RealizationError(Exception):
    """
    Exception raised when stochastic realization (SSM identification)
    fails due to numerical issues (ill-conditioning, non-positive-definite, etc.).
    """
    pass

class Realization:
    """
    Exact subspace identification via canonical correlation analysis.
    
    Steps in fit(Y):
      1) Build past/future block Hankel matrices H_p, H_f
      2) Compute empirical covariances S_pp, S_ff, S_fp
      3) Add diagonal jitter ε to S_pp, S_ff for stability
      4) Check condition numbers; reject if too large
      5) Compute Cholesky factors L_pp, L_ff (S = L @ L.T)
      6) Solve triangular systems to get whitening matrices W_pp = L_pp⁻¹, W_ff = L_ff⁻¹
      7) Form T = W_ff @ S_fp @ W_pp and take its SVD → U, Lambda, Vᵀ
      8) Build state map B = Lambda^{1/2} Vᵀ W_pp
      9) Save singular values Lambda for downstream loss or analysis
    """
    def __init__(
        self,
        past_horizon: int,
        jitter: float = 1e-6,
        cond_thresh: float = 1e12,
        rank: int | None = None,
        reg_type : str = "sum",
    ):
        """
        Args:
        past_horizon:  number of time-lags in past/future Hankel blocks
        jitter:        small epsilon to add to diagonals of covariances
        cond_thresh:   max allowed condition number before rejection
        """
        # **修正**: 数値パラメータの型変換を明示的に実行
        self.h = int(past_horizon)
        self.jitter = float(jitter)  # **追加**: 明示的なfloat変換
        self.cond_thresh = float(cond_thresh)  # **追加**: 明示的なfloat変換
        
        # rank処理
        if rank is not None:
            self.rank = int(rank)  # **修正**: 明示的なint変換
        else:
            self.rank = rank
            
        self.reg_type = str(reg_type)  # **追加**: 明示的なstr変換
        
        # 初期化
        self.B = None
        self._L_vals = None

        # for debug
        self._Spp_eigvals = None
        self.H = None

    def fit(self, Y: torch.Tensor):
        T, p = Y.shape
        h = self.h
        N = T - 2*h + 1
        if N <= 0:
            # raise RealizationError("Time series too short for horizon h")
            return

        # 0) 中心化
        mu = Y.mean(dim=0, keepdim=True)
        Y_c = Y - mu 
        
        # # 1) ブロック‐ハンケル行列
        # H_p_float = torch.stack([Y_c[i : i+h].flip(dims=(0,)).reshape(-1) for i in range(N)], dim=1)
        # H_f_float = torch.stack([Y_c[i+1 : i+h+1].reshape(-1) for i in range(N)], dim=1)
        # H_p = H_p_float.double()
        # H_f = H_f_float.double()

        device = Y_c.device
        # m      = getattr(self, "m", 2048)
        m = 500
        rank   = self.rank      # 低ランク近似の次数
        # eps_chol   = 1e-7
        # eps_jitter = 1e-10
        eps_chol   = float(self.jitter)
        eps_jitter = float(self.jitter)
        q_over     = 5

        # 1. ── ラグ共分散 (バッチ外積, float32) -----------------
        idx = torch.randint(0, T - 2 * h - 1, (m,), device=device)
        Y0  = Y_c[idx]                          # (m, p)
        Lambda    = {}
        
        for l in range(0, 2 * h):
            Yl = Y_c[idx + l]                  # (m, p)
            cov = (Yl.T @ Y0) / m              # (p, p)
            Lambda[ l]  = cov
            if h + 1 > l > 0:
                Lambda[-l] = cov.T

        # 2. ── ブロック行列 (k×k ブロック) ----------------------
        dim_H   = h * p
        H32  = torch.zeros(dim_H, dim_H, dtype=torch.float32, device=device)
        Tp32 = torch.zeros_like(H32)
        
        z = torch.zeros(p, p, dtype=torch.float32, device=device)
        for i in range(h):
            for j in range(h):
                H32 [i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i + j + 1, z)
                Tp32[i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i - j, z)
        
        Tp32.diagonal().add_(eps_jitter)        # jitter for SPD

        # 3. ── 逆平方根 (float64 で) -----------------------------
        Tp64 = Tp32.to(torch.float64)
        Tp64 = 0.5 * (Tp64 + Tp64.T)
        try:
            L64  = torch.linalg.cholesky(
                      Tp64 + eps_chol*torch.eye(dim_H, device=device, dtype=torch.float64))
            W64  = torch.linalg.solve_triangular(
                      L64, torch.eye(dim_H, device=device, dtype=torch.float64), upper=False)
        except RuntimeError as e:
            print("real.fit failed cholesky decom.")
            eigvals, eigvecs = torch.linalg.eig(Tp64)   # eigvals: (n,), eigvecs: (n,n) complex
            eigvals = eigvals.real                   # 形状 (n,)
            eigvecs = eigvecs.real                   # 形状 (n,n)
            inv_sqrt = eigvals.rsqrt()       # (n,)
            D = torch.diag(inv_sqrt)                  # (n, n)
            W64 = eigvecs @ D @ eigvecs.T               # (n, n)
        
        # 4. ── 正規化 Hankel (float64) --------------------------
        T64 = W64.T @ (H32.to(torch.float64) @ W64)
        
        # 5. ── ランダム化 rank‑r SVD  or SVD----------------------------
        if rank is None or rank >= dim_H:
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            # B = Lambda^{1/2} Vᵀ W_pp
            self.B = torch.diag(S.pow(0.5)) @ Vh @ W64.to(torch.float32)
            self._L_vals = S
        else:                                  # ランク r の切り出し
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank,:]
            self.B = torch.diag(S_r.pow(0.5)) @ Vh_r @ W64.to(torch.float32)
            self._L_vals = S_r
            
            
        # r = min(rank, dim_H)
        # G   = torch.randn(dim_H, r + q_over, dtype=torch.float64, device=device)
        # Y2  = T64 @ G
        # Q, _= torch.linalg.qr(Y2, mode='reduced')
        # B   = Q.T @ T64
        # Ub, S64, Vh64 = torch.linalg.svd(B, full_matrices=False)
        # U64 = Q @ Ub
        # U_r = U64[:, :r]
        # S_r = S64[:r]
        # V_r = Vh64[:r, :].T       
        
        

        

        # #for debug
        # self.H =H_p
        # # print(f'first Y: {Y.detach().cpu()[:4, :4]}')
        
        # # 2) 経験共分散
        # # print(f'print N before computing S_pp: {N}') #debug
        # S_pp = (H_p @ H_p.T) / (N-1)
        # S_ff = (H_f @ H_f.T) / (N-1)
        # S_fp = (H_f @ H_p.T) / (N-1)


        # # 3) ジッター
        # I_pp = torch.eye(S_pp.size(0), device=S_pp.device).to(S_pp.dtype)
        # I_ff = torch.eye(S_ff.size(0), device=S_ff.device).to(S_pp.dtype)
        # S_pp = S_pp + self.jitter * I_pp
        # S_ff = S_ff + self.jitter * I_ff

        # # 3.5)対称化
        # S_pp = 0.5 * (S_pp + S_pp.T)
        # S_ff = 0.5 * (S_ff + S_ff.T)

        # #debug
        # self._Spp_eigvals = eigvalsh(S_pp)
        
        # # 4) 条件数チェック
        # def compute_eigvals(A):
        #     try:
        #         return eigvalsh(A)
        #     except RuntimeError:
        #         # eigh がダメなら SVD で特異値（≒固有値）取得
        #         return svd(A, compute_uv=False)
        '''
        #一旦除去
        # cond_pp = compute_eigvals(S_pp).max() / compute_eigvals(S_pp).min()
        # cond_ff = compute_eigvals(S_ff).max() / compute_eigvals(S_ff).min()
        # if cond_pp > self.cond_thresh or cond_ff > self.cond_thresh:
        #     raise RealizationError("block Covariance : ill-conditioned")
        # print(f'(real) debag: eigvalsh(S_pp):{eigvalsh(S_pp)}')
        '''
        '''
        # # -- debug dump --
        # # 1) そもそも finite か？
        # if not torch.isfinite(S_pp).all():
        #     print("S_pp contains non-finite entries!")
        # # 2) 先頭要素が何か
        # # print(f"S_pp[0,0]={S_pp[0,0].item()}, diag min={torch.min(torch.diag(S_pp)).item()}")
        # # 3) 固有値最小値／最大値
        # try:
        #     eigs = torch.linalg.eigvalsh(S_pp)
        #     # print(f"eigvals (min, max) = ({eigs[0].item()}, {eigs[-1].item()})")
        # except Exception as e:
        #     print("eigvalsh failed:", e)
        # # 4) 完全な行列のサンプル
        # # print("S_pp[0:3,0:3] =\n", S_pp[:3,:3].cpu().numpy())
        # # -- end debug dump --
        '''

        # # 5) Cholesky
        # L_pp = cholesky(S_pp)  # lower
        # L_ff = cholesky(S_ff)
        
        # # L_pp = L_pp.float(); L_ff = L_ff.float()
        # # I_pp = L_pp.float(); I_ff = L_ff.float()


        # # 6) Calculate root-inverse
        # W_pp = solve_triangular(L_pp, I_pp, upper=False)
        # W_ff = solve_triangular(L_ff, I_ff, upper=False)

        # W_pp = W_pp.float(); W_ff = W_ff.float(); S_fp = S_fp.float()

        '''
        # # 5.6) compute square-root inverse
        # eigval_p, Eigvecs_p = torch.linalg.eigh(S_pp)
        # inv_sqrt_p = eigval_p.rsqrt()
        # W_pp = Eigvecs_p @ torch.diag(inv_sqrt_p) @ Eigvecs_p.T
        # eigval_f, Eigvecs_f = torch.linalg.eigh(S_ff) 
        # inv_sqrt_f = eigval_f.rsqrt()
        # W_ff = Eigvecs_f @ torch.diag(inv_sqrt_f) @ Eigvecs_f.T
        '''

        # # 7) SVD
        # T_mat = W_ff @ S_fp @ W_pp.T
        # U, L_vals, Vt = svd(T_mat)

        # # 8) 低ランク切り出し
        # if 0 < self.rank < L_vals.numel():
        #     U_r   = U[:, :self.rank]            # (ph, rank)
        #     L_r   = L_vals[: self.rank]         # (rank,)
        #     Vt_r  = Vt[: self.rank, :]          # (rank, ph)
        #     L_vals = L_r
        #     Vt     = Vt_r
        #     # B = Lambda^{1/2} Vᵀ W_pp
        #     self.B = torch.diag(L_r.pow(0.5)) @ Vt_r @ W_pp
        # else:
        #     # フルランク版
        #     self.B = torch.diag(L_vals.pow(0.5)) @ Vt @ W_pp

        # # 9) 特異値を保存
        # self._L_vals = L_vals

    def filter(self, Y: torch.Tensor) -> torch.Tensor:
        h = self.h
        N = Y.shape[0] - 2*h + 1
        # Yf = torch.stack([Y[i+1 : i+h+1].reshape(-1) for i in range(N)], dim=1)
        Yp = torch.stack([Y[i : i + h].flip(dims=(0, )).reshape(-1) for i in range(N)], dim=1)
        X_state = (self.B @ Yp).T  # shape (N, r), time ; t = h,...,h+N-1

        self.X_state_torch = X_state

        return X_state

    def singular_value_reg(self, sv_weight : float) -> torch.Tensor:
        """
        特異値正則化を返す。
        reg_type=="sum"     -> sum(σ_i)
        reg_type=="squared" -> sum((1 - σ_i)^2)
        """
        if self.reg_type == "sum":
            _tr = self._L_vals.sum()
            _reg = -_tr
        elif self.reg_type == "squared":
            _reg = ((1 - self._L_vals) ** 2).sum()
        elif self.reg_type == "abs":
            _reg = (1 - self._L_vals).abs().sum()
            
        elif self.reg_type == "bounded":
            _reg = (1 - self._L_vals ** 2).sum()
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")

        return sv_weight * _reg

    """
    For filtering method
    """
    def filter_with_kalman(
        self,
        m_series: torch.Tensor,
        df_state_layer,  # DFStateLayerインスタンス
        df_obs_layer,    # DFObservationLayerインスタンス
        kalman_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        作用素ベースKalman更新による状態推定
        
        既存の決定的実現を拡張し、Algorithm 1による逐次状態推定を実行。
        状態の不確実性（共分散）も出力。
        
        Args:
            m_series: スカラー特徴量系列 (T,)
            df_state_layer: 学習済みDF-A層
            df_obs_layer: 学習済みDF-B層  
            kalman_config: Kalman Filter設定
            
        Returns:
            X_means: 状態平均系列 (T, r)
            X_covariances: 状態共分散系列 (T, r, r)
        """
        if kalman_config is None:
            kalman_config = {
                'noise_estimation': {'gamma_Q': 1e-6, 'gamma_R': 1e-6},
                'initialization': {'method': 'data_driven'},
                'device': 'cpu'
            }
        
        # 1. 従来のfilter()で初期状態推定
        if m_series.dim() == 1:
            m_input = m_series.unsqueeze(1)  # (T,) → (T, 1)
        else:
            m_input = m_series
            
        X_initial = self.filter(m_input)  # (T_eff, r)
        T_eff, r = X_initial.shape
        
        print(f"Initial state estimation: {X_initial.shape}")
        
        # 2. 演算子抽出と推論エンジン作成  
        try:
            estimator = self._create_kalman_estimator(
                df_state_layer, df_obs_layer, kalman_config
            )
        except Exception as e:
            warnings.warn(f"Kalman estimator creation failed: {e}. Using deterministic fallback.")
            return self._deterministic_fallback(X_initial)
        
        # 3. Algorithm 1実行
        try:
            # キャリブレーションデータ準備（初期部分を使用）
            n_calib = min(20, len(m_series) // 3)
            calib_data = self._prepare_calibration_data(m_series[:n_calib])
            
            # ノイズ推定と初期化
            estimator.estimate_noise_covariances(calib_data)
            estimator.initialize_filtering(calib_data[:10] if len(calib_data) > 10 else calib_data)
            
            # 観測系列準備（スカラー特徴量→多変量観測へのダミー変換）
            observations = self._prepare_observations_for_kalman(m_series)
            
            # バッチフィルタリング実行
            X_means, X_covariances = estimator.filter_sequence(observations)
            
            print(f"Kalman filtering completed: X_means={X_means.shape}, X_covariances={X_covariances.shape}")
            
            return X_means, X_covariances
            
        except Exception as e:
            warnings.warn(f"Kalman filtering failed: {e}. Using deterministic fallback.")
            return self._deterministic_fallback(X_initial)

    def _create_kalman_estimator(
        self,
        df_state_layer,
        df_obs_layer,
        kalman_config: Dict[str, Any]
    ):
        """Kalman推論エンジンの作成"""
        from ..inference.state_estimator import StateEstimator
        
        # 転送作用素の抽出確認
        if not (hasattr(df_state_layer, 'V_A') and df_state_layer.V_A is not None):
            raise RuntimeError("V_A not found in DF-A layer")
        if not (hasattr(df_state_layer, 'U_A') and df_state_layer.U_A is not None):
            raise RuntimeError("U_A not found in DF-A layer")
        if not (hasattr(df_obs_layer, 'V_B') and df_obs_layer.V_B is not None):
            raise RuntimeError("V_B not found in DF-B layer")  
        if not (hasattr(df_obs_layer, 'u_B') and df_obs_layer.u_B is not None):
            raise RuntimeError("u_B not found in DF-B layer")
        
        # StateEstimator設定
        estimator_config = {
            'device': kalman_config.get('device', 'cpu'),
            'model': {
                'df_state': {
                    'state_dim': df_state_layer.state_dim,
                    'feature_dim': df_state_layer.feature_dim
                },
                'df_obs': {
                    'obs_feature_dim': df_obs_layer.obs_feature_dim
                },
                'encoder': {
                    'input_dim': 1  # スカラー特徴量
                }
            },
            'noise_estimation': kalman_config.get('noise_estimation', {}),
            'initialization': kalman_config.get('initialization', {}),
            'numerical': kalman_config.get('numerical', {})
        }
        
        # StateEstimator作成
        estimator = StateEstimator(estimator_config)
        
        # 学習済みコンポーネントを手動設定
        estimator.df_state_layer = df_state_layer
        estimator.df_obs_layer = df_obs_layer
        
        # エンコーダは簡易版（恒等写像）
        class IdentityEncoder(torch.nn.Module):
            def forward(self, x):
                if x.dim() == 1:
                    return x.unsqueeze(0)  # (T,) → (1, T)
                return x.squeeze(-1) if x.size(-1) == 1 else x.mean(dim=-1)
        
        estimator.encoder = IdentityEncoder()
        
        # 演算子抽出
        estimator.V_A = df_state_layer.V_A.clone().detach()
        estimator.V_B = df_obs_layer.V_B.clone().detach() 
        estimator.U_A = df_state_layer.U_A.clone().detach()
        estimator.u_B = df_obs_layer.u_B.clone().detach()
        
        return estimator

    def _prepare_calibration_data(self, m_series: torch.Tensor) -> torch.Tensor:
        """キャリブレーション用データの準備"""
        # スカラー特徴量を多変量観測にダミー変換
        if m_series.dim() == 1:
            # 遅延埋め込みによる多変量化
            n_delays = 5
            calib_data = []
            for i in range(len(m_series)):
                delayed = []
                for d in range(n_delays):
                    if i - d >= 0:
                        delayed.append(m_series[i - d])
                    else:
                        delayed.append(torch.zeros_like(m_series[0]))
                calib_data.append(torch.stack(delayed))
            return torch.stack(calib_data)  # (T, n_delays)
        else:
            return m_series

    def _prepare_observations_for_kalman(self, m_series: torch.Tensor) -> torch.Tensor:
        """Kalman Filter用観測データの準備"""
        return self._prepare_calibration_data(m_series)

    def _deterministic_fallback(
        self, 
        X_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        決定的フォールバック
        
        Kalman更新が失敗した場合の代替手段。
        決定的推定値に固定共分散を付与。
        """
        T_eff, r = X_initial.shape
        
        # 状態平均はそのまま使用
        X_means = X_initial
        
        # 共分散は固定値（小さな不確実性）
        base_covariance = 0.01 * torch.eye(r, dtype=X_initial.dtype, device=X_initial.device)
        X_covariances = base_covariance.unsqueeze(0).expand(T_eff, r, r).clone()
        
        warnings.warn("Using deterministic fallback with fixed covariances")
        
        return X_means, X_covariances

    # =====================================
    # 既存機能の改善と追加ユーティリティ
    # =====================================

    def get_state_statistics(self) -> Dict[str, Any]:
        """
        状態推定の統計情報取得
        
        Returns:
            Dict: 統計情報
        """
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            return {"status": "not_fitted"}
        
        X = self.X_state_torch
        
        return {
            "state_shape": X.shape,
            "state_dimension": X.size(1),
            "sequence_length": X.size(0),
            "state_statistics": {
                "mean": torch.mean(X, dim=0).tolist(),
                "std": torch.std(X, dim=0).tolist(),
                "min": torch.min(X, dim=0)[0].tolist(),
                "max": torch.max(X, dim=0)[0].tolist()
            },
            "singular_values": {
                "values": self._L_vals.tolist() if hasattr(self, '_L_vals') and self._L_vals is not None else None,
                "condition_number": (self._L_vals.max() / self._L_vals.min()).item() if hasattr(self, '_L_vals') and self._L_vals is not None else None
            }
        }

    def predict_states(
        self,
        n_steps: int = 1,
        method: str = "linear"
    ) -> torch.Tensor:
        """
        状態の将来予測
        
        Args:
            n_steps: 予測ステップ数
            method: 予測手法 ("linear" | "last_value")
            
        Returns:
            torch.Tensor: 予測状態 (n_steps, r)
        """
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self.X_state_torch
        T, r = X.shape
        
        if method == "linear":
            # 線形外挿
            if T >= 2:
                trend = X[-1] - X[-2]  # 最新のトレンド
                predictions = []
                for step in range(1, n_steps + 1):
                    pred = X[-1] + step * trend
                    predictions.append(pred)
                return torch.stack(predictions)
            else:
                method = "last_value"  # フォールバック
        
        if method == "last_value":
            # 最後の値を繰り返し
            last_state = X[-1]
            return last_state.unsqueeze(0).expand(n_steps, r).clone()
        
        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def validate_kalman_compatibility(
        self,
        df_state_layer,
        df_obs_layer
    ) -> Dict[str, Any]:
        """
        Kalman更新との互換性検証
        
        Args:
            df_state_layer: DF-A層
            df_obs_layer: DF-B層
            
        Returns:
            Dict: 検証結果
        """
        validation = {
            "compatible": True,
            "issues": [],
            "requirements_met": {},
            "recommendations": []
        }
        
        # 必須コンポーネントの存在確認
        requirements = [
            ("df_state_layer.V_A", hasattr(df_state_layer, 'V_A') and df_state_layer.V_A is not None),
            ("df_state_layer.U_A", hasattr(df_state_layer, 'U_A') and df_state_layer.U_A is not None),
            ("df_obs_layer.V_B", hasattr(df_obs_layer, 'V_B') and df_obs_layer.V_B is not None),
            ("df_obs_layer.u_B", hasattr(df_obs_layer, 'u_B') and df_obs_layer.u_B is not None),
            ("realization_fitted", hasattr(self, 'X_state_torch') and self.X_state_torch is not None)
        ]
        
        for req_name, req_met in requirements:
            validation["requirements_met"][req_name] = req_met
            if not req_met:
                validation["compatible"] = False
                validation["issues"].append(f"Missing requirement: {req_name}")
        
        # 次元整合性チェック
        if validation["compatible"]:
            try:
                r_realization = self.rank if self.rank is not None else self.X_state_torch.size(1)
                r_df_state = df_state_layer.state_dim
                
                if r_realization != r_df_state:
                    validation["issues"].append(f"State dimension mismatch: realization={r_realization}, df_state={r_df_state}")
                    validation["compatible"] = False
                    
                # 特徴次元
                dA = df_state_layer.feature_dim
                dB = df_obs_layer.obs_feature_dim
                
                validation["dimensions"] = {
                    "state_dim": r_realization,
                    "feature_dim_A": dA,
                    "feature_dim_B": dB
                }
                
                # 推奨事項
                if dA < 2 * r_realization:
                    validation["recommendations"].append(f"Consider increasing feature_dim_A (current: {dA}, recommended: >={2*r_realization})")
                    
            except Exception as e:
                validation["compatible"] = False
                validation["issues"].append(f"Dimension check failed: {e}")
        
        return validation

    # =====================================
    # 設定ファイル対応
    # =====================================

    def create_kalman_config(
        self,
        **overrides
    ) -> Dict[str, Any]:
        """
        Kalman Filter用設定作成
        
        Args:
            **overrides: 設定上書き
            
        Returns:
            Dict: Kalman設定
        """
        default_config = {
            'noise_estimation': {
                'method': 'residual_based',
                'gamma_Q': 1e-6,
                'gamma_R': 1e-6
            },
            'initialization': {
                'method': 'data_driven',
                'n_init_samples': 10
            },
            'numerical': {
                'condition_threshold': 1e12,
                'min_eigenvalue': 1e-8,
                'jitter': 1e-6
            },
            'device': 'cpu'
        }
        
        # 上書き適用
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        config = default_config.copy()
        deep_update(config, overrides)
        
        return config
    

def build_realization(cfg) -> Realization:
    """
    Factory for convenience: 
      - cfg.h, cfg.jitter, cfg.cond_thresh are required.
      - cfg.rank may be None or >=0.
    Raises ValueError if rank is negative.
    """
    # rank の検証
    r = getattr(cfg, "rank", None)
    if r is not None and r < 0:
        raise ValueError(f"Invalid rank: {r} (must be >= 0 or None)")

    return Realization(
        past_horizon=cfg.h,
        jitter=cfg.jitter,
        cond_thresh=cfg.cond_thresh,
        rank=r,
        reg_type=getattr(cfg, "svd_reg_type", "sum"),
    )