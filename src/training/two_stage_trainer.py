# src/training/two_stage_trainer.py
"""
TwoStageTrainer: 提案手法の2段階学習戦略実装

Phase-1: DF-A/DF-B の Stage-1/Stage-2 交互学習
Phase-2: End-to-end 微調整

学習戦略:
**DF-A (State Layer)**:
for epoch in Phase1:
  for t = 1 to T1:  # Stage-1
    V_A^{(-k)} = 閉形式解(Φ_minus, Φ_plus, ϕ_θ固定)
    ϕ_θ ← ϕ_θ - α∇L1(V_A^{(-k)}, ϕ_θ)  # ϕ_θ更新
 
  for t = 1 to T2:  # Stage-2
    U_A = 閉形式解(H^{(cf)}_A, X_+)        # U_A更新（閉形式解のみ）

**DF-B (Observation Layer)**:
for epoch in Phase1:
  for t = 1 to T1:  # Stage-1
    V_B = 閉形式解(Φ_prev, Ψ_curr)       # V_B計算（ψ_ω固定）
    ϕ_θ ← ϕ_θ - α∇L1(V_B, ϕ_θ)         # ϕ_θ更新（ψ_ω固定）
 
  for t = 1 to T2:  # Stage-2 
    u_B = 閉形式解(H^{(cf)}_B, m)        # u_B計算（ϕ_θ固定）
    ψ_ω ← ψ_ω - α∇L2(u_B, ψ_ω)         # ψ_ω更新（ϕ_θ固定）

Phase-2: End-to-end微調整
for epoch in Phase2:
  # 推論パス固定
  x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
  m̂_{t|t-1} = u_B^T V_B ϕ_θ(x̂_{t|t-1})
  ŷ_{t|t-1} = g_α(m̂_{t|t-1})
 
  # 損失
  L_total = L_rec + λ_c L_cca
 
  # 選択的更新
  (u_η, g_α, ϕ_θ, ψ_ω).backward()
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import warnings
from pathlib import Path
import json
import csv
from dataclasses import dataclass
from enum import Enum

# 修正済みコンポーネントのインポート
from ..ssm.df_state_layer import DFStateLayer
from ..ssm.df_observation_layer import DFObservationLayer
from ..ssm.realization import Realization
from ..models.architectures.tcn import tcnEncoder, tcnDecoder


class TrainingPhase(Enum):
    """学習フェーズの定義"""
    PHASE1_DF_A = "phase1_df_a"
    PHASE1_DF_B = "phase1_df_b"
    PHASE2_E2E = "phase2_e2e"


@dataclass
class TrainingConfig:
    """学習設定の構造化"""
    # Phase-1設定
    phase1_epochs: int = 50
    T1_iterations: int = 10  # Stage-1反復数
    T2_iterations: int = 5   # Stage-2反復数
    df_a_warmup_epochs: int = 5  # DF-Bを開始する前のDF-Aウォームアップ
    
    # Phase-2設定
    phase2_epochs: int = 100
    lambda_cca: float = 0.1
    update_strategy: str = "all"  # "all" or "encoder_decoder_only"
    
    # 学習率
    lr_phi: float = 1e-3     # φ_θ (状態特徴) 学習率
    lr_psi: float = 1e-3     # ψ_ω (観測特徴) 学習率
    lr_encoder: float = 1e-4 # エンコーダ学習率
    lr_decoder: float = 1e-4 # デコーダ学習率
    
    # ログ・保存設定
    log_interval: int = 1    # ログ出力間隔（エポック）
    save_interval: int = 10  # モデル保存間隔（エポック）
    verbose: bool = True     # 詳細ログ
    
    def __post_init__(self):
        """初期化後の型変換と検証"""
        # 数値型の確実な変換（YAML読み込み対策）
        self.phase1_epochs = int(self.phase1_epochs)
        self.T1_iterations = int(self.T1_iterations)
        self.T2_iterations = int(self.T2_iterations)
        self.df_a_warmup_epochs = int(self.df_a_warmup_epochs)
        self.phase2_epochs = int(self.phase2_epochs)
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        
        # 学習率の型変換
        self.lr_phi = float(self.lr_phi)
        self.lr_psi = float(self.lr_psi)
        self.lr_encoder = float(self.lr_encoder)
        self.lr_decoder = float(self.lr_decoder)
        self.lambda_cca = float(self.lambda_cca)
        
        # 文字列型の正規化
        self.update_strategy = str(self.update_strategy)
        
        # 真偽値の変換（"true"/"false"文字列対策）
        if isinstance(self.verbose, str):
            self.verbose = self.verbose.lower() in ('true', '1', 'yes', 'on')
        else:
            self.verbose = bool(self.verbose)


class TrainingLogger:
    """学習過程のログ管理"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログデータ
        self.phase1_logs: List[Dict[str, Any]] = []
        self.phase2_logs: List[Dict[str, Any]] = []
        self.current_phase: Optional[TrainingPhase] = None
        
        # CSV ファイル初期化
        self.phase1_csv_path = self.output_dir / "phase1_training.csv"
        self.phase2_csv_path = self.output_dir / "phase2_training.csv"
        
        self._init_csv_files()
    
    def _init_csv_files(self):
        """CSVファイルのヘッダ初期化"""
        # Phase-1 CSV
        with open(self.phase1_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'stage', 'iteration', 
                'df_a_stage1_loss', 'df_a_stage2_loss',
                'df_b_stage1_loss', 'df_b_stage2_loss',
                'lr_phi', 'lr_psi'
            ])
        
        # Phase-2 CSV
        with open(self.phase2_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'total_loss', 'rec_loss', 'cca_loss',
                'lr_encoder', 'lr_decoder', 'lr_phi', 'lr_psi'
            ])
    def enable_time_alignment_debug(self):
        """時間対応のデバッグ情報を有効化"""
        self._debug_time_alignment = True
        if self.config.verbose:
            print("時間対応デバッグモード有効化")

    def disable_time_alignment_debug(self):
        """時間対応のデバッグ情報を無効化"""
        if hasattr(self, '_debug_time_alignment'):
            delattr(self, '_debug_time_alignment')
    
    def log_phase1(self, epoch: int, phase: TrainingPhase, stage: str, 
                   iteration: int, metrics: Dict[str, float], 
                   learning_rates: Dict[str, float]):
        """Phase-1ログ記録"""
        log_entry = {
            'epoch': epoch,
            'phase': phase.value,
            'stage': stage,
            'iteration': iteration,
            'metrics': metrics,
            'learning_rates': learning_rates,
            'timestamp': torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        }
        
        self.phase1_logs.append(log_entry)
        
        # CSV書き込み
        with open(self.phase1_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase.value, stage, iteration,
                metrics.get('df_a_stage1_loss', ''),
                metrics.get('df_a_stage2_loss', ''),
                metrics.get('df_b_stage1_loss', ''),
                metrics.get('df_b_stage2_loss', ''),
                learning_rates.get('lr_phi', ''),
                learning_rates.get('lr_psi', '')
            ])
    
    def log_phase2(self, epoch: int, total_loss: float, rec_loss: float, 
                   cca_loss: float, learning_rates: Dict[str, float]):
        """Phase-2ログ記録"""
        log_entry = {
            'epoch': epoch,
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'cca_loss': cca_loss,
            'learning_rates': learning_rates
        }
        
        self.phase2_logs.append(log_entry)
        
        # CSV書き込み
        with open(self.phase2_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, total_loss, rec_loss, cca_loss,
                learning_rates.get('lr_encoder', ''),
                learning_rates.get('lr_decoder', ''),
                learning_rates.get('lr_phi', ''),
                learning_rates.get('lr_psi', '')
            ])
    
    def save_summary(self):
        """学習サマリをJSONで保存"""
        summary = {
            'phase1_summary': {
                'total_epochs': len(set(log['epoch'] for log in self.phase1_logs)),
                'total_iterations': len(self.phase1_logs),
                'final_metrics': self.phase1_logs[-1]['metrics'] if self.phase1_logs else {}
            },
            'phase2_summary': {
                'total_epochs': len(self.phase2_logs),
                'final_loss': self.phase2_logs[-1]['total_loss'] if self.phase2_logs else None
            }
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


class TwoStageTrainer:
    """
    提案手法の2段階学習戦略を実行するメインクラス
    
    統合的な学習管理:
    1. Phase-1: DF-A/DF-B の協調学習
    2. Phase-2: End-to-end微調整
    3. 学習過程の詳細ログ・可視化
    4. モデル保存・復元
    """
    
    def __init__(
        self,
        encoder: tcnEncoder,
        decoder: tcnDecoder,
        realization: Realization,
        df_state_config: Dict[str, Any],
        df_obs_config: Dict[str, Any],
        training_config: TrainingConfig,
        device: torch.device,
        output_dir: str
    ):
        """
        Args:
            encoder: TCNエンコーダ（スカラー特徴量生成）
            decoder: TCNデコーダ（観測再構成）
            realization: 確率的実現クラス
            df_state_config: DF-A設定
            df_obs_config: DF-B設定
            training_config: 学習設定
            device: 計算デバイス
            output_dir: 出力ディレクトリ
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.realization = realization
        self.config = training_config
        self.device = device
        
        # ログ管理
        self.logger = TrainingLogger(output_dir)
        self.output_dir = Path(output_dir)
        
        # DF-A/DF-B初期化（遅延初期化）
        self.df_state: Optional[DFStateLayer] = None
        self.df_obs: Optional[DFObservationLayer] = None
        self.df_state_config = df_state_config
        self.df_obs_config = df_obs_config
        
        # 最適化器（遅延初期化）
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        
        # 学習状態
        self.current_epoch = 0
        self.phase1_complete = False
        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': []
        }
        
        # 一時データ保存
        self._temp_data = {}
    
    def _initialize_df_layers(self, X_states: torch.Tensor):
        """
        DF-AとDF-Bを初期化
        
        Args:
            X_states: 初期状態系列（状態次元推定用）
        """
        if self.df_state is not None:
            return  # 既に初期化済み
        
        _, state_dim = X_states.shape
        
        # DF-A初期化
        self.df_state = DFStateLayer(
            state_dim=state_dim,
            **self.df_state_config
        )
        
        # DF-B初期化（DF-Aに依存）
        self.df_obs = DFObservationLayer(
            df_state_layer=self.df_state,
            **self.df_obs_config
        )
        
        if self.config.verbose:
            print(f"DF-A初期化: 状態次元={state_dim}, 特徴次元={self.df_state.feature_dim}")
            print(f"DF-B初期化: 観測特徴次元={self.df_obs.obs_feature_dim}")
    
    def _initialize_optimizers(self):
        """最適化器の初期化"""
        if self.df_state is None or self.df_obs is None:
            raise RuntimeError("DF layersが初期化されていません")
        
        self.optimizers = {
            'phi': torch.optim.Adam(
                self.df_state.phi_theta.parameters(),
                lr=self.config.lr_phi
            ),
            'psi': torch.optim.Adam(
                self.df_obs.psi_omega.parameters(),
                lr=self.config.lr_psi
            ),
            'encoder': torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.config.lr_encoder
            ),
            'decoder': torch.optim.Adam(
                self.decoder.parameters(),
                lr=self.config.lr_decoder
            ),
            'e2e': None  # Phase-2で初期化
        }
        
        # Phase-2用統合オプティマイザ
        self.optimizers['e2e'] = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.config.lr_encoder},
            {'params': self.decoder.parameters(), 'lr': self.config.lr_decoder},
            {'params': self.df_state.phi_theta.parameters(), 'lr': self.config.lr_phi},
            {'params': self.df_obs.psi_omega.parameters(), 'lr': self.config.lr_psi}
        ])
    
    def _prepare_data(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データ前処理: エンコード→状態推定
        
        Args:
            Y_train: 訓練観測系列 (T, d)
            
        Returns:
            m_series: スカラー特徴量系列 (T,)
            X_states: 状態系列 (T_eff, r)
        """
        T, d = Y_train.shape
        
        # 1. エンコード: y_t → m_t
            # バッチ次元追加: (T, d) -> (1, T, d)
        m_tensor = self.encoder(Y_train.unsqueeze(0))  # (1, T, 1)
        m_series = m_tensor.squeeze()  # (T,)
        
        if m_series.dim() == 0:  # スカラーの場合
            m_series = m_series.unsqueeze(0)
        
        if self.config.verbose:
            print(f"エンコード完了: {Y_train.shape} -> {m_series.shape}")
        
        # 2. 確率的実現: m_t → x_t
        self.realization.fit(m_series.unsqueeze(1))  # (T,) -> (T, 1)
        X_states = self.realization.filter(m_series.unsqueeze(1))  # (T_eff, r)
        
        if self.config.verbose:
            print(f"状態推定完了: {m_series.shape} -> {X_states.shape}")
        
        # 一時保存
        self._temp_data = {
            'Y_train': Y_train,
            'm_series': m_series,
            'X_states': X_states
        }
        
        return m_series, X_states
    
    def train_phase1(self, Y_train: torch.Tensor) -> Dict[str, Any]:
        """
        Phase-1: DF-A/DF-B の協調学習
        
        Args:
            Y_train: 訓練観測系列 (T, d)
            
        Returns:
            Phase-1メトリクス
        """
        print("\n=== Phase-1: DF学習開始 ===")
        
        # データ準備
        m_series, X_states = self._prepare_data(Y_train)
        
        # DF layers初期化
        self._initialize_df_layers(X_states)
        self._initialize_optimizers()
        
        # Phase-1学習ループ
        for epoch in range(self.config.phase1_epochs):
            self.current_epoch = epoch
            epoch_metrics = {}
            
            # DF-A学習
            df_a_metrics = self._train_df_a_epoch(X_states, epoch)
            epoch_metrics.update(df_a_metrics)
            
            # DF-B学習（ウォームアップ後）
            if epoch >= self.config.df_a_warmup_epochs:
                df_b_metrics = self._train_df_b_epoch(X_states, m_series, epoch)
                epoch_metrics.update(df_b_metrics)
            
            # ログ記録
            self.training_history['phase1_metrics'].append(epoch_metrics)
            
            # ログ出力
            if epoch % self.config.log_interval == 0 and self.config.verbose:
                self._print_phase1_progress(epoch, epoch_metrics)
            
            # モデル保存
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE1_DF_A)
        
        self.phase1_complete = True
        print("Phase-1 学習完了")
        
        return self.training_history['phase1_metrics']
    
    def _train_df_a_epoch(self, X_states: torch.Tensor, epoch: int) -> Dict[str, float]:
        """DF-A（状態層）のエポック学習（定式化準拠）"""
        metrics = {}
        opt_phi = self.optimizers['phi']
        
        # **修正**: Stage-1を1回だけ呼び出し、内部でT1反復
        stage1_metrics = self.df_state.train_stage1_with_gradients(
            X_states, 
            opt_phi,
            T1_iterations=self.config.T1_iterations  # **修正**: 設定から取得
        )
        
        metrics['df_a_stage1_loss'] = stage1_metrics['stage1_loss']
        
        # ログ記録（T1回分をまとめて記録）
        self.logger.log_phase1(
            epoch, TrainingPhase.PHASE1_DF_A, 'stage1', 0,
            stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
        )
        
        # Stage-2: U_A推定（T2回実行）
        stage2_losses = []
        for t in range(self.config.T2_iterations):
            stage2_metrics = self.df_state.train_stage2_closed_form()
            stage2_losses.append(stage2_metrics['stage2_loss'])
            
            # ログ記録
            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_A, 'stage2', t,
                stage2_metrics, {}
            )
        
        metrics['df_a_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        
        return metrics
    
    def _train_df_b_epoch(self, X_states: torch.Tensor, m_series: torch.Tensor, epoch: int) -> Dict[str, float]:
        """DF-B（観測層）のエポック学習（定式化準拠）"""
        metrics = {}
        opt_phi = self.optimizers['phi']
        opt_psi = self.optimizers['psi']
        
        # DF-Aからの状態予測を取得
        X_hat_states = self.df_state.predict_sequence(X_states)
        
        # **修正**: Stage-1を1回だけ呼び出し、内部でT1反復
        stage1_metrics = self.df_obs.train_stage1_with_gradients(
            X_hat_states, 
            m_series, 
            opt_phi, 
            T1_iterations=self.config.T1_iterations,  # **修正**: 設定から取得
            fix_psi_omega=True
        )
        
        metrics['df_b_stage1_loss'] = stage1_metrics['stage1_loss']
        
        # ログ記録
        self.logger.log_phase1(
            epoch, TrainingPhase.PHASE1_DF_B, 'stage1', 0,
            stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
        )
        
        # Stage-2: u_B推定 + ψ_ω更新（T2回反復）
        stage2_losses = []
        for t in range(self.config.T2_iterations):
            stage2_metrics = self.df_obs.train_stage2_with_gradients(
                m_series, opt_psi, fix_phi_theta=True
            )
            stage2_losses.append(stage2_metrics['stage2_loss'])
            
            # ログ記録
            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_B, 'stage2', t,
                stage2_metrics, {'lr_psi': opt_psi.param_groups[0]['lr']}
            )
        
        metrics['df_b_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        
        return metrics
    
    def train_phase2(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Phase-2: End-to-end微調整
        
        固定推論パス:
        x̂_{t|t-1} = U_A^T V_A φ_θ(x_{t-1})
        m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
        ŷ_{t|t-1} = g_α(m̂_{t|t-1})
        
        Args:
            Y_train: 訓練観測系列
            Y_val: 検証観測系列（オプション）
        """
        print("\n=== Phase-2: End-to-end微調整開始 ===")
        
        if not self.phase1_complete:
            raise RuntimeError("Phase-1が完了していません")
        
        opt_e2e = self.optimizers['e2e']
        
        # Phase-2学習ループ
        for epoch in range(self.config.phase2_epochs):
            self.current_epoch = self.config.phase1_epochs + epoch
            
            # 前向き推論と損失計算
            loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train)
            
            # 逆伝播
            opt_e2e.zero_grad()
            loss_total.backward()
            opt_e2e.step()
            
            # ログ記録
            lr_dict = {f'lr_{name}': group['lr'] for name, group in 
                      zip(['encoder', 'decoder', 'phi', 'psi'], opt_e2e.param_groups)}
            self.logger.log_phase2(epoch, loss_total.item(), rec_loss.item(), 
                                  cca_loss.item(), lr_dict)
            
            self.training_history['phase2_losses'].append({
                'epoch': epoch,
                'total_loss': loss_total.item(),
                'rec_loss': rec_loss.item(),
                'cca_loss': cca_loss.item()
            })
            
            # 進捗表示
            if epoch % self.config.log_interval == 0 and self.config.verbose:
                print(f"Phase-2 Epoch {epoch}: Total={loss_total.item():.6f}, "
                      f"Rec={rec_loss.item():.6f}, CCA={cca_loss.item():.6f}")
            
            # モデル保存
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE2_E2E)
        
        print("Phase-2 学習完了")
        return self.training_history['phase2_losses']
    
    def _forward_and_loss_phase2(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        **修正版**: Phase-2の前向き推論と損失計算（厳密な時間管理）
        
        時間関係の厳密な定式化:
        Step 1: {y_t}_{t=0}^{T-1} → {m_t}_{t=0}^{T-1}
        Step 2: {m_t} → {x_t}_{t=h}^{T-h} (length: T_eff = T-2h+1)
        Step 3: x_{t-1} → x̂_{t|t-1} for t ∈ {h+1, ..., T-h} (length: T_pred = T_eff-1)
        Step 4: x̂_{t|t-1} → m̂_{t|t-1} for t ∈ {h+1, ..., T-h}
        Step 5: m̂_{t|t-1} → ŷ_{t|t-1}
        Step 6: Compare ŷ_{t|t-1} with y_t for t ∈ {h+1, ..., T-h}
        
        Returns:
            (total_loss, rec_loss, cca_loss)
        """
        T, d = Y_train.shape
        h = self.realization.h
        
        if T <= 2 * h:
            # 時系列が短すぎる場合のフォールバック
            return (torch.tensor(1e6, device=self.device), 
                    torch.tensor(0., device=self.device), 
                    torch.tensor(0., device=self.device))
        
        # Step 1: エンコード {y_t}_{t=0}^{T-1} → {m_t}_{t=0}^{T-1}
        m_tensor = self.encoder(Y_train.unsqueeze(0))  # (1, T, 1)
        m_series = m_tensor.squeeze()  # (T,)
        if m_series.dim() == 0:  # スカラーの場合
            m_series = m_series.unsqueeze(0)
        
        # Step 2: 確率的実現 {m_t} → {x_t}_{t=h}^{T-h}
        self.realization.fit(m_series.unsqueeze(1))
        X_states = self.realization.filter(m_series.unsqueeze(1))  # (T_eff, r)
        T_eff = X_states.size(0)  # T_eff = T - 2h + 1
        
        if T_eff <= 1:
            # 状態系列が短すぎる場合
            return (torch.tensor(1e6, device=self.device), 
                    torch.tensor(0., device=self.device), 
                    torch.tensor(0., device=self.device))
        
        # 時間対応の確認（デバッグ用）
        if self.config.verbose and hasattr(self, '_debug_time_alignment'):
            print(f"[DEBUG] Time alignment: T={T}, h={h}, T_eff={T_eff}")
            print(f"[DEBUG] X_states corresponds to x_{{h}} to x_{{T-h}} = x_{{{h}}} to x_{{{h+T_eff-1}}}")
        
        # Step 3: DF-A予測 x_{t-1} → x̂_{t|t-1}
        # 入力: x_h, x_{h+1}, ..., x_{T-h-1} (X_states[0:T_eff-1])
        # 出力: x̂_{h+1|h}, x̂_{h+2|h+1}, ..., x̂_{T-h|T-h-1} (length: T_pred = T_eff-1)
        X_hat_states = self.df_state.predict_sequence(X_states)  # (T_pred, r)
        T_pred = X_hat_states.size(0)  # T_pred = T_eff - 1 = T - 2h
        
        if T_pred <= 0:
            return (torch.tensor(1e6, device=self.device), 
                    torch.tensor(0., device=self.device), 
                    torch.tensor(0., device=self.device))
        
        # Step 4: DF-B予測 x̂_{t|t-1} → m̂_{t|t-1}
        # X_hat_states[i] = x̂_{h+1+i|h+i} for i = 0, 1, ..., T_pred-1
        m_hat_series = torch.zeros(T_pred, device=self.device, dtype=m_series.dtype)
        
        for i in range(T_pred):
            # X_hat_states[i] corresponds to x̂_{h+1+i|h+i}
            # 予測: m̂_{h+1+i|h+i}
            m_hat_i = self.df_obs.predict_one_step(X_hat_states[i])
            
            # スカラー確保
            if m_hat_i.dim() == 0:
                m_hat_series[i] = m_hat_i
            else:
                m_hat_series[i] = m_hat_i.squeeze()
        
        # Step 5: デコード m̂_{t|t-1} → ŷ_{t|t-1}
        # TCNデコーダは [B, T, 1] を期待
        m_hat_input = m_hat_series.unsqueeze(0).unsqueeze(2)  # (1, T_pred, 1)
        Y_hat = self.decoder(m_hat_input).squeeze(0)  # (T_pred, d)
        
        # Step 6: 損失計算 - 厳密な時間対応
        # 予測: ŷ_{h+1|h}, ŷ_{h+2|h+1}, ..., ŷ_{T-h|T-h-1}
        # 真値: y_{h+1}, y_{h+2}, ..., y_{T-h}
        # インデックス: h+1, h+2, ..., h+T_pred = h+1, ..., T-h
        
        target_start_idx = h + 1
        target_end_idx = target_start_idx + T_pred
        
        # 範囲チェック
        if target_end_idx > T:
            # 安全のため調整
            T_pred_safe = T - target_start_idx
            Y_hat = Y_hat[:T_pred_safe, :]
            target_end_idx = target_start_idx + T_pred_safe
        
        Y_target = Y_train[target_start_idx:target_end_idx, :]  # (T_pred, d)
        
        # 形状一致確認
        if Y_hat.size(0) != Y_target.size(0):
            min_len = min(Y_hat.size(0), Y_target.size(0))
            Y_hat = Y_hat[:min_len, :]
            Y_target = Y_target[:min_len, :]
        
        # デバッグ情報
        if self.config.verbose and hasattr(self, '_debug_time_alignment'):
            print(f"[DEBUG] Prediction: ŷ_{{t|t-1}} for t ∈ [{target_start_idx}, {target_end_idx-1}]")
            print(f"[DEBUG] Target: y_t for t ∈ [{target_start_idx}, {target_end_idx-1}]")
            print(f"[DEBUG] Y_hat.shape: {Y_hat.shape}, Y_target.shape: {Y_target.shape}")
        
        # 再構成損失
        rec_loss = torch.norm(Y_hat - Y_target, p='fro') ** 2
        
        # 正準相関損失の計算
        cca_loss = self._compute_cca_loss(m_series, X_states)
        
        # 総損失
        total_loss = rec_loss + self.config.lambda_cca * cca_loss
        
        return total_loss, rec_loss, cca_loss
    
    def _compute_cca_loss(self, m_series: torch.Tensor, X_states: torch.Tensor) -> torch.Tensor:
        """
        正準相関損失の計算
        
        論文Section 2.6の L_cca 実装:
        L_cca = -Σ ρ_i^2 (正準相関係数の二乗和を最大化)
        
        Args:
            m_series: スカラー特徴量系列 (T,)
            X_states: 状態系列 (T_eff, r)
        
        Returns:
            CCA損失: -Σ ρ_i^2
        """
        # Method 1: 確率的実現から特異値（正準相関係数）を取得
        if hasattr(self.realization, '_L_vals') and self.realization._L_vals is not None:
            singular_values = self.realization._L_vals  # σ_1, σ_2, ..., σ_r
            
            # 数値安定性のためのクリッピング
            singular_values = torch.clamp(singular_values, min=1e-8, max=1.0)
            
            # 正準相関係数の二乗和を最大化 → 負の損失
            cca_loss = -torch.sum(singular_values ** 2)
            
            if self.config.verbose and hasattr(self, '_debug_cca'):
                print(f"[DEBUG CCA] Singular values: {singular_values.cpu().numpy()}")
                print(f"[DEBUG CCA] CCA loss: {cca_loss.item():.6f}")
            
            return cca_loss
        else:
            # Method 2: フォールバック - エンコーダ特徴量と状態の共分散最大化
            if self.config.verbose:
                warnings.warn("確率的実現の特異値が利用できません。共分散ベースのCCA損失を使用します。")
            return self._compute_covariance_based_cca_loss(m_series, X_states)

    def _compute_covariance_based_cca_loss(self, m_series: torch.Tensor, X_states: torch.Tensor) -> torch.Tensor:
        """
        共分散ベースの代替CCA損失（フォールバック）
        
        スカラー特徴量と状態系列間の正規化共分散を最大化
        """
        h = self.realization.h
        T_eff = X_states.size(0)
        
        # 時間対応の調整：X_states[i] corresponds to x_{h+i}
        # m_series の対応する部分を取得
        m_aligned = m_series[h:h+T_eff]  # (T_eff,)
        X_aligned = X_states  # (T_eff, r)
        
        # データ長が一致しない場合の調整
        min_len = min(m_aligned.size(0), X_aligned.size(0))
        m_aligned = m_aligned[:min_len]
        X_aligned = X_aligned[:min_len, :]
        
        if min_len < 2:
            # データが不足している場合
            return torch.tensor(0.0, device=self.device)
        
        # 中心化
        m_centered = m_aligned - m_aligned.mean()  # (min_len,)
        X_centered = X_aligned - X_aligned.mean(dim=0, keepdim=True)  # (min_len, r)
        
        # 共分散計算
        cross_cov = torch.abs(torch.sum(m_centered.unsqueeze(1) * X_centered, dim=0))  # (r,)
        
        # 正規化項（数値安定性のための小さな値を追加）
        m_var = torch.var(m_centered, unbiased=False) + 1e-8
        X_var = torch.var(X_centered, dim=0, unbiased=False) + 1e-8  # (r,)
        
        # 正規化共分散（正準相関係数の近似）
        normalized_cov = cross_cov / torch.sqrt(m_var * X_var)  # (r,)
        
        # 二乗和を最大化 → 負の損失
        cca_loss = -torch.sum(normalized_cov ** 2)
        
        if self.config.verbose and hasattr(self, '_debug_cca'):
            print(f"[DEBUG CCA Fallback] Normalized covariances: {normalized_cov.cpu().numpy()}")
            print(f"[DEBUG CCA Fallback] CCA loss: {cca_loss.item():.6f}")
        
        return cca_loss

    def enable_cca_debug(self):
        """CCA損失のデバッグ情報を有効化"""
        self._debug_cca = True
        if self.config.verbose:
            print("CCA損失デバッグモード有効化")

    def disable_cca_debug(self):
        """CCA損失のデバッグ情報を無効化"""
        if hasattr(self, '_debug_cca'):
            delattr(self, '_debug_cca')

        
    def fit(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        完全な2段階学習実行
        
        Args:
            Y_train: 訓練観測系列 (T, d)
            Y_val: 検証観測系列 (T_val, d) [optional]
            
        Returns:
            学習履歴辞書
        """
        print("=== 提案手法の2段階学習開始 ===")
        
        try:
            # Phase-1学習
            phase1_metrics = self.train_phase1(Y_train)
            
            # Phase-2学習
            phase2_metrics = self.train_phase2(Y_train, Y_val)
            
            # 最終保存
            self._save_final_model()
            self.logger.save_summary()
            
            return {
                'phase1_metrics': phase1_metrics,
                'phase2_losses': phase2_metrics,
                'training_config': self.config.__dict__,
                'model_paths': {
                    'final_model': str(self.output_dir / 'final_model.pth'),
                    'logs': str(self.logger.output_dir)
                }
            }
            
        except Exception as e:
            print(f"学習中にエラーが発生: {e}")
            # 緊急保存
            self._save_checkpoint(self.current_epoch, TrainingPhase.PHASE1_DF_A, emergency=True)
            raise
    
    def _print_phase1_progress(self, epoch: int, metrics: Dict[str, float]):
        """Phase-1進捗表示"""
        df_a_s1 = metrics.get('df_a_stage1_loss', 0)
        df_a_s2 = metrics.get('df_a_stage2_loss', 0)
        df_b_s1 = metrics.get('df_b_stage1_loss', 0)
        df_b_s2 = metrics.get('df_b_stage2_loss', 0)
        
        print(f"Phase-1 Epoch {epoch:3d}: "
              f"DF-A(S1={df_a_s1:.4f}, S2={df_a_s2:.4f}) "
              f"DF-B(S1={df_b_s1:.4f}, S2={df_b_s2:.4f})")
    
    def _save_checkpoint(self, epoch: int, phase: TrainingPhase, emergency: bool = False):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase.value,
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'training_config': self.config.__dict__,
            'training_history': self.training_history,
            'phase1_complete': self.phase1_complete
        }
        
        # DF layers状態
        if self.df_state is not None:
            checkpoint['df_state'] = self.df_state.get_state_dict()
        if self.df_obs is not None:
            checkpoint['df_obs'] = self.df_obs.get_state_dict()
        
        # 最適化器状態
        opt_states = {}
        for name, opt in self.optimizers.items():
            if opt is not None:
                opt_states[name] = opt.state_dict()
        checkpoint['optimizer_states'] = opt_states
        
        # 保存パス
        if emergency:
            save_path = self.output_dir / f'emergency_checkpoint_epoch_{epoch}.pth'
        else:
            save_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
        
        if self.config.verbose:
            print(f"チェックポイント保存: {save_path}")
    
    def _save_final_model(self):
        """最終モデル保存"""
        final_model = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'df_state': self.df_state.get_state_dict() if self.df_state else None,
            'df_obs': self.df_obs.get_state_dict() if self.df_obs else None,
            'realization_config': {
                'past_horizon': self.realization.h,
                'jitter': self.realization.jitter,
                'cond_thresh': self.realization.cond_thresh,
                'rank': self.realization.rank,
                'reg_type': self.realization.reg_type
            },
            'training_config': self.config.__dict__,
            'training_complete': True,
            'final_metrics': {
                'phase1': self.training_history['phase1_metrics'][-1] if self.training_history['phase1_metrics'] else {},
                'phase2': self.training_history['phase2_losses'][-1] if self.training_history['phase2_losses'] else {}
            }
        }
        
        save_path = self.output_dir / 'final_model.pth'
        torch.save(final_model, save_path)
        print(f"最終モデル保存完了: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """チェックポイント復元"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # モデル状態復元
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.decoder.load_state_dict(checkpoint['decoder_state'])
            
            # 学習状態復元
            self.current_epoch = checkpoint['epoch']
            self.phase1_complete = checkpoint.get('phase1_complete', False)
            self.training_history = checkpoint.get('training_history', {'phase1_metrics': [], 'phase2_losses': []})
            
            print(f"チェックポイント復元完了: エポック={self.current_epoch}")
            return True
            
        except Exception as e:
            print(f"チェックポイント復元失敗: {e}")
            return False
    
    def predict(self, Y_test: torch.Tensor, forecast_steps: int = 1) -> torch.Tensor:
        """
        予測実行
        
        Args:
            Y_test: テスト観測系列 (T_test, d)
            forecast_steps: 予測ステップ数
            
        Returns:
            予測系列 (forecast_steps, d)
        """
        if not self.phase1_complete:
            raise RuntimeError("学習が完了していません")
        
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # 初期状態推定
            T_test, d = Y_test.shape
            warmup_len = min(T_test, self.realization.h + 10)
            Y_warmup = Y_test[:warmup_len]
            
            # エンコード
            m_warmup = self.encoder(Y_warmup.unsqueeze(0)).squeeze()
            
            # 状態推定
            self.realization.fit(m_warmup.unsqueeze(1))
            X_warmup = self.realization.filter(m_warmup.unsqueeze(1))
            
            # 逐次予測
            predictions = []
            x_current = X_warmup[-1]  # 最新状態
            
            for step in range(forecast_steps):
                # DF-A: 状態予測
                x_pred = self.df_state.predict_one_step(x_current)
                
                # DF-B: 特徴量予測
                m_pred = self.df_obs.predict_one_step(x_pred)
                
                # デコード: 観測予測
                m_input = m_pred.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1)
                y_pred = self.decoder(m_input).squeeze()  # (d,)
                
                predictions.append(y_pred)
                x_current = x_pred  # 状態更新
            
            return torch.stack(predictions)  # (forecast_steps, d)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """学習サマリ取得"""
        summary = {
            'training_complete': self.phase1_complete,
            'total_epochs': {
                'phase1': self.config.phase1_epochs,
                'phase2': self.config.phase2_epochs if self.phase1_complete else 0
            },
            'final_losses': {},
            'model_info': {
                'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
                'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
                'df_state_params': sum(p.numel() for p in self.df_state.phi_theta.parameters()) if self.df_state else 0,
                'df_obs_params': sum(p.numel() for p in self.df_obs.psi_omega.parameters()) if self.df_obs else 0
            }
        }
        
        if self.training_history['phase1_metrics']:
            summary['final_losses']['phase1'] = self.training_history['phase1_metrics'][-1]
        
        if self.training_history['phase2_losses']:
            summary['final_losses']['phase2'] = self.training_history['phase2_losses'][-1]
        
        return summary


# ユーティリティ関数
def create_trainer_from_config(config_path: str, device: torch.device, output_dir: str) -> TwoStageTrainer:
    """
    設定ファイルからトレーナーを作成
    
    Args:
        config_path: YAML設定ファイルパス
        device: 計算デバイス
        output_dir: 出力ディレクトリ
        
    Returns:
        TwoStageTrainer インスタンス
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # モデル初期化
    encoder = tcnEncoder(**config['model']['encoder'])
    decoder = tcnDecoder(**config['model']['decoder'])
    realization = Realization(**config['ssm']['realization'])
    
    # 学習設定
    try:
        training_config = TrainingConfig(**config['training'])
    except (TypeError, ValueError) as e:
        # ネスト構造の場合のフォールバック
        print(f"標準設定読み込み失敗 ({e}), ネスト構造を試行")
        training_config = TrainingConfig.from_nested_dict(config['training'])
    
    # トレーナー作成
    trainer = TwoStageTrainer(
        encoder=encoder,
        decoder=decoder,
        realization=realization,
        df_state_config=config['ssm']['df_state'],
        df_obs_config=config['ssm']['df_observation'],
        training_config=training_config,
        device=device,
        output_dir=output_dir
    )
    
    return trainer


def run_training_experiment(
    config_path: str,
    data_path: str,
    output_dir: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    学習実験の実行
    
    Args:
        config_path: 設定ファイルパス
        data_path: データファイルパス (.npz)
        output_dir: 結果出力ディレクトリ
        device: 計算デバイス（Noneなら自動選択）
        
    Returns:
        実験結果辞書
    """
    import numpy as np
    from ..utils.gpu_utils import select_device
    
    # デバイス設定
    if device is None:
        device = select_device()
    
    print(f"実験開始: device={device}")
    
    # データ読み込み
    data = np.load(data_path)
    if 'Y' in data:
        Y_train = torch.tensor(data['Y'], dtype=torch.float32, device=device)
    elif 'arr_0' in data:
        Y_train = torch.tensor(data['arr_0'], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"データファイルに 'Y' または 'arr_0' キーが見つかりません")
    
    print(f"データ読み込み完了: {Y_train.shape}")
    
    # トレーナー作成
    trainer = create_trainer_from_config(config_path, device, output_dir)
    
    # 学習実行
    results = trainer.fit(Y_train)
    
    # サマリ追加
    results['experiment_summary'] = trainer.get_training_summary()
    results['data_info'] = {
        'data_path': data_path,
        'data_shape': tuple(Y_train.shape),
        'device': str(device)
    }
    
    print(f"実験完了: 結果は {output_dir} に保存されました")
    
    return results