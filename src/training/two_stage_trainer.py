# ===== src/training/two_stage_trainer.py 完全修正版 =====
# 修正1-4統合: 時間調整 + 計算グラフ分離 + ヘルパー関数

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

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import warnings
from pathlib import Path
import json
import csv
from dataclasses import dataclass
from enum import Enum
import gc

# コンポーネントインポート
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
    """学習ログ管理クラス"""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSVファイルパス
        self.phase1_csv_path = self.output_dir / 'phase1_training.csv'
        self.phase2_csv_path = self.output_dir / 'phase2_training.csv'
        
        # ログデータ
        self.phase1_logs = []
        self.phase2_logs = []
        
        # CSV初期化
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """CSVファイルのヘッダー初期化"""
        # Phase-1 CSV
        with open(self.phase1_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'stage', 'iteration', 'loss', 
                'lr_phi', 'lr_psi'
            ])
        
        # Phase-2 CSV
        with open(self.phase2_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'total_loss', 'rec_loss', 'cca_loss',
                'lr_encoder', 'lr_decoder', 'lr_phi', 'lr_psi'
            ])
    
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
            'learning_rates': learning_rates
        }
        
        self.phase1_logs.append(log_entry)
        
        # CSV書き込み
        with open(self.phase1_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            loss_value = metrics.get('stage1_loss') or metrics.get('stage2_loss', '')
            writer.writerow([
                epoch, phase.value, stage, iteration, loss_value,
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
    4. 時間インデックス調整とメモリ効率化
    """
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, realization: Realization,
                 df_state_config: Dict[str, Any], df_obs_config: Dict[str, Any],
                 training_config: TrainingConfig, device: torch.device, output_dir: str):
        # 基本設定
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.realization = realization
        self.config = training_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # **修正4: 時間調整用の状態管理**
        self._last_X_states_length = None  # 状態系列長キャッシュ
        
        # DF layers設定保存
        self.df_state_config = df_state_config
        self.df_obs_config = df_obs_config
        
        # 学習状態
        self.df_state = None
        self.df_obs = None
        self.optimizers = {}
        self.current_epoch = 0
        self.phase1_complete = False
        
        # 学習履歴
        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': []
        }
        
        # 一時データ保存
        self._temp_data = {}
        
        # ログ管理
        self.logger = TrainingLogger(self.output_dir)
        
        print(f"TwoStageTrainer初期化完了: {device}")
    
    def _initialize_df_layers(self, X_states: torch.Tensor):
        """DF layers初期化"""
        _, r = X_states.shape
        
        # DF-A初期化
        self.df_state = DFStateLayer(
            state_dim=r,
            **self.df_state_config
        ).to(self.device)
        
        # DF-B初期化（DF-Aを参照）
        self.df_obs = DFObservationLayer(
            df_state_layer=self.df_state,
            **self.df_obs_config
        ).to(self.device)
        
        print(f"DF layers初期化完了: state_dim={r}")
    
    def _initialize_optimizers(self):
        """最適化器初期化"""
        # Phase-1用の個別最適化器
        self.optimizers['phi'] = torch.optim.Adam(
            self.df_state.phi_theta.parameters(), 
            lr=self.config.lr_phi
        )
        
        self.optimizers['psi'] = torch.optim.Adam(
            self.df_obs.psi_omega.parameters(), 
            lr=self.config.lr_psi
        )
        
        # Phase-2用の統合最適化器
        if self.config.update_strategy == "all":
            # 全パラメータ更新
            phase2_params = list(self.encoder.parameters()) + \
                           list(self.decoder.parameters()) + \
                           list(self.df_state.phi_theta.parameters()) + \
                           list(self.df_obs.psi_omega.parameters())
        else:
            # エンコーダ・デコーダのみ更新
            phase2_params = list(self.encoder.parameters()) + \
                           list(self.decoder.parameters())
        
        self.optimizers['e2e'] = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.config.lr_encoder},
            {'params': self.decoder.parameters(), 'lr': self.config.lr_decoder},
            {'params': self.df_state.phi_theta.parameters(), 'lr': self.config.lr_phi},
            {'params': self.df_obs.psi_omega.parameters(), 'lr': self.config.lr_psi}
        ])
        
        print("最適化器初期化完了")
    
    def _prepare_data(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        **修正4統合**: データ準備（状態系列長の記録付き）
        """
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
        
        # **修正4: 状態系列長を記録（時間調整用）**
        self._last_X_states_length = X_states.size(0)
        
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
        """
        **修正2統合**: DF-A（状態層）のエポック学習（完全グラフ分離版）
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        
        # **修正2: 完全にデタッチされた入力で独立グラフ作成**
        X_states_detached = X_states.detach().requires_grad_(False)
        
        # **修正2: 独立計算コンテキストでDF-A学習**
        with torch.enable_grad():
            stage1_metrics = self.df_state.train_stage1_with_gradients(
                X_states_detached, 
                opt_phi,
                T1_iterations=self.config.T1_iterations
            )
        
        metrics['df_a_stage1_loss'] = stage1_metrics['stage1_loss']
        
        # ログ記録
        self.logger.log_phase1(
            epoch, TrainingPhase.PHASE1_DF_A, 'stage1', 0,
            stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
        )
        
        # Stage-2: U_A推定（T2回実行）
        stage2_losses = []
        for t in range(self.config.T2_iterations):
            with torch.no_grad():  # **修正2: Stage-2は勾配なし**
                stage2_metrics = self.df_state.train_stage2_closed_form()
                stage2_losses.append(stage2_metrics['stage2_loss'])
                
                # ログ記録
                self.logger.log_phase1(
                    epoch, TrainingPhase.PHASE1_DF_A, 'stage2', t,
                    stage2_metrics, {}
                )
        
        metrics['df_a_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        
        # **修正2: 明示的グラフクリア**
        self._clear_computation_graph()
        
        return metrics
    
    def _train_df_b_epoch(self, X_states: torch.Tensor, m_series: torch.Tensor, 
                         epoch: int) -> Dict[str, float]:
        """
        **修正2+4統合**: DF-B（観測層）のエポック学習（完全グラフ分離 + ヘルパー使用）
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        opt_psi = self.optimizers['psi']
        
        # **修正2: 完全にデタッチされた入力で独立グラフ作成**
        X_states_detached = X_states.detach().requires_grad_(False)
        
        # **修正2: 独立計算コンテキストでDF-B学習**
        with torch.enable_grad():
            # DF-Aからの状態予測を取得
            X_hat_states = self.df_state.predict_sequence(X_states_detached)
            
            # **修正4: ヘルパーメソッドを使用した時間インデックス調整**
            m_aligned = self._align_time_series(
                X_hat_states, m_series, X_states.size(0), epoch, "DF-B"
            )
            
            # Stage-1学習
            stage1_metrics = self.df_obs.train_stage1_with_gradients(
                X_hat_states, 
                m_aligned,  # **修正2+4: 時間調整済み**
                opt_phi, 
                T1_iterations=self.config.T1_iterations,
                fix_psi_omega=True
            )
        
        metrics['df_b_stage1_loss'] = stage1_metrics['stage1_loss']
        
        # ログ記録
        self.logger.log_phase1(
            epoch, TrainingPhase.PHASE1_DF_B, 'stage1', 0,
            stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
        )
        
        # **修正2: 明示的グラフクリア**
        self._clear_computation_graph()
        
        # Stage-2: u_B推定 + ψ_ω更新（T2回反復）
        stage2_losses = []
        for t in range(self.config.T2_iterations):
            with torch.enable_grad():
                stage2_metrics = self.df_obs.train_stage2_with_gradients(
                    m_aligned,  # **修正2+4: 時間調整済み**
                    opt_psi, 
                    fix_phi_theta=True
                )
                stage2_losses.append(stage2_metrics['stage2_loss'])
                
                # ログ記録
                self.logger.log_phase1(
                    epoch, TrainingPhase.PHASE1_DF_B, 'stage2', t,
                    stage2_metrics, {'lr_psi': opt_psi.param_groups[0]['lr']}
                )
        
        metrics['df_b_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        
        # **修正2: 最終グラフクリア**
        self._clear_computation_graph()
        
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
        **修正4統合**: Phase-2の前向き推論と損失計算（時間調整ヘルパー使用）
        """
        T, d = Y_train.shape
        h = self.realization.h
        
        if T <= 2 * h:
            # 時系列が短すぎる場合のフォールバック
            return self._handle_short_timeseries_phase2(Y_train)
        
        # Step 1: エンコード y_t → m_t
        m_series = self.encoder(Y_train.unsqueeze(0)).squeeze()  # (T, 1) -> (T,)
        if m_series.dim() == 2:
            m_series = m_series.squeeze(1)
        
        # Step 2: 確率的実現 m_t → x_t
        self.realization.fit(m_series.unsqueeze(1))
        X_states = self.realization.filter(m_series.unsqueeze(1))  # (T_eff, r)
        T_eff = X_states.size(0)
        
        # Step 3: DF-A予測 x_{t-1} → x̂_{t|t-1}
        X_hat_states = self.df_state.predict_sequence(X_states)  # (T_pred, r)
        T_pred = X_hat_states.size(0)
        
        # **修正4: ヘルパーメソッドを使用した時間調整**
        m_aligned = self._align_time_series(
            X_hat_states, m_series, X_states.size(0), 0, "Phase2"
        )
        
        # Step 4: DF-B予測 x̂_{t|t-1} → m̂_{t|t-1}
        m_hat_series = []
        for t in range(T_pred):
            m_hat_t = self.df_obs.predict_one_step(X_hat_states[t])
            m_hat_series.append(m_hat_t)
        m_hat_tensor = torch.stack(m_hat_series)  # (T_pred,)
        
        # Step 5: デコード m̂_{t|t-1} → ŷ_{t|t-1}
        Y_hat = self.decoder(m_hat_tensor.unsqueeze(0).unsqueeze(2)).squeeze(0)  # (T_pred, d)
        
        # Step 6: 対応する真値取得
        Y_target = Y_train[h+1:h+1+T_pred]  # 対応する観測
        
        # 損失計算
        loss_rec = torch.norm(Y_hat - Y_target, p='fro') ** 2
        
        # CCA損失（オプション）
        if self.config.lambda_cca > 0:
            loss_cca = self._compute_cca_loss(m_hat_tensor, m_aligned)
        else:
            loss_cca = torch.tensor(0.0, requires_grad=True)
        
        loss_total = loss_rec + self.config.lambda_cca * loss_cca
        
        return loss_total, loss_rec, loss_cca
    
    def _handle_short_timeseries_phase2(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """短い時系列用のフォールバック処理"""
        T, d = Y_train.shape
        h = self.realization.h
        warnings.warn(f"時系列長({T})が短すぎます。h={h}")
        
        # 最小限の処理
        m_series = self.encoder(Y_train.unsqueeze(0)).squeeze()
        if m_series.dim() == 2:
            m_series = m_series.squeeze(1)
        
        self.realization.fit(m_series.unsqueeze(1))
        X_states = self.realization.filter(m_series.unsqueeze(1))
        
        # 短縮処理
        if X_states.size(0) > 1:
            X_hat_states = self.df_state.predict_sequence(X_states)
            # **修正4: ヘルパー使用**
            m_aligned = self._align_time_series(
                X_hat_states, m_series, X_states.size(0), 0, "Phase2-Short"
            )
            
            loss_rec = torch.norm(m_aligned - m_aligned, p=2) ** 2  # ダミー損失
            loss_cca = torch.tensor(0.0, requires_grad=True)
            loss_total = loss_rec + self.config.lambda_cca * loss_cca
        else:
            loss_rec = torch.tensor(0.0, requires_grad=True)
            loss_cca = torch.tensor(0.0, requires_grad=True) 
            loss_total = loss_rec
        
        return loss_total, loss_rec, loss_cca
    
    def _compute_cca_loss(self, m_hat: torch.Tensor, m_target: torch.Tensor) -> torch.Tensor:
        """CCA損失計算（簡易版）"""
        # 正規化
        m_hat_norm = (m_hat - m_hat.mean()) / (m_hat.std() + 1e-8)
        m_target_norm = (m_target - m_target.mean()) / (m_target.std() + 1e-8)
        
        # 相関係数（負の値なので最小化で相関最大化）
        correlation = torch.corrcoef(torch.stack([m_hat_norm, m_target_norm]))[0, 1]
        
        return -correlation  # 相関最大化のため負号
    
    # ===== 修正4: ヘルパーメソッド群 =====
    
    def _align_time_series(self, X_hat_states: torch.Tensor, m_series: torch.Tensor, 
                          T_states: int, epoch: int, component: str) -> torch.Tensor:
        """
        **修正4**: 統合時間系列調整ヘルパー
        
        Args:
            X_hat_states: 状態予測系列
            m_series: 元のスカラー特徴系列
            T_states: 状態系列長（X_statesの長さ）
            epoch: 現在のエポック（ログ用）
            component: コンポーネント名（ログ用）
        
        Returns:
            torch.Tensor: 時間調整済みスカラー特徴系列
        """
        T_pred = X_hat_states.size(0)
        T_original = m_series.size(0)
        
        # オフセット計算
        total_offset = self._get_time_alignment_offset(T_original, T_states, T_pred)
        
        # 時間調整されたm_seriesを抽出
        if total_offset + T_pred <= T_original:
            m_aligned = m_series[total_offset:total_offset + T_pred]
        else:
            # 安全措置: 末尾から必要な長さを取得
            m_aligned = m_series[-T_pred:]
            if self.config.verbose:
                print(f"警告: {component} でオフセット調整失敗、末尾切り取り使用")
        
        # 時間整合性検証
        self._validate_time_alignment(X_hat_states, m_aligned, component)
        
        # デバッグ情報（verbose時のみ）
        if self.config.verbose and epoch % 10 == 0:
            print(f"{component}時間調整 - Epoch {epoch}: "
                  f"X_hat: {X_hat_states.shape}, m_aligned: {m_aligned.shape}, "
                  f"offset: {total_offset}")
        
        return m_aligned
    
    def _get_time_alignment_offset(self, T_original: int, T_states: int, T_pred: int) -> int:
        """
        **修正4**: 時間インデックス調整のオフセット計算
        
        Args:
            T_original: 元系列長
            T_states: 状態系列長
            T_pred: 予測系列長
            
        Returns:
            int: m_series用のオフセット
        """
        # 確率的実現パラメータから正確な計算
        try:
            h = getattr(self.realization, 'h', 
                getattr(self.realization, 'past_horizon', 0))
            if h > 0:
                # 理論値使用: 確率的実現h + 予測1ステップ
                return h + 1
            else:
                # 動的計算
                return (T_original - T_states) + (T_states - T_pred)
        except Exception:
            # フォールバック
            return (T_original - T_states) + (T_states - T_pred)
    
    def _validate_time_alignment(self, X_hat_states: torch.Tensor, m_aligned: torch.Tensor, 
                               component: str = "unknown") -> None:
        """
        **修正4**: 時間インデックス整合性の検証
        
        Args:
            X_hat_states: 状態予測
            m_aligned: 調整済みスカラー特徴量
            component: コンポーネント名（ログ用）
        """
        if X_hat_states.size(0) != m_aligned.size(0):
            raise RuntimeError(
                f"{component}の時間インデックス不整合: "
                f"X_hat={X_hat_states.shape} vs m_aligned={m_aligned.shape}"
            )
        
        if self.config.verbose:
            print(f"{component} 時間整合確認: {X_hat_states.shape} ↔ {m_aligned.shape}")
    
    def _clear_computation_graph(self):
        """
        **修正2**: 計算グラフの明示的クリア
        """
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # CPU ガベージコレクション
        gc.collect()
    
    # ===== 既存メソッド（修正なし） =====
    
    def forecast(self, Y_test: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        """予測実行"""
        self.encoder.eval()
        self.decoder.eval()
        self.df_state.eval()
        self.df_obs.eval()
        
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
    
    def train_full(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """完全学習実行（Phase-1 + Phase-2）"""
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
        model_state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'df_state': self.df_state.get_state_dict() if self.df_state else None,
            'df_obs': self.df_obs.get_state_dict() if self.df_obs else None,
            'realization_config': self.realization.__dict__,
            'training_config': self.config.__dict__
        }
        
        save_path = self.output_dir / 'final_model.pth'
        torch.save(model_state, save_path)
        
        print(f"最終モデル保存: {save_path}")
    
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
        TwoStageTrainer: 初期化済みトレーナー
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # モデル初期化
    encoder = tcnEncoder(**config['model']['encoder'])
    decoder = tcnDecoder(**config['model']['decoder'])
    realization = Realization(**config['ssm']['realization'])
    
    # 設定変換
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
    **元関数の修正版**: 学習実験の実行
    
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
    try:
        data = np.load(data_path)
        if 'Y' in data:
            Y_train = torch.tensor(data['Y'], dtype=torch.float32, device=device)
        elif 'arr_0' in data:
            Y_train = torch.tensor(data['arr_0'], dtype=torch.float32, device=device)
        else:
            available_keys = list(data.keys())
            raise ValueError(
                f"データファイルに 'Y' または 'arr_0' キーが見つかりません。"
                f"利用可能なキー: {available_keys}"
            )
    except Exception as e:
        raise RuntimeError(f"データ読み込み失敗: {data_path}. エラー: {e}")
    
    print(f"データ読み込み完了: {Y_train.shape}")
    
    # データ検証
    if Y_train.dim() != 2:
        raise ValueError(f"データは2次元 (T, d) である必要があります: got {Y_train.shape}")
    
    T, d = Y_train.shape
    if T < 50:
        warnings.warn(f"時系列長が短すぎる可能性があります: T={T}")
    
    # トレーナー作成
    try:
        trainer = create_trainer_from_config(config_path, device, output_dir)
    except Exception as e:
        raise RuntimeError(f"トレーナー作成失敗: {config_path}. エラー: {e}")
    
    # **修正**: train_full メソッドを使用（元のfit→train_fullに変更）
    try:
        results = trainer.train_full(Y_train)
    except Exception as e:
        print(f"学習中にエラーが発生: {e}")
        # 緊急保存試行
        try:
            trainer._save_checkpoint(
                trainer.current_epoch, 
                TrainingPhase.PHASE1_DF_A, 
                emergency=True
            )
            print(f"緊急チェックポイント保存: {trainer.output_dir}")
        except:
            print("緊急保存も失敗しました")
        raise
    
    # **修正**: サマリ追加（修正版メソッド名に対応）
    try:
        experiment_summary = trainer.get_training_summary()
        results['experiment_summary'] = experiment_summary
        results['data_info'] = {
            'data_path': data_path,
            'data_shape': tuple(Y_train.shape),
            'device': str(device),
            'total_parameters': experiment_summary.get('model_info', {}).get('total_params', 0)
        }
        
        # **追加**: 設定ファイルのバックアップ
        config_backup_path = Path(output_dir) / 'config_used.yaml'
        if not config_backup_path.exists():
            import shutil
            shutil.copy2(config_path, config_backup_path)
            print(f"設定ファイルをバックアップ: {config_backup_path}")
            
    except Exception as e:
        warnings.warn(f"サマリ作成でエラー: {e}")
        results['experiment_summary'] = {'error': str(e)}
        results['data_info'] = {
            'data_path': data_path,
            'data_shape': tuple(Y_train.shape),
            'device': str(device)
        }
    
    print(f"実験完了: 結果は {output_dir} に保存されました")
    
    return results


def run_validation(
    trainer: TwoStageTrainer, 
    Y_test: torch.Tensor, 
    output_dir: str,
    forecast_steps: int = 96
) -> Dict[str, Any]:
    """
    **新機能**: 学習済みモデルの検証実行
    
    Args:
        trainer: 学習済みトレーナー
        Y_test: テストデータ
        output_dir: 結果出力ディレクトリ
        forecast_steps: 予測ステップ数
        
    Returns:
        検証結果辞書
    """
    print("検証開始...")
    
    try:
        # 予測実行
        predictions = trainer.forecast(Y_test, forecast_steps)
        
        # 予測精度計算
        if Y_test.size(0) > forecast_steps:
            Y_true = Y_test[-forecast_steps:]
            mse = torch.mean((predictions - Y_true) ** 2).item()
            mae = torch.mean(torch.abs(predictions - Y_true)).item()
            
            # 相対誤差
            relative_error = (torch.norm(predictions - Y_true) / torch.norm(Y_true)).item()
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': mse ** 0.5,
                'relative_error': relative_error,
                'forecast_steps': forecast_steps
            }
        else:
            warnings.warn("テストデータが予測ステップ数より短いため、精度計算をスキップ")
            metrics = {
                'forecast_steps': forecast_steps,
                'note': 'テストデータ不足のため精度計算不可'
            }
        
        # 結果保存
        validation_results = {
            'metrics': metrics,
            'predictions_shape': tuple(predictions.shape),
            'test_data_shape': tuple(Y_test.shape),
            'model_summary': trainer.get_training_summary()
        }
        
        # 予測結果をnumpy配列として保存
        output_path = Path(output_dir)
        predictions_path = output_path / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=predictions.cpu().numpy(),
            Y_test=Y_test.cpu().numpy()
        )
        
        print(f"検証完了: 精度指標 MSE={metrics.get('mse', 'N/A'):.6f}")
        
        return validation_results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'test_data_shape': tuple(Y_test.shape),
            'forecast_steps': forecast_steps
        }
        print(f"検証中にエラー: {e}")
        return error_result


def plot_training_results(output_dir: str) -> None:
    """
    **新機能**: 学習結果の可視化
    
    Args:
        output_dir: 結果ディレクトリ
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        output_path = Path(output_dir)
        
        # Phase-1 損失プロット
        phase1_csv = output_path / 'phase1_training.csv'
        if phase1_csv.exists():
            df_phase1 = pd.read_csv(phase1_csv)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Phase-1 Training Progress')
            
            # DF-A Stage-1
            df_a_s1 = df_phase1[(df_phase1['phase'] == 'phase1_df_a') & (df_phase1['stage'] == 'stage1')]
            if not df_a_s1.empty:
                axes[0, 0].plot(df_a_s1['epoch'], df_a_s1['loss'])
                axes[0, 0].set_title('DF-A Stage-1 Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
            
            # DF-A Stage-2
            df_a_s2 = df_phase1[(df_phase1['phase'] == 'phase1_df_a') & (df_phase1['stage'] == 'stage2')]
            if not df_a_s2.empty:
                axes[0, 1].plot(df_a_s2['epoch'], df_a_s2['loss'])
                axes[0, 1].set_title('DF-A Stage-2 Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
            
            # DF-B Stage-1
            df_b_s1 = df_phase1[(df_phase1['phase'] == 'phase1_df_b') & (df_phase1['stage'] == 'stage1')]
            if not df_b_s1.empty:
                axes[1, 0].plot(df_b_s1['epoch'], df_b_s1['loss'])
                axes[1, 0].set_title('DF-B Stage-1 Loss')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
            
            # DF-B Stage-2
            df_b_s2 = df_phase1[(df_phase1['phase'] == 'phase1_df_b') & (df_phase1['stage'] == 'stage2')]
            if not df_b_s2.empty:
                axes[1, 1].plot(df_b_s2['epoch'], df_b_s2['loss'])
                axes[1, 1].set_title('DF-B Stage-2 Loss')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(output_path / 'phase1_losses.png', dpi=150)
            plt.close()
        
        # Phase-2 損失プロット
        phase2_csv = output_path / 'phase2_training.csv'
        if phase2_csv.exists():
            df_phase2 = pd.read_csv(phase2_csv)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Phase-2 Training Progress')
            
            axes[0].plot(df_phase2['epoch'], df_phase2['total_loss'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            
            axes[1].plot(df_phase2['epoch'], df_phase2['rec_loss'])
            axes[1].set_title('Reconstruction Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            
            axes[2].plot(df_phase2['epoch'], df_phase2['cca_loss'])
            axes[2].set_title('CCA Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(output_path / 'phase2_losses.png', dpi=150)
            plt.close()
        
        print(f"可視化完了: {output_path}")
        
    except ImportError:
        warnings.warn("matplotlib/pandasが利用できないため、可視化をスキップ")
    except Exception as e:
        warnings.warn(f"可視化中にエラー: {e}")


if __name__ == "__main__":
    # 簡単なテスト用コード
    print("TwoStageTrainer修正版読み込み完了")