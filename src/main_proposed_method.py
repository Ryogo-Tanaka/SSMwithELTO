#!/usr/bin/env python3
# main_proposed_method.py
"""
提案手法の完全実装: DF-A + DF-B + 2段階学習戦略

計算フロー:
1. TCNエンコーダ: {y_t} → {m_t} (スカラー特徴系列)
2. 確率的実現: {m_t} → {x_t} (状態系列)  
3. DF-A (State Layer): {x_t} で2SLS学習
4. DF-B (Observation Layer): {x̂_t, m_t} で2SLS学習
5. TCNデコーダ: {m̂_t} → {ŷ_t}
6. Phase-2: End-to-end微調整
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import csv

# 既存コンポーネントのインポート
from src.ssm.df_state_layer import DFStateLayer
from src.ssm.df_observation_layer import DFObservationLayer  
from src.ssm.realization import Realization
from src.models.architectures.tcn import tcnEncoder, tcnDecoder
from src.utils.gpu_utils import select_device


class ProposedMethodTrainer:
    """
    提案手法の2段階学習を実行するメインクラス
    
    Phase-1: DF-A/DF-B の Stage-1/Stage-2 交互学習
    Phase-2: End-to-end 微調整
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.cfg = config
        self.device = device
        
        # Phase-1 設定
        self.T1 = config['training']['phase1']['T1_iterations']  # Stage-1 反復数
        self.T2 = config['training']['phase1']['T2_iterations']  # Stage-2 反復数
        self.phase1_epochs = config['training']['phase1']['epochs']
        
        # Phase-2 設定
        self.phase2_epochs = config['training']['phase2']['epochs']
        self.lambda_c = config['training']['phase2']['lambda_cca']
        
        # 学習率
        self.lr_phi = config['training']['lr_phi']      # φ_θ 学習率
        self.lr_psi = config['training']['lr_psi']      # ψ_ω 学習率  
        self.lr_encoder = config['training']['lr_encoder']
        self.lr_decoder = config['training']['lr_decoder']
        
        self._build_models()
        self._build_optimizers()
        
        # 学習状態
        self._phase1_complete = False
        self._history = {'losses': [], 'phase1_metrics': []}
        
    def _build_models(self):
        """モデル初期化"""
        # 1. TCNエンコーダ (multivariate → scalar)
        enc_cfg = self.cfg['model']['encoder']
        self.encoder = tcnEncoder(
            input_dim=enc_cfg['input_dim'],
            output_dim=1,  # スカラー特徴量
            channels=enc_cfg.get('channels', 64),
            layers=enc_cfg.get('layers', 6),
            kernel_size=enc_cfg.get('kernel_size', 3),
            activation=enc_cfg.get('activation', 'GELU'),
            dropout=enc_cfg.get('dropout', 0.0)
        ).to(self.device)
        
        # 2. 確率的実現 (スカラー特徴量 → 状態)
        real_cfg = self.cfg['ssm']['realization']
        self.realization = Realization(**real_cfg)
        
        # 3. DF-A (状態層)
        df_a_cfg = self.cfg['ssm']['df_state']
        self.df_state = DFStateLayer(
            state_dim=real_cfg.get('rank', 5),  # 実現で推定された状態次元
            feature_dim=df_a_cfg['feature_dim'],
            lambda_A=df_a_cfg['lambda_A'],
            lambda_B=df_a_cfg['lambda_B'],
            feature_net_config=df_a_cfg.get('feature_net', {}),
            cross_fitting_config=df_a_cfg.get('cross_fitting', {})
        )
        
        # 4. DF-B (観測層) 
        df_b_cfg = self.cfg['ssm']['df_observation']
        self.df_obs = DFObservationLayer(
            df_state_layer=None,  # 後で設定
            obs_feature_dim=df_b_cfg['obs_feature_dim'],
            lambda_B=df_b_cfg['lambda_B'],
            lambda_dB=df_b_cfg['lambda_dB'],
            obs_net_config=df_b_cfg.get('obs_net', {}),
            cross_fitting_config=df_b_cfg.get('cross_fitting', {})
        )
        
        # 5. TCNデコーダ (scalar → multivariate)
        dec_cfg = self.cfg['model']['decoder']
        self.decoder = tcnDecoder(
            output_dim=dec_cfg['output_dim'],
            window=dec_cfg.get('window', 8),
            tau=dec_cfg.get('tau', 1),
            hidden=dec_cfg.get('hidden', 64),
            ma_kernel=dec_cfg.get('ma_kernel', 64),
            gru_hidden=dec_cfg.get('gru_hidden', 64),
            activation=dec_cfg.get('activation', 'GELU'),
            dropout=dec_cfg.get('dropout', 0.0)
        ).to(self.device)
        
    def _build_optimizers(self):
        """最適化器初期化"""
        # Phase-1用 (個別パラメータ)
        self.opt_phi = torch.optim.Adam(
            self.df_state.phi_theta.parameters(), 
            lr=self.lr_phi
        )
        self.opt_psi = torch.optim.Adam(
            self.df_obs.psi_omega.parameters(),
            lr=self.lr_psi
        )
        
        # Phase-2用 (End-to-end)
        self.opt_e2e = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.lr_encoder},
            {'params': self.decoder.parameters(), 'lr': self.lr_decoder},
            {'params': self.df_state.phi_theta.parameters(), 'lr': self.lr_phi},
            {'params': self.df_obs.psi_omega.parameters(), 'lr': self.lr_psi}
        ])
        
    def fit(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        提案手法の完全学習
        
        Args:
            Y_train: 訓練観測系列 (T, d)
            Y_val: 検証観測系列 (T_val, d) [optional]
            
        Returns:
            学習履歴辞書
        """
        print("=== 提案手法の2段階学習開始 ===")
        
        # Phase-1: DF学習
        self._phase1_training(Y_train)
        
        # Phase-2: End-to-end微調整
        self._phase2_training(Y_train, Y_val)
        
        return self._history
        
    def _phase1_training(self, Y_train: torch.Tensor):
        """
        Phase-1: DF-A/DF-B の交互学習
        """
        print("\n--- Phase-1: DF学習 ---")
        T, d = Y_train.shape
        
        # 1. エンコードして状態系列を取得
        with torch.no_grad():
            m_series = self.encoder(Y_train.unsqueeze(0)).squeeze(0)  # (T, 1) → (T,)
            if m_series.dim() == 2:
                m_series = m_series.squeeze(1)
                
        # 2. 確率的実現で状態推定
        self.realization.fit(m_series.unsqueeze(1))  # (T,) → (T, 1) for realization
        X_states = self.realization.filter(m_series.unsqueeze(1))  # (T_eff, r)
        
        print(f"状態系列形状: {X_states.shape}")
        print(f"スカラー特徴系列形状: {m_series.shape}")
        
        # Phase-1 エポックループ
        for epoch in range(self.phase1_epochs):
            epoch_metrics = {}
            
            # DF-A 学習
            df_a_metrics = self._train_df_a(X_states, epoch)
            epoch_metrics.update({f'df_a_{k}': v for k, v in df_a_metrics.items()})
            
            # DF-B 学習 (DF-Aが学習済みになった後)
            if epoch >= 5:  # DF-Aが安定してから
                df_b_metrics = self._train_df_b(X_states, m_series, epoch)
                epoch_metrics.update({f'df_b_{k}': v for k, v in df_b_metrics.items()})
            
            # ログ
            self._history['phase1_metrics'].append(epoch_metrics)
            if epoch % 10 == 0:
                print(f"Phase-1 Epoch {epoch}: {epoch_metrics}")
                
        self._phase1_complete = True
        print("Phase-1 完了")
        
    def _train_df_a(self, X_states: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        DF-A (状態層) の学習
        
        Stage-1: V_A推定 + φ_θ更新を T1 回反復
        Stage-2: U_A推定を T2 回実行 (閉形式解のみ)
        """
        metrics = {}
        
        # Stage-1: V_A推定 + φ_θ更新
        for t in range(self.T1):
            self.opt_phi.zero_grad()
            
            # V_A^{(-k)} を閉形式解で計算（クロスフィッティング）
            # これは既存のDFStateLayer.fit_two_stage()内部で行われる
            
            # φ_θ 更新のための勾配計算
            # L1 = Frobenius loss of Stage-1 prediction
            loss_stage1 = self._compute_df_a_stage1_loss(X_states)
            loss_stage1.backward()
            self.opt_phi.step()
            
            if t == 0:  # 最初の反復のみ記録
                metrics['stage1_loss'] = loss_stage1.item()
        
        # Stage-2: U_A推定 (閉形式解のみ、勾配なし)
        with torch.no_grad():
            # DFStateLayerの内部でU_A更新は閉形式解で実行される
            self.df_state.fit_two_stage(X_states, use_cross_fitting=True, verbose=False)
            
        return metrics
        
    def _compute_df_a_stage1_loss(self, X_states: torch.Tensor) -> torch.Tensor:
        """
        DF-A Stage-1 損失計算
        
        L1 = ||Φ^+ - V_A Φ^-||_F^2 + 正則化
        """
        # 特徴量計算
        phi_seq = self.df_state.phi_theta(X_states)  # (T, d_A)
        
        # 過去/未来分割
        phi_minus = phi_seq[:-1]  # (T-1, d_A)
        phi_plus = phi_seq[1:]    # (T-1, d_A)
        
        # 簡単化のため：全データでV_A推定（実際はクロスフィッティング）
        V_A = self.df_state._ridge_stage1(phi_minus, phi_plus, self.df_state.lambda_A)
        
        # 予測誤差
        phi_pred = (V_A @ phi_minus.T).T  # (T-1, d_A)
        loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2
        
        return loss
        
    def _train_df_b(self, X_states: torch.Tensor, m_series: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        DF-B (観測層) の学習
        
        Stage-1: V_B推定 + φ_θ更新を T1 回反復 (ψ_ω固定)
        Stage-2: u_B推定 + ψ_ω更新を T2 回反復 (φ_θ固定)
        """
        metrics = {}
        
        # DF-Aからの状態予測を取得
        X_hat_states = self.df_state.predict_sequence(X_states)  # (T-1, r)
        
        # Stage-1: V_B推定 + φ_θ更新 (ψ_ω固定)
        for t in range(self.T1):
            self.opt_phi.zero_grad()
            
            # V_B 計算 + φ_θ 更新
            loss_stage1 = self._compute_df_b_stage1_loss(X_hat_states, m_series)
            loss_stage1.backward()
            self.opt_phi.step()
            
            if t == 0:
                metrics['stage1_loss'] = loss_stage1.item()
        
        # Stage-2: u_B推定 + ψ_ω更新 (φ_θ固定)  
        for t in range(self.T2):
            self.opt_psi.zero_grad()
            
            # u_B 計算 + ψ_ω 更新
            loss_stage2 = self._compute_df_b_stage2_loss(X_hat_states, m_series)
            loss_stage2.backward()
            self.opt_psi.step()
            
            if t == 0:
                metrics['stage2_loss'] = loss_stage2.item()
                
        return metrics
        
    def _compute_df_b_stage1_loss(self, X_hat_states: torch.Tensor, m_series: torch.Tensor) -> torch.Tensor:
        """DF-B Stage-1 損失計算"""
        # 操作変数特徴量（状態予測から）
        phi_instrument = self.df_state.phi_theta(X_hat_states)  # (T-1, d_A)
        
        # 観測特徴量（固定のψ_ω）
        with torch.no_grad():
            psi_obs = self.df_obs.psi_omega(m_series[1:].unsqueeze(1))  # (T-1, d_B)
        
        # V_B推定
        V_B = self.df_obs._ridge_stage1_vb(phi_instrument, psi_obs, self.df_obs.lambda_B)
        
        # 予測誤差
        psi_pred = (V_B @ phi_instrument.T).T  # (T-1, d_B)
        loss = torch.norm(psi_pred - psi_obs, p='fro') ** 2
        
        return loss
        
    def _compute_df_b_stage2_loss(self, X_hat_states: torch.Tensor, m_series: torch.Tensor) -> torch.Tensor:
        """DF-B Stage-2 損失計算"""
        # 固定のφ_θで操作変数特徴量
        with torch.no_grad():
            phi_instrument = self.df_state.phi_theta(X_hat_states)  # (T-1, d_A)
            # 簡略化V_B計算
            psi_obs = self.df_obs.psi_omega(m_series[1:].unsqueeze(1))
            V_B = self.df_obs._ridge_stage1_vb(phi_instrument, psi_obs, self.df_obs.lambda_B)
            H = (V_B @ phi_instrument.T).T  # (T-1, d_B)
        
        # u_B推定
        u_B = self.df_obs._ridge_stage2_ub(H, m_series[1:], self.df_obs.lambda_dB)
        
        # 予測誤差
        m_pred = (H * u_B).sum(dim=1)  # (T-1,)
        loss = torch.norm(m_pred - m_series[1:], p=2) ** 2
        
        return loss
        
    def _phase2_training(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None):
        """
        Phase-2: End-to-end 微調整
        
        固定推論パス:
        x̂_{t|t-1} = U_A^T V_A φ_θ(x_{t-1})
        m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})  
        ŷ_{t|t-1} = g_α(m̂_{t|t-1})
        """
        print("\n--- Phase-2: End-to-end微調整 ---")
        
        if not self._phase1_complete:
            raise RuntimeError("Phase-1が完了していません")
            
        # DF-B初期化（DF-Aの結果を使用）
        self.df_obs.df_state = self.df_state
        
        # Phase-2エポックループ
        for epoch in range(self.phase2_epochs):
            self.opt_e2e.zero_grad()
            
            # 前向き推論パス
            loss_total = self._forward_and_loss_phase2(Y_train)
            
            # 逆伝播
            loss_total.backward()
            self.opt_e2e.step()
            
            # ログ
            self._history['losses'].append(loss_total.item())
            if epoch % 10 == 0:
                print(f"Phase-2 Epoch {epoch}: Loss = {loss_total.item():.6f}")
                
        print("Phase-2 完了")
        
    def _forward_and_loss_phase2(self, Y_train: torch.Tensor) -> torch.Tensor:
        """
        Phase-2の前向き推論と損失計算
        """
        T, d = Y_train.shape
        
        # 1. エンコード: y_t → m_t
        m_series = self.encoder(Y_train.unsqueeze(0)).squeeze(0)  # (T, 1)
        if m_series.dim() == 2:
            m_series = m_series.squeeze(1)  # (T,)
            
        # 2. 確率的実現: m_t → x_t
        self.realization.fit(m_series.unsqueeze(1))
        X_states = self.realization.filter(m_series.unsqueeze(1))  # (T_eff, r)
        
        # 3. DF-A予測: x_{t-1} → x̂_{t|t-1}
        X_hat_states = self.df_state.predict_sequence(X_states)  # (T_eff-1, r)
        
        # 4. DF-B予測: x̂_{t|t-1} → m̂_{t|t-1}
        m_hat_series = []
        for t in range(X_hat_states.size(0)):
            m_hat_t = self.df_obs.predict_one_step(X_hat_states[t])
            m_hat_series.append(m_hat_t)
        m_hat_tensor = torch.stack(m_hat_series)  # (T_eff-1,)
        
        # 5. デコード: m̂_{t|t-1} → ŷ_{t|t-1}
        Y_hat = self.decoder(m_hat_tensor.unsqueeze(0).unsqueeze(2)).squeeze(0)  # (T_eff-1, d)
        
        # 6. 損失計算
        # 時間合わせ
        h = self.realization.h
        Y_target = Y_train[h+1:h+1+Y_hat.size(0), :]  # (T_eff-1, d)
        
        # 再構成損失
        L_rec = torch.norm(Y_hat - Y_target, p='fro') ** 2
        
        # 正準相関損失 (簡略化)
        L_cca = torch.tensor(0.0, device=self.device)  # TODO: 実装
        
        # 総損失
        L_total = L_rec + self.lambda_c * L_cca
        
        return L_total
        
    def predict(self, Y_test: torch.Tensor, forecast_steps: int = 1) -> torch.Tensor:
        """
        予測実行
        
        Args:
            Y_test: テスト観測系列 (T_test, d)
            forecast_steps: 予測ステップ数
            
        Returns:
            予測系列 (forecast_steps, d)
        """
        if not self._phase1_complete:
            raise RuntimeError("学習が完了していません")
            
        # 実装: 逐次予測ロジック
        # TODO: 実装詳細
        pass
        
    def save_model(self, save_path: str):
        """モデル保存"""
        state_dict = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'df_state': self.df_state.get_state_dict(),
            'df_obs': self.df_obs.get_state_dict(),
            'realization_config': self.cfg['ssm']['realization'],
            'training_history': self._history
        }
        torch.save(state_dict, save_path)
        print(f"モデル保存完了: {save_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="提案手法の完全実装")
    parser.add_argument('--config', type=str, required=True, help='設定ファイルパス')
    parser.add_argument('--data-path', type=str, required=True, help='データファイルパス')
    parser.add_argument('--output-dir', type=str, default='results/proposed_method', help='出力ディレクトリ')
    parser.add_argument('--device', type=str, default=None, help='計算デバイス')
    args = parser.parse_args()
    
    # 設定読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # デバイス設定
    device = torch.device(args.device) if args.device else select_device()
    print(f"Using device: {device}")
    
    # 出力ディレクトリ準備
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    data = np.load(args.data_path)
    if 'Y' in data:
        Y_train = torch.tensor(data['Y'], dtype=torch.float32, device=device)
    elif 'arr_0' in data:
        Y_train = torch.tensor(data['arr_0'], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"データファイルに 'Y' または 'arr_0' キーが見つかりません: {list(data.keys())}")
    
    print(f"データ形状: {Y_train.shape}")
    
    # 訓練実行
    trainer = ProposedMethodTrainer(config, device)
    history = trainer.fit(Y_train)
    
    # 結果保存
    trainer.save_model(output_dir / 'model.pth')
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"結果保存完了: {output_dir}")


if __name__ == '__main__':
    main()