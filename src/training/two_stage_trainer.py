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
    U_B = 閉形式解(H^{(cf)}_B, M)        # U_B計算（ϕ_θ固定）
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
from ..ssm.realization import Realization, StochasticRealizationWithEncoder, RealizationError
from ..models.architectures.time_invariant import time_invariantEncoder, time_invariantDecoder


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
    log_interval: int = 5    # ログ出力間隔（エポック）
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

    @classmethod
    def from_nested_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """入れ子設定辞書から平坦なTrainingConfigを作成"""
        # デフォルト値を設定
        phase1_config = config_dict.get('phase1', {})
        phase2_config = config_dict.get('phase2', {})
        
        return cls(
            # Phase-1設定
            phase1_epochs=phase1_config.get('epochs', 50),
            T1_iterations=phase1_config.get('T1_iterations', 10),
            T2_iterations=phase1_config.get('T2_iterations', 5),
            df_a_warmup_epochs=phase1_config.get('df_a', {}).get('warmup_epochs', 5),
            
            # Phase-2設定
            phase2_epochs=phase2_config.get('epochs', 100),
            lambda_cca=phase2_config.get('lambda_cca', 0.1),
            update_strategy=phase2_config.get('update_strategy', "all"),
            
            # 学習率（Phase-2内またはトップレベルから取得）
            lr_phi=phase2_config.get('lr_phi', phase1_config.get('df_a', {}).get('lr', 1e-3)),
            lr_psi=phase2_config.get('lr_psi', phase1_config.get('df_b', {}).get('lr', 1e-3)),
            lr_encoder=phase2_config.get('lr_encoder', 1e-4),
            lr_decoder=phase2_config.get('lr_decoder', 1e-4),
            
            # ログ・保存設定（トップレベルから）
            log_interval=config_dict.get('log_interval', 5),
            save_interval=config_dict.get('checkpoint', {}).get('save_every', 10),
            verbose=config_dict.get('verbose', True)
        )


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
    
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None, realization: Realization = None,
                 df_state_config: Dict[str, Any] = None, df_obs_config: Dict[str, Any] = None,
                 training_config: TrainingConfig = None, device: torch.device = None, output_dir: str = None,
                 use_kalman_filtering: bool = True,
                calibration_ratio: float = 0.1,
                auto_inference_setup: bool = True,
                config: Dict[str, Any] = None):
        
        # config引数が渡された場合は設定から初期化
        if config is not None:
            self._init_from_config(config, device, output_dir, use_kalman_filtering)
        else:
            # 従来の個別引数からの初期化
            self._init_from_args(encoder, decoder, realization, df_state_config, df_obs_config,
                               training_config, device, output_dir, use_kalman_filtering,
                               calibration_ratio, auto_inference_setup)
    
    def _init_from_config(self, config: Dict[str, Any], device: torch.device, output_dir: str,
                         use_kalman_filtering: bool):
        """設定辞書からの初期化"""
        # モデル初期化（新しいtime_invariantアーキテクチャ使用）
        encoder_config = config['model']['encoder'].copy()
        decoder_config = config['model']['decoder'].copy()

        # time_invariant用の設定調整
        if 'type' not in encoder_config:
            encoder_config['type'] = 'time_invariant'
        if 'output_dim' not in encoder_config:
            encoder_config['output_dim'] = encoder_config.get('channels', 32)

        encoder = time_invariantEncoder(**encoder_config)
        decoder = time_invariantDecoder(**decoder_config)

        # 確率実現（新しいクラス優先）
        realization_config = config['ssm']['realization']
        if config.get('evaluation', {}).get('use_new_realization', True):
            # StochasticRealizationWithEncoderはencoder引数が必要
            realization_config_copy = realization_config.copy()
            realization = StochasticRealizationWithEncoder(
                encoder=encoder,
                **realization_config_copy
            )
        else:
            realization = Realization(**realization_config)
        
        # 設定変換
        training_config = TrainingConfig.from_nested_dict(config['training'])
        
        # 個別引数での初期化に委譲
        calibration_ratio = config['training'].get('calibration_ratio', 0.25)
        auto_inference_setup = config['training'].get('auto_inference_setup', True)

        self._init_from_args(encoder, decoder, realization,
                           config['ssm']['df_state'], config['ssm']['df_observation'],
                           training_config, device, output_dir, use_kalman_filtering,
                           calibration_ratio, auto_inference_setup)

    @classmethod
    def from_trained_model(cls, model_path: str, device: torch.device = None,
                          output_dir: str = None) -> 'TwoStageTrainer':
        """学習済みモデルから推論専用インスタンス作成（設定ファイル不要）"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if output_dir is None:
            output_dir = 'temp_inference'

        # 学習済みモデル読み込み
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # パラメータから構造検出
        encoder_config = cls._detect_encoder_structure(state_dict.get('encoder', {}))
        decoder_config = cls._detect_decoder_structure(state_dict.get('decoder', {}))

        # 検出された構造でモデル初期化（新しいアーキテクチャ使用）
        encoder = time_invariantEncoder(**encoder_config)
        decoder = time_invariantDecoder(**decoder_config)

        # 最小限の設定で初期化
        realization = Realization(past_horizon=10, rank=3)
        df_state_config = {'feature_dim': 16}
        df_obs_config = {'obs_feature_dim': 8}
        training_config = TrainingConfig()

        # インスタンス作成
        instance = cls._init_from_args_direct(
            encoder, decoder, realization, df_state_config, df_obs_config,
            training_config, device, output_dir, use_kalman_filtering=False
        )

        # 重みを読み込み
        instance.encoder.load_state_dict(state_dict.get('encoder', {}))
        instance.decoder.load_state_dict(state_dict.get('decoder', {}))

        return instance

    @classmethod
    def _detect_encoder_structure(cls, encoder_dict: Dict[str, Any]) -> Dict[str, Any]:
        """エンコーダパラメータから構造検出（time_invariant対応、汎用的）"""
        input_dim = 6  # デフォルト
        output_dim = 32  # デフォルト
        architecture = 'mlp'  # デフォルト
        hidden_dims = [64, 32]  # デフォルト

        # 構造の自動検出
        if not encoder_dict:
            # 空の場合はデフォルト設定
            pass
        elif 'core_net.0.weight' in encoder_dict:
            # time_invariantアーキテクチャの検出
            architecture = 'mlp'

            # 入力次元（core_netの最初の層）
            input_dim = encoder_dict['core_net.0.weight'].shape[1]

            # output_dim（output_meanから検出）
            if 'output_mean' in encoder_dict:
                output_dim = encoder_dict['output_mean'].shape[0]
            else:
                # core_netの最終層から推定
                for key in encoder_dict.keys():
                    if key.startswith('core_net.') and key.endswith('.weight'):
                        output_dim = encoder_dict[key].shape[0]

            # 隠れ層次元の検出
            hidden_dims = []
            layer_keys = [k for k in encoder_dict.keys() if k.startswith('core_net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for i, key in enumerate(layer_keys[:-1]):  # 最終層以外
                hidden_dims.append(encoder_dict[key].shape[0])

        elif 'layers.0.weight' in encoder_dict:
            # MLP構造の検出
            architecture = 'mlp'

            # 入力次元（最初の線形層）
            input_dim = encoder_dict['layers.0.weight'].shape[1]

            # 隠れ層次元とoutput_dimの検出
            hidden_dims = []
            max_layer_idx = -1

            for key in encoder_dict.keys():
                if key.startswith('layers.') and key.endswith('.weight'):
                    try:
                        layer_idx = int(key.split('.')[1])
                        max_layer_idx = max(max_layer_idx, layer_idx)

                        # 隠れ層の出力次元を収集
                        if layer_idx > 0:  # 最初の層以外
                            layer_output_dim = encoder_dict[key].shape[0]
                            if layer_idx == max_layer_idx:
                                output_dim = layer_output_dim  # 最終層が出力次元
                            else:
                                hidden_dims.append(layer_output_dim)
                    except (ValueError, IndexError):
                        continue

            if not hidden_dims:
                hidden_dims = [64, 32]  # デフォルト

        elif any(key.startswith('tcn.') for key in encoder_dict.keys()):
            # 旧TCN構造（後方互換）- time_invariantにマッピング
            architecture = 'mlp'  # time_invariantのmlpにマッピング

            if 'in_proj.weight' in encoder_dict:
                input_dim = encoder_dict['in_proj.weight'].shape[1]
                output_dim = encoder_dict['in_proj.weight'].shape[0]

        elif any(key.startswith('conv') for key in encoder_dict.keys()):
            # CNN系構造
            architecture = 'resnet'  # time_invariantのresnetにマッピング

            # 最初のconv層から入力次元推定
            first_conv_keys = [k for k in encoder_dict.keys() if 'conv' in k and 'weight' in k]
            if first_conv_keys:
                first_key = sorted(first_conv_keys)[0]
                # conv層の場合、入力チャンネル数を取得
                conv_shape = encoder_dict[first_key].shape
                if len(conv_shape) >= 2:
                    input_dim = conv_shape[1] if len(conv_shape) == 4 else conv_shape[1]
        else:
            # その他の構造 - 最小限の推定
            weight_keys = [k for k in encoder_dict.keys() if 'weight' in k]
            if weight_keys:
                # 最初のweight層から推定
                first_weight = encoder_dict[weight_keys[0]]
                if len(first_weight.shape) >= 2:
                    input_dim = first_weight.shape[1]
                    output_dim = first_weight.shape[0]

        return {
            'type': 'time_invariant',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'architecture': architecture,
            'hidden_dims': hidden_dims,
            'activation': 'GELU',
            'dropout': 0.1,
            'normalize_input': True,
            'normalize_output': True
        }

    @classmethod
    def _detect_decoder_structure(cls, decoder_dict: Dict[str, Any]) -> Dict[str, Any]:
        """デコーダパラメータから構造検出（time_invariant対応、汎用的）"""
        input_dim = 32  # デフォルト（エンコーダー出力と一致想定）
        output_dim = 6  # デフォルト
        architecture = 'mlp'  # デフォルト
        hidden_dims = [64, 32]  # デフォルト

        # 構造の自動検出
        if not decoder_dict:
            # 空の場合はデフォルト設定
            pass
        elif 'net.0.weight' in decoder_dict:
            # time_invariantDecoderアーキテクチャの検出
            architecture = 'mlp'

            # 入力次元（netの最初の層）
            input_dim = decoder_dict['net.0.weight'].shape[1]

            # 出力次元（netの最終層）
            output_dim = 6  # デフォルト
            for key in decoder_dict.keys():
                if key.startswith('net.') and key.endswith('.weight'):
                    output_dim = decoder_dict[key].shape[0]

            # 隠れ層次元の検出
            hidden_dims = []
            layer_keys = [k for k in decoder_dict.keys() if k.startswith('net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for key in layer_keys[:-1]:  # 最終層以外
                hidden_dims.append(decoder_dict[key].shape[0])

        elif 'layers.0.weight' in decoder_dict:
            # MLP構造の検出
            architecture = 'mlp'

            # 入力次元（最初の線形層）
            input_dim = decoder_dict['layers.0.weight'].shape[1]

            # 隠れ層次元とoutput_dimの検出
            hidden_dims = []
            max_layer_idx = -1

            for key in decoder_dict.keys():
                if key.startswith('layers.') and key.endswith('.weight'):
                    try:
                        layer_idx = int(key.split('.')[1])
                        max_layer_idx = max(max_layer_idx, layer_idx)

                        # 隠れ層の出力次元を収集
                        if layer_idx > 0:  # 最初の層以外
                            layer_output_dim = decoder_dict[key].shape[0]
                            if layer_idx == max_layer_idx:
                                output_dim = layer_output_dim  # 最終層が出力次元
                            else:
                                hidden_dims.append(layer_output_dim)
                    except (ValueError, IndexError):
                        continue

            if not hidden_dims:
                hidden_dims = [64, 32]  # デフォルト

        elif 'out_proj.weight' in decoder_dict:
            # 旧TCN系構造（後方互換）- time_invariantにマッピング
            architecture = 'mlp'
            output_dim = decoder_dict['out_proj.weight'].shape[0]

            if 'takens_proj.weight' in decoder_dict:
                input_dim = decoder_dict['takens_proj.weight'].shape[1]

        elif any(key.startswith('conv') for key in decoder_dict.keys()):
            # CNN系構造
            architecture = 'resnet'

            # conv層から次元推定
            conv_keys = [k for k in decoder_dict.keys() if 'conv' in k and 'weight' in k]
            if conv_keys:
                # 最後のconv層から出力次元推定
                last_key = sorted(conv_keys)[-1]
                conv_shape = decoder_dict[last_key].shape
                if len(conv_shape) >= 2:
                    output_dim = conv_shape[0] if len(conv_shape) == 4 else conv_shape[0]
        else:
            # その他の構造 - 最小限の推定
            weight_keys = [k for k in decoder_dict.keys() if 'weight' in k]
            if weight_keys:
                # 最後のweight層から出力次元推定
                last_weight = decoder_dict[weight_keys[-1]]
                if len(last_weight.shape) >= 2:
                    output_dim = last_weight.shape[0]
                    input_dim = last_weight.shape[1]

        return {
            'type': 'time_invariant',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'architecture': architecture,
            'hidden_dims': hidden_dims,
            'activation': 'GELU',
            'dropout': 0.1,
            'normalize_input': True,
            'normalize_output': True
        }

    @classmethod
    def _init_from_args_direct(cls, encoder, decoder, realization, df_state_config,
                              df_obs_config, training_config, device, output_dir,
                              use_kalman_filtering):
        """直接初期化（クラスメソッド用）"""
        instance = cls.__new__(cls)
        instance._init_from_args(encoder, decoder, realization, df_state_config,
                               df_obs_config, training_config, device, output_dir,
                               use_kalman_filtering, calibration_ratio=0.1,
                               auto_inference_setup=False)
        return instance
    
    def _init_from_args(self, encoder: nn.Module, decoder: nn.Module, realization: Realization,
                       df_state_config: Dict[str, Any], df_obs_config: Dict[str, Any],
                       training_config: TrainingConfig, device: torch.device, output_dir: str,
                       use_kalman_filtering: bool, calibration_ratio: float = 0.1,
                       auto_inference_setup: bool = True):
        """個別引数からの初期化"""
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
        
        # ===== 追加：Kalman関連設定 =====
        self.use_kalman_filtering = use_kalman_filtering
        self.calibration_ratio = calibration_ratio
        self.auto_inference_setup = auto_inference_setup
        
        # ===== 追加：Kalman用データ =====
        self.calibration_data: Optional[torch.Tensor] = None
        self.inference_model: Optional[Any] = None

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
        """DF layers初期化（GPU統一版）"""
        # DF-A初期化
        _, r = X_states.shape
        self.df_state = DFStateLayer(
            state_dim=r,
            feature_dim=self.df_state_config['feature_dim'],
            lambda_A=self.df_state_config['lambda_A'],
            lambda_B=self.df_state_config['lambda_B'],
            feature_net_config=self.df_state_config.get('feature_net'),
            cross_fitting_config=self.df_state_config.get('cross_fitting')
        )
        
        # DF-Aの内部ニューラルネットワークをGPUに移動
        self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
        
        # DF-B初期化
        self.df_obs = DFObservationLayer(
            df_state_layer=self.df_state,
            obs_feature_dim=self.df_obs_config['obs_feature_dim'],
            lambda_B=self.df_obs_config['lambda_B'],
            lambda_dB=self.df_obs_config['lambda_dB'],
            obs_net_config=self.df_obs_config.get('obs_net'),
            cross_fitting_config=self.df_obs_config.get('cross_fitting')
        )
        # DF-Bの内部ニューラルネットワークをGPUに移動
        self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)
        
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
        **多変量対応版**: データ準備（StochasticRealizationWithEncoder使用）

        Returns:
            M_features: エンコーダ出力系列 (T, m) - 多変量特徴量
            X_states: 状態推定系列 (T_eff, r)
        """
        print(f"📊 入力データ: Y_train.shape = {Y_train.shape}")

        # 入力データをGPUに移動
        Y_train = self._ensure_device(Y_train)

        # 1. エンコード: y_t → m_t ∈ R^m（多変量特徴量）
        with torch.no_grad():
            self.encoder.eval()  # BatchNorm等の正確な推論

            if Y_train.dim() == 2:
                # (T, d) → (1, T, d) for TCN
                Y_batch = Y_train.unsqueeze(0)
                M_batch = self.encoder(Y_batch).squeeze(0)  # (T, m)
            else:
                M_batch = self.encoder(Y_train)

            # 多変量特徴量の形状確認
            if M_batch.dim() == 1:
                # スカラー出力 → (T, 1)に変換
                M_features = M_batch.unsqueeze(1)
            elif M_batch.dim() == 2:
                # 正常な多変量特徴量 (T, m)
                M_features = M_batch
            else:
                raise ValueError(f"エンコーダ出力の形状が不正: {M_batch.shape}. 期待: (T, m)")

        print(f"📊 エンコーダ出力: M_features.shape = {M_features.shape} (多変量特徴量)")

        # 2. StochasticRealizationWithEncoderを使用して多変量特徴量を直接処理
        
        # ===== 追加：キャリブレーションデータ分割 =====
        if hasattr(self, 'use_kalman_filtering') and self.use_kalman_filtering:
            n_calib = int(Y_train.size(0) * getattr(self, 'calibration_ratio', 0.1))
            self.calibration_data = Y_train[:n_calib].clone()
            if self.config.verbose:
                print(f"Calibration data prepared: {self.calibration_data.shape}")

        # 2. StochasticRealizationWithEncoderを使用した確率的実現
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # 多変量特徴量を直接処理: Y_train + encoder → x_t
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
                print(f"📊 StochasticRealization使用: {Y_train.shape} → M_features={M_features.shape} → X_states={X_states.shape}")
            else:
                # 従来のRealizationの場合：多変量特徴量をスカラー化
                m_scalar = M_features.mean(dim=1)  # (T, m) → (T,) スカラー特徴量
                self.realization.fit(m_scalar.unsqueeze(1))  # (T,) → (T, 1)
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
                print(f"📊 従来Realization使用: M_features={M_features.shape} → m_scalar={m_scalar.shape} → X_states={X_states.shape}")
        except RealizationError as e:
            print(f"⚠️ RealizationError発生: {e}")
            # RealizationErrorを上位に再投げして完全エポックスキップを実行
            raise RealizationError(f"Phase1 realization failed: {e}") from e

        # ===== 追加：Kalman使用時の分岐処理（StochasticRealizationWithEncoder対応）=====
        if (hasattr(self, 'use_kalman_filtering') and self.use_kalman_filtering and
            hasattr(self, 'phase1_complete') and self.phase1_complete and
            hasattr(self, 'df_state') and self.df_state is not None and
            hasattr(self, 'df_obs') and self.df_obs is not None):

            try:
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    # StochasticRealizationWithEncoderの場合：多変量特徴量でKalman
                    X_means, X_covariances = self.realization.filter_with_kalman(
                        M_features, self.df_state, self.df_obs
                    )
                else:
                    # 従来Realizationの場合：スカラー特徴量でKalman
                    m_scalar = M_features.mean(dim=1)
                    X_means, X_covariances = self.realization.filter_with_kalman(
                        m_scalar, self.df_state, self.df_obs
                    )

                if self.config.verbose:
                    print(f"Kalman filtering applied: {X_means.shape}")

                # ===== 重要：共分散は内部保存、戻り値は既存形式 =====
                self._last_covariances = X_covariances  # 新しい内部属性
                X_states = X_means  # 既存変数名で返す

            except Exception as e:
                warnings.warn(f"Kalman filtering failed, using deterministic: {e}")
                # Kalman失敗時は上記で計算済みのX_statesを使用
                self._last_covariances = None
        
        # **修正4: 状態系列長を記録（時間調整用）**
        self._last_X_states_length = X_states.size(0)

        if self.config.verbose:
            print(f"状態推定完了: M_features={M_features.shape} -> X_states={X_states.shape}")

        # 一時保存（多変量特徴量対応）
        self._temp_data = {
            'Y_train': Y_train,
            'M_features': M_features,  # 多変量特徴量
            'X_states': X_states
        }

        return M_features, X_states  # 多変量特徴量を返す
    
    def train_phase1(self, Y_train: torch.Tensor) -> Dict[str, Any]:
        """
        Phase-1: DF-A/DF-B の協調学習
        
        Args:
            Y_train: 訓練観測系列 (T, d)
            
        Returns:
            Phase-1メトリクス
        """
        print("\n=== Phase-1: DF学習開始 ===")

        # データ準備（多変量特徴量対応）
        M_features, X_states = self._prepare_data(Y_train)
        
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
                df_b_metrics = self._train_df_b_epoch(X_states, M_features, epoch)
                epoch_metrics.update(df_b_metrics)
            
            # ログ記録
            self.training_history['phase1_metrics'].append(epoch_metrics)
            
            # ログ出力
            if epoch % self.config.log_interval == 0 and self.config.verbose:
                self._print_phase1_progress(epoch, epoch_metrics)
                # 多変量対応の追加ログ
                self.log_multivariate_training_progress(epoch, "phase1")
            
            # モデル保存
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE1_DF_A)
        
        # Phase-1完了後: DFLayerのfit()でV_A/V_B/U_A/u_Bを計算
        print("🔄 最終作用素（V_A/V_B/U_A/u_B）計算中...")
        self._compute_final_operators(Y_train)

        self.phase1_complete = True
        print("Phase-1 学習完了")

        return self.training_history['phase1_metrics']

    def _compute_final_operators(self, Y_train: torch.Tensor):
        """Phase-1完了後に最終作用素V_A/V_B/U_A/U_Bを計算（多変量特徴量対応）"""
        # データ準備（多変量特徴量対応）
        M_features, X_states = self._prepare_data(Y_train)

        # DFStateLayer用データ準備
        with torch.no_grad():
            # 状態特徴量計算: φ_θ(x_t)
            Phi_full = self.df_state.phi_theta(X_states)  # (T, d_A)

            # 時間シフトしたデータ準備（元の学習と同じ時間対応）
            Phi_minus = Phi_full[:-1]  # φ(x_{t-1}): t=0,...,T-2
            Phi_plus = Phi_full[1:]    # φ(x_t): t=1,...,T-1
            X_plus = X_states[1:]      # x_{t}: t=1,...,T-1 (元の学習と同じ)

        # DF-A: V_A, U_A計算
        print("  🔄 DF-A作用素（V_A/U_A）計算中...")
        if hasattr(self.df_state, 'cf_config') and self.df_state.cf_config:
            self.df_state._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)
        else:
            self.df_state._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)

        print(f"  ✅ V_A shape: {self.df_state.V_A.shape}, U_A shape: {self.df_state.U_A.shape}")

        # DFObservationLayer用データ準備（DF-Aの結果を使用）
        if hasattr(self, 'df_obs') and self.df_obs is not None:
            print("  🔄 DF-B作用素（V_B/u_B）計算中...")

            # DF-Aによる1ステップ予測とエンコーダ出力の正しい使用
            with torch.no_grad():
                # 1ステップ予測: x̂_{t|t-1} = U_A^T V_A φ(x_{t-1})
                X_pred = (self.df_state.U_A.T @ (self.df_state.V_A @ Phi_minus.T)).T  # (T-1, d_x)
                # 予測を特徴量化: φ_θ(x̂_{t|t-1})
                Phi_pred = self.df_state.phi_theta(X_pred)  # (T-1, d_A)

                # realizationの時間短縮を考慮した多変量特徴量の取得
                # realization: T -> T_eff = T - 2*h + 1の短縮
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    # StochasticRealizationWithEncoderの場合
                    h = getattr(self.realization, 'window_length', 5)  # デフォルト値
                else:
                    # 従来Realizationの場合
                    h = self.realization.h
                T_states = X_states.shape[0]  # realization後の状態系列長

                # 多変量特徴量の正しい範囲（realizationと同じ時間範囲）
                M_curr = M_features[h:h+T_states]  # M_t: realization範囲と一致 (T_states, m)

                # 観測特徴量: ψ_ω(M_t) - 多変量対応
                Psi_curr = self.df_obs.psi_omega(M_curr)  # (T_states, d_B)

                # 同時刻の多変量特徴量
                m_curr = M_curr  # M_t (T_states, m)

                # データサイズ調整（最小サイズに合わせる）
                min_size = min(Phi_pred.shape[0], Psi_curr.shape[0], m_curr.shape[0])
                Phi_pred = Phi_pred[:min_size]  # φ_θ(x̂_{t|t-1})
                Psi_curr = Psi_curr[:min_size]  # ψ_ω(h_t)
                m_curr = m_curr[:min_size]     # m_t

            print(f"    📊 DF-B学習データ: Phi_pred={Phi_pred.shape}, Psi_curr={Psi_curr.shape}, m_curr={m_curr.shape}")
            print(f"    🕐 時間範囲: h={h}, T_states={T_states}, range=[{h}:{h+T_states}]")

            # DF-B: V_B, u_B計算 (φ_θ(x̂_{t|t-1}) → ψ_ω(h_t)の写像学習)
            if hasattr(self.df_obs, 'cf_config') and self.df_obs.cf_config:
                self.df_obs._fit_with_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)
            else:
                self.df_obs._fit_without_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)

            print(f"  ✅ V_B shape: {self.df_obs.V_B.shape}, U_B shape: {self.df_obs.U_B.shape}")

        print("🔄 最終作用素計算完了")

    def _train_df_a_epoch(self, X_states: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        **修正2統合**: DF-A（状態層）のエポック学習（完全グラフ分離版）
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        
        # **追加**: 入力データをGPUに移動
        X_states_gpu = X_states.to(self.device)
        
        # **修正2: 完全にデタッチされた入力で独立グラフ作成**
        X_states_detached = X_states_gpu.detach().requires_grad_(False)
        
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
        
        # Stage-2: U_A推定（一度だけ実行）
        with torch.no_grad():  # **修正2: Stage-2は勾配なし**
            stage2_metrics = self.df_state.train_stage2_closed_form()

            # ログ記録
            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_A, 'stage2', 0,
                stage2_metrics, {}
            )

        metrics['df_a_stage2_loss'] = stage2_metrics['stage2_loss']
        
        # **修正2: 明示的グラフクリア**
        self._clear_computation_graph()
        
        return metrics
    
    def _train_df_b_epoch(self, X_states: torch.Tensor, M_features: torch.Tensor,
                        epoch: int) -> Dict[str, float]:
        """
        **多変量対応版**: DF-B（観測層）のエポック学習（計算グラフ重複使用エラー対応）
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        opt_psi = self.optimizers['psi']

        # **追加**: 入力データをGPUに移動
        X_states_gpu = X_states.to(self.device)
        M_features_gpu = M_features.to(self.device)

        # **修正2: 完全にデタッチされた入力で独立グラフ作成**
        X_states_detached = X_states_gpu.detach().requires_grad_(False)
        
        # **修正2: 独立計算コンテキストでDF-B学習**
        with torch.enable_grad():
            # DF-Aからの状態予測を取得（操作変数として推論のみ）
            with torch.no_grad():
                X_hat_states = self.df_state.predict_sequence(X_states_detached)
            
            # 明示的に勾配グラフから切断
            X_hat_states = X_hat_states.detach().requires_grad_(False)
            
            # **修正4: ヘルパーメソッドを使用した時間インデックス調整（多変量対応）**
            M_aligned = self._align_time_series_multivariate(
                X_hat_states, M_features_gpu, X_states.size(0), epoch, "DF-B"
            )
            
            # Stage-1学習（多変量特徴量対応）
            stage1_metrics = self.df_obs.train_stage1_with_gradients(
                X_hat_states,
                M_aligned,  # **修正2+4: 時間調整済み多変量特徴量**
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
        
        # Stage-2: u_B推定 + ψ_ω更新（T2回反復、計算グラフ分離）
        stage2_losses = []
        for t in range(self.config.T2_iterations):
            with torch.enable_grad():
                # **修正**: 各反復で独立した計算グラフを作成（多変量対応）
                M_aligned_independent = M_aligned.detach().requires_grad_(True)

                stage2_metrics = self.df_obs.train_stage2_with_gradients(
                    M_aligned_independent,  # ← 修正: 独立した多変量テンソル
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
    
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        # テンソルを指定デバイスに移動（必要な場合のみ）
        return tensor.to(self.device) if tensor.device != self.device else tensor

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

        # GPUデバイス整合性を確保
        Y_train = self._ensure_device(Y_train)
        if Y_val is not None:
            Y_val = self._ensure_device(Y_val)
        
        if not self.phase1_complete:
            raise RuntimeError("Phase-1が完了していません")
        
        opt_e2e = self.optimizers['e2e']
        
        # Phase-2学習ループ
        for epoch in range(self.config.phase2_epochs):
            self.current_epoch = self.config.phase1_epochs + epoch
            
            try:
                # 前向き推論と損失計算
                loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train)
                
                # 逆伝播
                opt_e2e.zero_grad()
                loss_total.backward()
                opt_e2e.step()
                
            except RealizationError as e:
                print(f"🔄 Epoch {epoch} スキップ (Phase2数値実現失敗): {e}")
                # このエポックを完全スキップして次のエポックに進む
                continue
            
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
                # 多変量対応の追加ログ
                self.log_multivariate_training_progress(epoch, "phase2")
            
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
        
        # Step 1: エンコード y_t → m_t（多変量特徴量対応）
        M_features = self.encoder(Y_train.unsqueeze(0)).squeeze(0)  # (T, m)
        if M_features.dim() == 1:
            M_features = M_features.unsqueeze(1)  # (T,) → (T, 1)
        
        # Step 2: 確率的実現 M_t → x_t（多変量対応）
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # StochasticRealizationWithEncoderの場合：多変量特徴量を直接処理
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
            else:
                # 従来Realizationの場合：スカラー化
                m_scalar = M_features.mean(dim=1)  # (T, m) → (T,)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
        except RealizationError as e:
            print(f"⚠️ Phase2 RealizationError発生: {e}")
            # RealizationErrorを上位に再投げして完全エポックスキップを実行
            raise RealizationError(f"Phase2 realization failed: {e}") from e
        T_eff = X_states.size(0)
        
        # Step 3: DF-A予測 x_{t-1} → x̂_{t|t-1}
        X_hat_states = self.df_state.predict_sequence(X_states)  # (T_pred, r)
        T_pred = X_hat_states.size(0)
        
        # **修正4: ヘルパーメソッドを使用した時間調整（多変量対応）**
        M_aligned = self._align_time_series_multivariate(
            X_hat_states, M_features, X_states.size(0), 0, "Phase2"
        )
        
        # Step 4: DF-B予測 x̂_{t|t-1} → m̂_{t|t-1}（多変量対応）
        M_hat_series = []
        for t in range(T_pred):
            m_hat_t = self.df_obs.predict_one_step(X_hat_states[t])  # m ∈ R^m（既に多変量対応）
            M_hat_series.append(m_hat_t)
        M_hat_tensor = torch.stack(M_hat_series)  # (T_pred, m)
        M_hat_tensor = self._ensure_device(M_hat_tensor)  # GPUデバイス整合性を確保

        # Step 5: デコード m̂_{t|t-1} → ŷ_{t|t-1}（多変量対応）
        # time_invariantDecoderは (T, m) → (T, n) 変換
        Y_hat = self.decoder(M_hat_tensor)  # (T_pred, n)
        
        # Step 6: 対応する真値取得
        Y_target = Y_train[h+1:h+1+T_pred]  # 対応する観測
        Y_target = self._ensure_device(Y_target)  # GPUデバイス整合性を確保

        # 損失計算（正規化済み）
        loss_rec = torch.norm(Y_hat - Y_target, p='fro') ** 2 / Y_target.numel()
        
        # CCA損失（オプション）（多変量対応）
        if self.config.lambda_cca > 0:
            loss_cca = self._compute_cca_loss(M_hat_tensor, M_aligned)
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
        
        try:
            self.realization.fit(m_series.unsqueeze(1))
        except RealizationError as e:
            print(f"⚠️ Evaluation RealizationError発生: {e}")
            # 評価時はエラーを上位に投げて処理をスキップ
            raise RealizationError(f"Evaluation realization failed: {e}") from e
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
    
    def _compute_cca_loss(self, M_hat: torch.Tensor, M_target: torch.Tensor) -> torch.Tensor:
        """
        **修正版**: 確率的実現の正準相関係数に基づくCCA損失

        定式化に基づく正しい実装：
        L_cca = -Σ_i ρ_i （正準相関係数の和の最大化）

        Args:
            M_hat: 予測多変量特徴量 (T, m) - ログ・監視用（将来の動的更新に備えて保持）
            M_target: 真の多変量特徴量 (T, m) - ログ・監視用（将来の動的更新に備えて保持）

        Returns:
            torch.Tensor: CCA損失（負の正準相関係数和）
        """
        try:
            # 確率的実現から正準相関係数を取得
            canonical_correlations = self._get_canonical_correlations_from_realization()

            if canonical_correlations is None:
                # 正準相関係数が利用できない場合の警告とフォールバック
                if self.config.verbose:
                    print("⚠️ 正準相関係数が利用できません。CCA損失を0とします。")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

            # 正準相関係数の和の最大化 → 負号で最小化問題に変換
            cca_loss = -canonical_correlations.sum()

            # デバッグ・監視情報（verbose時のみ）
            if self.config.verbose and hasattr(self, 'current_epoch'):
                if getattr(self, 'current_epoch', 0) % self.config.log_interval == 0:
                    print(f"  📊 CCA損失: ρ_sum={canonical_correlations.sum().item():.4f}, "
                          f"loss={cca_loss.item():.4f}")

            return cca_loss

        except Exception as e:
            warnings.warn(f"CCA損失計算エラー: {e}. 損失を0とします。")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

    def _get_canonical_correlations_from_realization(self) -> Optional[torch.Tensor]:
        """
        確率的実現クラスから正準相関係数を取得

        Returns:
            torch.Tensor: 正準相関係数 ρ_i ∈ R^r, または None
        """
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # StochasticRealizationWithEncoderの場合
                if hasattr(self.realization, 'canonical_correlations') and \
                   self.realization.canonical_correlations is not None:
                    correlations = self.realization.canonical_correlations.clone()
                    # デバイス統一
                    return correlations.to(self.device)

            elif hasattr(self.realization, '_L_vals') and self.realization._L_vals is not None:
                # 従来Realizationの場合：特異値を正準相関係数として使用
                # 特異値は [0, 1] に正規化されている前提
                l_vals = self.realization._L_vals.clone()
                # デバイス統一
                return l_vals.to(self.device)

            return None

        except Exception as e:
            warnings.warn(f"正準相関係数取得エラー: {e}")
            return None
    
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
        
        # デバッグ情報（verbose時のみ） - コメントアウト
        # if self.config.verbose and epoch % 10 == 0:
        #     print(f"{component}時間調整 - Epoch {epoch}: "
        #           f"X_hat: {X_hat_states.shape}, m_aligned: {m_aligned.shape}, "
        #           f"offset: {total_offset}")
        
        return m_aligned

    def _align_time_series_multivariate(
        self,
        X_hat_states: torch.Tensor,
        M_features: torch.Tensor,
        T_states: int,
        epoch: int,
        component: str = "unknown"
    ) -> torch.Tensor:
        """
        **多変量対応版**: 時間インデックス調整（M_features ∈ R^(T×m)）

        Args:
            X_hat_states: 状態予測 (T_pred, r)
            M_features: 多変量特徴量 (T, m)
            T_states: 状態系列長
            epoch: エポック番号
            component: コンポーネント名

        Returns:
            torch.Tensor: 時間調整済み多変量特徴系列 (T_pred, m)
        """
        T_pred = X_hat_states.size(0)
        T_original = M_features.size(0)

        # オフセット計算（既存メソッド使用）
        total_offset = self._get_time_alignment_offset(T_original, T_states, T_pred)

        # 時間調整された多変量特徴量を抽出
        if total_offset + T_pred <= T_original:
            M_aligned = M_features[total_offset:total_offset + T_pred]  # (T_pred, m)
        else:
            # 安全措置: 末尾から必要な長さを取得
            M_aligned = M_features[-T_pred:]  # (T_pred, m)
            if self.config.verbose:
                print(f"警告: {component} でオフセット調整失敗、末尾切り取り使用")

        # 時間整合性検証（多変量対応）
        self._validate_time_alignment_multivariate(X_hat_states, M_aligned, component)

        return M_aligned

    def _get_time_alignment_offset(self, T_original: int, T_states: int, T_pred: int) -> int:
        """
        時間インデックス調整のオフセット計算
        理論: 
        - 確率的実現出力: x_h, x_{h+1}, ..., x_{h+T_states-1}
        - DF-A予測: x̂_{h+1|h}, x̂_{h+2|h+1}, ..., x̂_{h+T_pred|h+T_pred-1}
        - 正しい対応: x̂_{h+1|h} ↔ m_{h+1}
        Args:
            T_original: 元系列長
            T_states: 状態系列長  
            T_pred: 予測系列長
            
        Returns:
            int: m_seriesのオフセット (= h + 1)
        """
        # h値取得
        h_candidates = [
            'h',                    # Realizationの標準属性名
            'past_horizon',         # 初期化パラメータ名
            'lags',                 # 別名の可能性
            'window_size',          # 別名の可能性
        ]
        
        h = None
        for attr_name in h_candidates:
            if hasattr(self.realization, attr_name):
                h = getattr(self.realization, attr_name)
                if isinstance(h, (int, float)) and h > 0:
                    h = int(h)
                    break
        
        # フォールバック: T_original, T_statesから逆算
        if h is None:
            # T_states = T_original - 2*h + 1 から h を計算
            h = (T_original - T_states + 1) // 2
            if self.config.verbose:
                print(f"警告: h値を逆算で推定しました: h = {h}")
        
        # 検証
        expected_T_states = T_original - 2 * h + 1
        if abs(T_states - expected_T_states) > 1:  # 1の誤差は許容
            if self.config.verbose:
                print(f"警告: h={h}による期待T_states={expected_T_states}が実際値{T_states}と不一致")
        
        return h + 1
    
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
        
        # 時間整合確認メッセージ - コメントアウト
        # if self.config.verbose:
        #     print(f"{component} 時間整合確認: {X_hat_states.shape} ↔ {m_aligned.shape}")

    def _validate_time_alignment_multivariate(
        self,
        X_hat_states: torch.Tensor,
        M_aligned: torch.Tensor,
        component: str = "unknown"
    ) -> None:
        """
        **多変量対応版**: 時間インデックス整合性の検証

        Args:
            X_hat_states: 状態予測 (T_pred, r)
            M_aligned: 調整済み多変量特徴量 (T_pred, m)
            component: コンポーネント名（ログ用）
        """
        if X_hat_states.size(0) != M_aligned.size(0):
            raise RuntimeError(
                f"{component}の時間インデックス不整合: "
                f"X_hat={X_hat_states.shape} vs M_aligned={M_aligned.shape}"
            )

        # 多変量特徴量の次元確認
        if M_aligned.dim() != 2:
            raise RuntimeError(
                f"{component}の多変量特徴量形状エラー: "
                f"期待=(T_pred, m), 実際={M_aligned.shape}"
            )

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """テンソルを適切なデバイスに移動"""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def _clear_computation_graph(self):
        """
        **修正2**: 計算グラフの明示的クリア
        """
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CPU ガベージコレクション
        gc.collect()

    def verify_multivariate_dimensions(self, M_features: torch.Tensor, X_states: torch.Tensor) -> Dict[str, Any]:
        """
        多変量対応の次元整合性検証

        Args:
            M_features: 多変量特徴量 (T, m)
            X_states: 状態推定 (T_eff, r)

        Returns:
            Dict: 検証結果
        """
        result = {
            "status": "ok",
            "dimensions": {},
            "warnings": [],
            "errors": []
        }

        try:
            # 基本次元確認
            result["dimensions"] = {
                "M_features_shape": tuple(M_features.shape),
                "X_states_shape": tuple(X_states.shape),
                "encoder_output_dim": getattr(self.encoder, 'output_dim', 'unknown'),
                "expected_feature_dim": getattr(self.df_obs, 'multivariate_feature_dim', 'unknown'),
                "df_state_feature_dim": getattr(self.df_state, 'feature_dim', 'unknown'),
                "df_obs_feature_dim": getattr(self.df_obs, 'obs_feature_dim', 'unknown')
            }

            # 多変量特徴量形状チェック
            if M_features.dim() != 2:
                result["errors"].append(f"M_features形状エラー: 期待=(T, m), 実際={M_features.shape}")

            # DF-B多変量特徴量次元チェック
            if hasattr(self.df_obs, 'multivariate_feature_dim'):
                expected_m = self.df_obs.multivariate_feature_dim
                if M_features.size(1) != expected_m:
                    result["errors"].append(
                        f"特徴量次元不一致: M_features.size(1)={M_features.size(1)} vs "
                        f"expected={expected_m}"
                    )

            # U_B行列次元チェック
            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                U_B_shape = self.df_obs.U_B.shape
                expected_U_B_shape = (self.df_obs.obs_feature_dim, M_features.size(1))
                if tuple(U_B_shape) != expected_U_B_shape:
                    result["errors"].append(
                        f"U_B行列次元エラー: 実際={U_B_shape} vs 期待={expected_U_B_shape}"
                    )

            # 数値安定性チェック
            if hasattr(self.df_obs, 'V_B') and self.df_obs.V_B is not None:
                cond_V_B = torch.linalg.cond(self.df_obs.V_B).item()
                if cond_V_B > 1e12:
                    result["warnings"].append(f"V_B条件数が大きい: {cond_V_B:.2e}")

            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                # U_B @ U_B.T の条件数（行列化による数値安定性）
                U_B_gram = self.df_obs.U_B @ self.df_obs.U_B.T
                cond_U_B_gram = torch.linalg.cond(U_B_gram).item()
                if cond_U_B_gram > 1e12:
                    result["warnings"].append(f"U_B @ U_B.T条件数が大きい: {cond_U_B_gram:.2e}")

            if result["errors"]:
                result["status"] = "error"
            elif result["warnings"]:
                result["status"] = "warning"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"検証中にエラー: {str(e)}")

        return result

    def log_multivariate_training_progress(self, epoch: int, phase: str):
        """
        多変量対応の学習進捗ログ

        Args:
            epoch: エポック番号
            phase: 学習フェーズ ("phase1_df_a", "phase1_df_b", "phase2")
        """
        if not self.config.verbose or epoch % self.config.log_interval != 0:
            return

        try:
            print(f"\n📊 多変量学習進捗確認 - Epoch {epoch}, Phase: {phase}")

            # エンコーダー統計
            if hasattr(self, '_temp_data') and 'M_features' in self._temp_data:
                M_features = self._temp_data['M_features']
                print(f"  📈 多変量特徴量: shape={M_features.shape}, "
                      f"mean={M_features.mean().item():.4f}, "
                      f"std={M_features.std().item():.4f}")

            # DF-B統計
            if hasattr(self, 'df_obs') and self.df_obs is not None:
                if hasattr(self.df_obs, 'V_B') and self.df_obs.V_B is not None:
                    V_B_norm = torch.norm(self.df_obs.V_B, p='fro').item()
                    print(f"  🔗 V_B: shape={self.df_obs.V_B.shape}, norm={V_B_norm:.4f}")

                if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                    U_B_norm = torch.norm(self.df_obs.U_B, p='fro').item()
                    print(f"  📐 U_B: shape={self.df_obs.U_B.shape}, norm={U_B_norm:.4f}")

            # 数値安定性確認
            if hasattr(self, '_temp_data') and 'M_features' in self._temp_data:
                M_features = self._temp_data['M_features']
                X_states = self._temp_data['X_states']
                verification = self.verify_multivariate_dimensions(M_features, X_states)

                if verification["status"] == "error":
                    print(f"  ❌ 次元エラー: {verification['errors']}")
                elif verification["status"] == "warning":
                    print(f"  ⚠️  数値警告: {verification['warnings']}")
                else:
                    print(f"  ✅ 次元整合性: OK")

        except Exception as e:
            print(f"  ❌ ログ出力エラー: {e}")
    
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
            try:
                self.realization.fit(m_warmup.unsqueeze(1))
            except RealizationError as e:
                print(f"⚠️ Warmup RealizationError発生: {e}")
                # ウォームアップ時はエラーを上位に投げて処理をスキップ
                raise RealizationError(f"Warmup realization failed: {e}") from e
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
        print("DEBUG: _save_final_model called")
        # 学習時の完全な設定を復元
        complete_config = self._build_complete_config()
        
        model_state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'df_state': self.df_state.get_inference_state_dict() if self.df_state else None,
            'df_obs': self.df_obs.get_inference_state_dict() if self.df_obs else None,
            'realization_config': self.realization.__dict__,
            'training_config': self.config.__dict__,
            'config': complete_config  # 推論時に使用される完全な設定
        }
        
        save_path = self.output_dir / 'final_model.pth'
        torch.save(model_state, save_path)
        
        print(f"最終モデル保存: {save_path}")
    
    def _build_complete_config(self) -> Dict[str, Any]:
        """学習時の完全な設定を構築（推論時に使用）"""
        print("DEBUG: _build_complete_config called")
        
        complete_config = {
            'model': {
                'encoder': {
                    'input_dim': getattr(self.encoder, 'input_dim', 6),
                    'channels': getattr(self.encoder, 'channels', 64),
                    'layers': getattr(self.encoder, 'layers', 8),
                    'kernel_size': getattr(self.encoder, 'kernel_size', 3),
                    'activation': getattr(self.encoder, 'activation', 'GELU'),
                    'dropout': getattr(self.encoder, 'dropout', 0.1)
                },
                'decoder': {
                    'output_dim': getattr(self.decoder, 'output_dim', 6),
                    'window': getattr(self.decoder, 'window', 12),
                    'tau': getattr(self.decoder, 'tau', 1),
                    'hidden': getattr(self.decoder, 'hidden', 64),
                    'ma_kernel': getattr(self.decoder, 'ma_kernel', 24),
                    'gru_hidden': getattr(self.decoder, 'gru_hidden', 32),
                    'activation': getattr(self.decoder, 'activation', 'GELU'),
                    'dropout': getattr(self.decoder, 'dropout', 0.1)
                }
            },
            'ssm': {
                'realization': self.realization.__dict__,
                'df_state': self._extract_df_state_config(),
                'df_observation': self._extract_df_obs_config()
            }
        }
        
        print("DEBUG: _build_complete_config completed")
        return complete_config
    
    def _extract_df_state_config(self) -> Dict[str, Any]:
        """実際のDFStateLayerから設定を抽出"""
        base_config = self.df_state_config.copy()
        
        print(f"DEBUG: df_state exists: {self.df_state is not None}")
        
        # 実際のDFStateLayerから詳細設定を抽出
        if self.df_state and hasattr(self.df_state, 'phi_theta'):
            print("DEBUG: df_state has phi_theta")
            # StateFeatureNetの構造を解析
            phi_theta = self.df_state.phi_theta
            print(f"DEBUG: phi_theta type: {type(phi_theta)}")
            print(f"DEBUG: phi_theta has net: {hasattr(phi_theta, 'net')}")
            
            if hasattr(phi_theta, 'net') and len(phi_theta.net) > 0:
                print(f"DEBUG: phi_theta.net length: {len(phi_theta.net)}")
                print(f"DEBUG: phi_theta.net layers: {[type(layer).__name__ for layer in phi_theta.net]}")
                
                # ネットワーク構造から hidden_sizes を逆算
                hidden_sizes = []
                for i, layer in enumerate(phi_theta.net):
                    print(f"DEBUG: Layer {i}: {type(layer).__name__}")
                    if hasattr(layer, 'out_features'):
                        print(f"DEBUG: Layer {i} out_features: {layer.out_features}")
                        hidden_sizes.append(layer.out_features)
                
                print(f"DEBUG: Raw hidden_sizes: {hidden_sizes}")
                
                # 最後の層は除く（出力層）
                if len(hidden_sizes) > 1:
                    hidden_sizes = hidden_sizes[:-1]
                    print(f"DEBUG: Final hidden_sizes: {hidden_sizes}")
                
                # feature_net 設定を構築
                base_config['feature_net'] = {
                    'hidden_sizes': hidden_sizes,
                    'activation': getattr(phi_theta, 'activation', 'ReLU'),
                    'dropout': getattr(phi_theta, 'dropout', 0.1)
                }
                print(f"DEBUG: Created feature_net config: {base_config['feature_net']}")
            else:
                print("DEBUG: phi_theta.net not found or empty")
        else:
            print("DEBUG: df_state doesn't have phi_theta")
        
        print(f"DEBUG: Final base_config keys: {list(base_config.keys())}")
        return base_config
    
    def _extract_df_obs_config(self) -> Dict[str, Any]:
        """実際のDFObservationLayerから設定を抽出"""
        base_config = self.df_obs_config.copy()
        
        # 実際のDFObservationLayerから詳細設定を抽出
        if self.df_obs and hasattr(self.df_obs, 'psi_omega'):
            psi_omega = self.df_obs.psi_omega
            if hasattr(psi_omega, 'net') and len(psi_omega.net) > 0:
                # ネットワーク構造から hidden_sizes を逆算
                hidden_sizes = []
                for layer in psi_omega.net:
                    if hasattr(layer, 'out_features'):
                        hidden_sizes.append(layer.out_features)
                
                # 最後の層は除く（出力層）
                if len(hidden_sizes) > 1:
                    hidden_sizes = hidden_sizes[:-1]
                
                # obs_net 設定を構築
                base_config['obs_net'] = {
                    'hidden_sizes': hidden_sizes,
                    'activation': getattr(psi_omega, 'activation', 'ReLU'),
                    'dropout': getattr(psi_omega, 'dropout', 0.1)
                }
        
        return base_config
    
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
    
    """
    学習後の推論環境自動セットアップ
    """
    def post_training_setup(self, Y_train: torch.Tensor) -> Dict[str, Any]:
        if not self.use_kalman_filtering:
            return {"status": "kalman_disabled"}
        
        if not self.phase1_complete:
            return {"status": "phase1_incomplete"}
        
        print("Setting up post-training inference environment...")
        
        try:
            # 推論設定読み込み（後述の設定ファイルベース手法使用）
            inference_config = self._load_inference_config()
            
            # 一時的なモデル保存
            temp_model_path = self.output_dir / "temp_inference_model.pth"
            self._save_inference_ready_model(temp_model_path)
            
            # InferenceModel初期化
            from ..models.inference_model import InferenceModel
            
            self.inference_model = InferenceModel(
                trained_model_path=temp_model_path,
                inference_config=inference_config
            )
            
            # 推論環境セットアップ
            self.inference_model.setup_inference(calibration_data=self.calibration_data)
            
            # 最終エクスポート
            self.inference_model.export_for_deployment(
                export_path=self.output_dir / "inference_deployment"
            )
            
            # 一時ファイル削除
            if temp_model_path.exists():
                temp_model_path.unlink()
            
            return {"status": "success", "inference_model": self.inference_model}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _save_inference_ready_model(self, save_path):
        """推論用モデル保存"""
        print("DEBUG: _save_inference_ready_model called")
        
        model_state = {
            'config': {
                'ssm': {
                    'realization': {
                        'past_horizon': self.realization.h,
                        'rank': self.realization.rank,
                        'jitter': getattr(self.realization, 'jitter', 1e-3)
                    },
                    'df_state': self._extract_df_state_config(),
                    'df_observation': self._extract_df_obs_config()
                },
                'model': {
                    'encoder': {
                        'input_dim': getattr(self.encoder, 'input_dim', 7)
                    }
                }
            },
            'model_state_dict': {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'df_state': self.df_state.get_inference_state_dict(),
                'df_obs': self.df_obs.get_inference_state_dict()
            }
        }
        torch.save(model_state, save_path)

    def _load_inference_config(self) -> Dict[str, Any]:
        """推論設定の読み込み（警告のみ版）"""
        try:
            from configs.inference_config_loader import load_inference_config
            
            environment = "production" if not self.config.verbose else "development"
            inference_config = load_inference_config(environment=environment)
            inference_config["device"] = str(self.device)
            
            if self.config.verbose:
                print(f"推論設定読み込み完了（環境: {environment}）")
            
            return inference_config
            
        except ImportError as e:
            warnings.warn(f"InferenceConfigLoaderモジュールが見つかりません: {e}. 内蔵デフォルト値を使用します。")
            return self._use_builtin_defaults()
            
        except FileNotFoundError as e:
            warnings.warn(f"推論設定ファイルが見つかりません: {e}. 内蔵デフォルト値を使用します。")
            return self._use_builtin_defaults()
            
        except Exception as e:
            warnings.warn(f"推論設定読み込み中にエラーが発生: {e}. 内蔵デフォルト値を使用します。")
            return self._use_builtin_defaults()

    def _use_builtin_defaults(self) -> Dict[str, Any]:
        """内蔵デフォルト値を直接使用"""
        try:
            from configs.inference_config_loader import InferenceConfigLoader
            
            # クラスのデフォルト値メソッドを直接使用
            loader = InferenceConfigLoader.__new__(InferenceConfigLoader)  # __init__回避
            
            config = {
                'device': str(self.device),
                'noise_estimation': loader._get_default_section('noise_estimation'),
                'initialization': loader._get_default_section('initialization'),
                'numerical': loader._get_default_section('numerical'),
                # streamingとoutputは設定ファイル固有なので直接定義
                'streaming': {
                    'buffer_size': 100,
                    'batch_processing': False,
                    'anomaly_detection': True,
                    'anomaly_threshold': 3.0
                },
                'output': {
                    'save_states': True,
                    'save_covariances': False,
                    'save_likelihoods': True
                }
            }
            
            if self.config.verbose:
                print("内蔵デフォルト値を使用して推論設定を初期化しました")
            
            return config
            
        except Exception as nested_e:
            # この場合はクラス自体に問題があるので、エラーを発生させる
            raise RuntimeError(
                f"内蔵デフォルト値の取得にも失敗しました: {nested_e}. "
                f"InferenceConfigLoaderクラスの実装を確認してください。"
            ) from nested_e



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
    
    # モデル初期化（新しいtime_invariantアーキテクチャ使用）
    encoder_config = config['model']['encoder'].copy()
    decoder_config = config['model']['decoder'].copy()

    # time_invariant用の設定調整
    if 'output_dim' not in encoder_config:
        encoder_config['output_dim'] = encoder_config.get('channels', 32)

    encoder = time_invariantEncoder(**encoder_config)
    decoder = time_invariantDecoder(**decoder_config)

    # 確率実現（新しいクラス優先）
    realization_config = config['ssm']['realization']
    if config.get('evaluation', {}).get('use_new_realization', True):
        # StochasticRealizationWithEncoderはencoder引数が必要
        realization_config_copy = realization_config.copy()
        realization = StochasticRealizationWithEncoder(
            encoder=encoder,
            **realization_config_copy
        )
    else:
        realization = Realization(**realization_config)
    
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
    import yaml
    import numpy as np
    from ..utils.gpu_utils import select_device

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # デバイス設定
    if device is None:
        device = select_device()
    
    print(f"実験開始: device={device}")
    
    # データ読み込み
    try:
        from ..utils.data_loader import load_experimental_data
        
        # data設定が存在するかチェック
        if 'data' in config:
            print(f"📂 統一データローダーでデータ読み込み: {data_path}")
            data_dict = load_experimental_data(data_path, config['data'])
            Y_train = data_dict['train'].to(device)
            print(f"📊 データ形状: {Y_train.shape} (正規化: {data_dict['metadata'].normalization_method})")
        else:
            raise ImportError("data設定がないため従来方式を使用")
            
    except (ImportError, ModuleNotFoundError, Exception) as e:
        print(f"⚠️  統一データローダー使用不可、従来方式: {e}")
        
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