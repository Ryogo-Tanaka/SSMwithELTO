#!/usr/bin/env python3
# scripts/run_full_experiment.py
"""
タスク3: 実データ学習・評価環境構築
完全実験パイプライン実行スクリプト

機能:
- 統一データローダーによる前処理
- Phase-1 + Phase-2学習（Kalman含む）
- 学習済みモデル・転送作用素の保存
- 学習過程の可視化・ログ記録
- 実験再現性担保

実行例:
python scripts/run_full_experiment.py \
    --config configs/full_experiment_config.yaml \
    --data data/sim_complex.npz \
    --output results/full_experiment_001 \
    --use-kalman
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# プロジェクトルートパス設定
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 統一データローダー
from src.utils.data_loader import load_experimental_data, DataMetadata

# 既存の学習クラス
from src.training.two_stage_trainer import TwoStageTrainer
from src.utils.gpu_utils import select_device


class FullExperimentPipeline:
    """
    完全実験パイプライン実行クラス
    
    タスク3の要件を満たす完全な学習・評価環境
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        # 出力ディレクトリ準備
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'artifacts').mkdir(exist_ok=True)
        
        # ログ設定
        self.experiment_log = []
        self.start_time = datetime.now()
        
        self._log_experiment_start()
    
    def _log_experiment_start(self):
        """実験開始ログ"""
        log_entry = {
            'timestamp': self.start_time.isoformat(),
            'event': 'experiment_start',
            'config': self.config,
            'device': str(self.device),
            'output_dir': str(self.output_dir)
        }
        self.experiment_log.append(log_entry)
        print(f"🚀 実験開始: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print(f"🖥️  計算デバイス: {self.device}")
    
    def step_1_data_loading(self, data_path: str) -> Dict[str, torch.Tensor]:
        """
        Step 3.1: データ読み込み・前処理
        
        Args:
            data_path: データファイルパス
            
        Returns:
            処理済みデータ辞書
        """
        print("\n" + "="*50)
        print("Step 3.1: データ読み込み・前処理")
        print("="*50)
        
        start_time = datetime.now()
        
        # データ読み込み設定
        data_config = self.config.get('data', {})
        
        # 統一データローダーによる読み込み
        print(f"📂 データ読み込み中: {data_path}")
        data_dict = load_experimental_data(data_path, data_config)
        
        # データ統計表示
        metadata: DataMetadata = data_dict['metadata']
        print(f"📊 データ統計:")
        print(f"  - 元データ形状: {metadata.original_shape}")
        print(f"  - 特徴量数: {len(metadata.feature_names)}")
        print(f"  - 欠損値率: {metadata.missing_ratio:.2%}")
        print(f"  - 正規化方法: {metadata.normalization_method}")
        print(f"  - 訓練データ: {data_dict['train'].shape}")
        print(f"  - 検証データ: {data_dict['val'].shape}")
        print(f"  - テストデータ: {data_dict['test'].shape}")
        
        # データの次元数を取得して設定を動的に更新
        data_dim = data_dict['train'].shape[1]  # (T, d) の d を取得
        if 'model' in self.config:
            # エンコーダーの入力次元をデータに合わせて更新
            if 'encoder' in self.config['model']:
                original_input_dim = self.config['model']['encoder'].get('input_dim', data_dim)
                self.config['model']['encoder']['input_dim'] = data_dim
                if original_input_dim != data_dim:
                    print(f"🔧 エンコーダー入力次元を自動調整: {original_input_dim} → {data_dim}")
            
            # デコーダーの出力次元をデータに合わせて更新
            if 'decoder' in self.config['model']:
                original_output_dim = self.config['model']['decoder'].get('output_dim', data_dim)
                self.config['model']['decoder']['output_dim'] = data_dim
                if original_output_dim != data_dim:
                    print(f"🔧 デコーダー出力次元を自動調整: {original_output_dim} → {data_dim}")
        
        # データをデバイスに移動
        for key in ['train', 'val', 'test']:
            data_dict[key] = data_dict[key].to(self.device)
        
        # メタデータ保存
        with open(self.output_dir / 'logs' / 'data_metadata.json', 'w') as f:
            # DataMetadataは__dict__がないため、asdict()を使用
            metadata_dict = {
                'original_shape': metadata.original_shape,
                'feature_names': metadata.feature_names,
                'time_index': metadata.time_index,
                'sampling_rate': metadata.sampling_rate,
                'missing_ratio': metadata.missing_ratio,
                'data_source': metadata.data_source,
                'normalization_method': metadata.normalization_method,
                'train_indices': metadata.train_indices,
                'val_indices': metadata.val_indices,
                'test_indices': metadata.test_indices
            }
            json.dump(metadata_dict, f, indent=2)
        
        # データ可視化
        self._plot_data_overview(data_dict)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ データ前処理完了 ({elapsed:.1f}秒)")
        
        # ログ記録
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'data_loading_complete',
            'elapsed_seconds': elapsed,
            'data_shapes': {k: list(v.shape) for k, v in data_dict.items() if isinstance(v, torch.Tensor)},
            'metadata': metadata_dict
        })
        
        return data_dict
    
    def step_2_training_execution(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Step 3.2: 完全学習パイプライン実行
        
        Args:
            data_dict: 前処理済みデータ
            
        Returns:
            学習結果辞書
        """
        print("\n" + "="*50)
        print("Step 3.2: Phase-1 + Phase-2学習実行")
        print("="*50)
        
        start_time = datetime.now()
        
        # 学習器初期化
        use_kalman = self.config.get('training', {}).get('use_kalman_filtering', False)
        print(f"🔧 Kalmanフィルタリング: {'有効' if use_kalman else '無効'}")
        
        # 設定辞書を使用してTwoStageTrainerを直接初期化
        trainer = TwoStageTrainer(
            config=self.config,
            device=self.device,
            output_dir=str(self.output_dir),
            use_kalman_filtering=use_kalman
        )
        
        # Phase-1学習
        print("🏃‍♂️ Phase-1学習開始...")
        phase1_start = datetime.now()
        
        phase1_results = trainer.train_phase1(data_dict['train'])
        
        phase1_elapsed = (datetime.now() - phase1_start).total_seconds()
        print(f"✅ Phase-1完了 ({phase1_elapsed:.1f}秒)")
        
        # Phase-2学習（End-to-end微調整）
        print("🏃‍♂️ Phase-2学習開始...")
        phase2_start = datetime.now()
        
        phase2_results = trainer.train_phase2(data_dict['train'], data_dict['val'])
        
        phase2_elapsed = (datetime.now() - phase2_start).total_seconds()
        print(f"✅ Phase-2完了 ({phase2_elapsed:.1f}秒)")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ 全学習完了 ({total_elapsed:.1f}秒)")
        
        # 学習結果統合
        training_results = {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'total_time': total_elapsed,
            'phase1_time': phase1_elapsed,
            'phase2_time': phase2_elapsed,
            'use_kalman': use_kalman
        }
        
        # 学習過程可視化
        self._plot_training_progress(training_results)
        
        # 学習結果保存
        results_path = self.output_dir / 'logs' / 'training_results.json'
        with open(results_path, 'w') as f:
            # Tensor等はJSON非対応のため、シリアライズ可能な形式に変換
            serializable_results = self._make_json_serializable(training_results)
            json.dump(serializable_results, f, indent=2)
        
        # ログ記録
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'training_complete',
            'total_time': total_elapsed,
            'phase1_time': phase1_elapsed,
            'phase2_time': phase2_elapsed,
            'use_kalman': use_kalman
        })
        
        return {
            'trainer': trainer,
            'results': training_results
        }
    
    def step_3_model_analysis(self, trainer: TwoStageTrainer, data_dict: Dict[str, torch.Tensor]):
        """
        Step 3.3: 学習済み転送作用素の表現確認
        
        Args:
            trainer: 学習済み学習器
            data_dict: データ辞書
        """
        print("\n" + "="*50)
        print("Step 3.3: 転送作用素・表現分析")
        print("="*50)
        
        start_time = datetime.now()
        
        # 転送作用素取得・保存
        operators_info = self._analyze_transfer_operators(trainer)
        
        # 内部表現分析
        representations_info = self._analyze_internal_representations(trainer, data_dict)
        
        # 状態空間可視化
        self._visualize_state_space(trainer, data_dict['test'])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ 表現分析完了 ({elapsed:.1f}秒)")
        
        # 分析結果保存
        analysis_results = {
            'operators': operators_info,
            'representations': representations_info,
            'analysis_time': elapsed
        }
        
        with open(self.output_dir / 'logs' / 'model_analysis.json', 'w') as f:
            serializable_analysis = self._make_json_serializable(analysis_results)
            json.dump(serializable_analysis, f, indent=2)
        
        # ログ記録
        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'model_analysis_complete',
            'elapsed_seconds': elapsed
        })
    
    def finalize_experiment(self, trainer: TwoStageTrainer):
        """実験終了処理"""
        print("\n" + "="*50)
        print("実験終了処理")
        print("="*50)
        
        # 最終モデル保存
        model_path = self.output_dir / 'models' / 'final_model.pth'
        trainer._save_inference_ready_model(str(model_path))
        print(f"💾 最終モデル保存: {model_path}")
        
        # 実験設定保存
        config_path = self.output_dir / 'logs' / 'experiment_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # 完全実験ログ保存
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        self.experiment_log.append({
            'timestamp': end_time.isoformat(),
            'event': 'experiment_complete',
            'total_experiment_time': total_time
        })
        
        with open(self.output_dir / 'logs' / 'full_experiment_log.json', 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        print(f"⏱️  総実験時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        print(f"📊 実験完了: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 全結果保存先: {self.output_dir}")
    
    def _plot_data_overview(self, data_dict: Dict[str, torch.Tensor]):
        """データ概要プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Data Overview', fontsize=14)
        
        # 訓練データの時系列プロット（最初の3次元）
        train_data = data_dict['train'].cpu().numpy()
        for i in range(min(3, train_data.shape[1])):
            axes[0, 0].plot(train_data[:, i], label=f'Feature {i+1}')
        axes[0, 0].set_title('Training Data Time Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # データ分割比率
        sizes = [data_dict['train'].shape[0], data_dict['val'].shape[0], data_dict['test'].shape[0]]
        axes[0, 1].pie(sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
        axes[0, 1].set_title('Data Split Ratio')
        
        # 特徴量分布（訓練データ）
        axes[1, 0].hist(train_data.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Feature Value Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # データ統計
        stats_text = f"""
        Data Shape: {train_data.shape}
        Mean: {train_data.mean():.3f}
        Std: {train_data.std():.3f}
        Min: {train_data.min():.3f}
        Max: {train_data.max():.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='center', fontsize=10)
        axes[1, 1].set_title('Data Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'data_overview.png', dpi=300)
        plt.close()
    
    def _plot_training_progress(self, results: Dict[str, Any]):
        """学習過程可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=14)
        
        # Phase-1損失推移（例：実際のresultsから取得）
        if 'phase1' in results and 'losses' in results['phase1']:
            phase1_losses = results['phase1']['losses']
            axes[0, 0].plot(phase1_losses)
            axes[0, 0].set_title('Phase-1 Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Phase-2損失推移
        if 'phase2' in results and 'losses' in results['phase2']:
            phase2_losses = results['phase2']['losses']
            axes[0, 1].plot(phase2_losses)
            axes[0, 1].set_title('Phase-2 Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # 学習時間比較
        times = [results.get('phase1_time', 0), results.get('phase2_time', 0)]
        axes[1, 0].bar(['Phase-1', 'Phase-2'], times)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # 学習設定情報
        info_text = f"""
        Phase-1 Time: {results.get('phase1_time', 0):.1f}s
        Phase-2 Time: {results.get('phase2_time', 0):.1f}s
        Total Time: {results.get('total_time', 0):.1f}s
        Kalman Used: {results.get('use_kalman', False)}
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                        verticalalignment='center', fontsize=10)
        axes[1, 1].set_title('Training Info')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_progress.png', dpi=300)
        plt.close()
    
    def _analyze_transfer_operators(self, trainer: TwoStageTrainer) -> Dict[str, Any]:
        """転送作用素分析"""
        operators_info = {}
        
        # DF-A転送作用素（V_A, U_A）
        if hasattr(trainer, 'df_state') and trainer.df_state is not None:
            try:
                state_dict = trainer.df_state.get_state_dict()
                if 'V_A' in state_dict:
                    V_A = state_dict['V_A']
                    operators_info['V_A_shape'] = list(V_A.shape)
                    operators_info['V_A_norm'] = float(torch.norm(V_A).item())
                if 'U_A' in state_dict:
                    U_A = state_dict['U_A']
                    operators_info['U_A_shape'] = list(U_A.shape)
                    operators_info['U_A_norm'] = float(torch.norm(U_A).item())
            except Exception as e:
                print(f"⚠️  DF-A転送作用素分析エラー: {e}")
        
        # DF-B転送作用素（V_B, u_B）
        if hasattr(trainer, 'df_obs') and trainer.df_obs is not None:
            try:
                obs_dict = trainer.df_obs.get_state_dict()
                if 'V_B' in obs_dict:
                    V_B = obs_dict['V_B']
                    operators_info['V_B_shape'] = list(V_B.shape)
                    operators_info['V_B_norm'] = float(torch.norm(V_B).item())
                if 'u_B' in obs_dict:
                    u_B = obs_dict['u_B']
                    operators_info['u_B_shape'] = list(u_B.shape)
                    operators_info['u_B_norm'] = float(torch.norm(u_B).item())
            except Exception as e:
                print(f"⚠️  DF-B転送作用素分析エラー: {e}")
        
        # 転送作用素保存
        operators_path = self.output_dir / 'artifacts' / 'transfer_operators.pth'
        try:
            operators_data = {
                'df_state': trainer.df_state.get_state_dict() if hasattr(trainer, 'df_state') and trainer.df_state else None,
                'df_obs': trainer.df_obs.get_state_dict() if hasattr(trainer, 'df_obs') and trainer.df_obs else None
            }
            torch.save(operators_data, operators_path)
            print(f"💾 転送作用素保存: {operators_path}")
        except Exception as e:
            print(f"⚠️  転送作用素保存エラー: {e}")
        
        return operators_info
    
    def _analyze_internal_representations(self, trainer: TwoStageTrainer, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """内部表現分析"""
        representations_info = {}
        
        try:
            # エンコーダ表現分析
            if hasattr(trainer, 'encoder'):
                test_sample = data_dict['test'][:100]  # 最初の100サンプル
                with torch.no_grad():
                    encoded = trainer.encoder(test_sample.unsqueeze(0)).squeeze(0)
                    representations_info['encoder_output_shape'] = list(encoded.shape)
                    representations_info['encoder_output_mean'] = float(encoded.mean().item())
                    representations_info['encoder_output_std'] = float(encoded.std().item())
        
        except Exception as e:
            print(f"⚠️  内部表現分析エラー: {e}")
        
        return representations_info
    
    def _visualize_state_space(self, trainer: TwoStageTrainer, test_data: torch.Tensor):
        """状態空間可視化"""
        try:
            # 状態推定（簡略版）
            with torch.no_grad():
                # エンコード
                if hasattr(trainer, 'encoder'):
                    encoded = trainer.encoder(test_data[:200].unsqueeze(0)).squeeze(0)
                    
                    # 2D プロット（最初の2次元または主成分）
                    if encoded.shape[1] >= 2:
                        plt.figure(figsize=(10, 6))
                        
                        plt.subplot(1, 2, 1)
                        plt.plot(encoded[:, 0].cpu().numpy(), label='State Dim 1')
                        plt.plot(encoded[:, 1].cpu().numpy(), label='State Dim 2')
                        plt.title('State Trajectory (Time Series)')
                        plt.xlabel('Time')
                        plt.ylabel('State Value')
                        plt.legend()
                        plt.grid(True)
                        
                        plt.subplot(1, 2, 2)
                        plt.scatter(encoded[:, 0].cpu().numpy(), encoded[:, 1].cpu().numpy(), 
                                  c=np.arange(len(encoded)), cmap='viridis', alpha=0.6)
                        plt.colorbar(label='Time')
                        plt.title('State Space Plot')
                        plt.xlabel('State Dim 1')
                        plt.ylabel('State Dim 2')
                        plt.grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(self.output_dir / 'plots' / 'state_space_visualization.png', dpi=300)
                        plt.close()
                        
                        print("📊 状態空間可視化完了")
            
        except Exception as e:
            print(f"⚠️  状態空間可視化エラー: {e}")
    
    def _make_json_serializable(self, obj):
        """JSON対応形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj


def parse_args():
    """引数解析"""
    parser = argparse.ArgumentParser(description="タスク3: 完全実験パイプライン実行")
    
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='実験設定ファイル (.yaml)'
    )
    parser.add_argument(
        '--data', '-d', type=str, required=True,
        help='データファイルパス'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='結果出力ディレクトリ'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='計算デバイス (auto選択時はNone)'
    )
    parser.add_argument(
        '--use-kalman', action='store_true',
        help='Kalmanフィルタリング有効化'
    )
    parser.add_argument(
        '--skip-analysis', action='store_true',
        help='Step 3.3 表現分析をスキップ'
    )
    
    return parser.parse_args()


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """実験設定読み込み"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 必須セクションチェック
    required_sections = ['model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"設定に {section} セクションが必要です")
    
    return config


def main():
    """メイン実行関数"""
    args = parse_args()
    
    print("🚀 タスク3: 完全実験パイプライン開始")
    print("="*60)
    
    # 設定読み込み
    config = load_experiment_config(args.config)
    
    # Kalmanフラグ設定
    if args.use_kalman:
        config.setdefault('training', {})['use_kalman_filtering'] = True
    
    # デバイス設定
    device = torch.device(args.device) if args.device else select_device()
    
    # 出力ディレクトリ
    output_dir = Path(args.output)
    
    # パイプライン実行
    pipeline = FullExperimentPipeline(config, output_dir, device)
    
    try:
        # Step 3.1: データ前処理
        data_dict = pipeline.step_1_data_loading(args.data)
        
        # Step 3.2: 完全学習実行
        training_result = pipeline.step_2_training_execution(data_dict)
        
        # Step 3.3: 表現分析（オプション）
        if not args.skip_analysis:
            pipeline.step_3_model_analysis(training_result['trainer'], data_dict)
        
        # 実験完了処理
        pipeline.finalize_experiment(training_result['trainer'])
        
        print("\n🎉 完全実験パイプライン正常終了！")
        print(f"📁 結果: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())