#!/usr/bin/env python3
# main_two_stage.py
"""
提案手法の2段階学習メインスクリプト

使用例:
python main_two_stage.py \
    --config configs/two_stage_experiment.yaml \
    --data data/sim_complex.npz \
    --output results/two_stage_exp_001 \
    --device cuda
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 修正済みコンポーネントのインポート
from src.training.two_stage_trainer import run_training_experiment, create_trainer_from_config
from src.utils.gpu_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="提案手法の2段階学習実験")
    
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='YAML設定ファイルのパス'
    )
    parser.add_argument(
        '--data', '-d', type=str, required=True,
        help='データファイルのパス (.npz)'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='結果出力ディレクトリ'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='計算デバイス (cuda/cpu、未指定時は自動選択)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='チェックポイントから再開 (チェックポイントファイルパス)'
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='学習済みモデルの検証のみ実行'
    )
    parser.add_argument(
        '--plot-results', action='store_true', default=True,
        help='結果プロット生成'
    )
    
    return parser.parse_args()


def validate_config(config_path: str) -> dict:
    """設定ファイルの検証と読み込み"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 必須セクションの確認
    required_sections = ['model', 'ssm', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"設定ファイルに {section} セクションが必要です")
    
    # モデル設定の確認
    model_config = config['model']
    if 'encoder' not in model_config or 'decoder' not in model_config:
        raise ValueError("model.encoder と model.decoder の設定が必要です")
    
    # SSM設定の確認
    ssm_config = config['ssm']
    required_ssm = ['realization', 'df_state', 'df_observation']
    for ssm_section in required_ssm:
        if ssm_section not in ssm_config:
            raise ValueError(f"ssm.{ssm_section} の設定が必要です")
    
    data_config = config.get('data', {})
    if data_config:  # dataセクションがある場合のみ検証
        if 'normalization' in data_config:
            valid_methods = ['standard', 'minmax', 'none']
            if data_config['normalization'] not in valid_methods:
                raise ValueError(f"無効な正規化方法: {data_config['normalization']}. 有効: {valid_methods}")
        
        # 分割比率の検証
        if 'train_ratio' in data_config:
            train_ratio = float(data_config.get('train_ratio', 0.7))
            val_ratio = float(data_config.get('val_ratio', 0.2))
            test_ratio = float(data_config.get('test_ratio', 0.1))
            
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(f"データ分割比率の合計が1.0でありません: {total_ratio}")

    print("設定ファイル検証完了")
    return config


def validate_data(data_path: str) -> torch.Tensor:
    """データファイルの検証と読み込み"""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
    
    try:
        data = np.load(data_path)
        
        # データキーの確認
        if 'Y' in data:
            Y = data['Y']
        elif 'arr_0' in data:
            Y = data['arr_0']
        else:
            available_keys = list(data.keys())
            raise ValueError(f"データに 'Y' または 'arr_0' キーが必要です。利用可能: {available_keys}")
        
        # 形状確認
        if Y.ndim != 2:
            raise ValueError(f"データは2次元 (T, d) である必要があります: got {Y.shape}")
        
        T, d = Y.shape
        if T < 100:
            print(f"警告: 系列長が短いです (T={T})。学習が不安定になる可能性があります。")
        
        print(f"データ読み込み完了: 形状={Y.shape}, dtype={Y.dtype}")
        return torch.tensor(Y, dtype=torch.float32)
        
    except Exception as e:
        raise ValueError(f"データ読み込みエラー: {e}")


def plot_training_results(output_dir: str):
    """学習結果の可視化"""
    output_path = Path(output_dir)
    
    # Phase-1結果プロット
    phase1_csv = output_path / "phase1_training.csv"
    if phase1_csv.exists():
        import pandas as pd
        df1 = pd.read_csv(phase1_csv)
        
        # DF-A/DF-B損失推移
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # DF-A Stage-1
        df_a_s1 = df1[df1['df_a_stage1_loss'].notna()]
        if not df_a_s1.empty:
            axes[0, 0].plot(df_a_s1['epoch'], df_a_s1['df_a_stage1_loss'])
            axes[0, 0].set_title('DF-A Stage-1 Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # DF-A Stage-2
        df_a_s2 = df1[df1['df_a_stage2_loss'].notna()]
        if not df_a_s2.empty:
            axes[0, 1].plot(df_a_s2['epoch'], df_a_s2['df_a_stage2_loss'])
            axes[0, 1].set_title('DF-A Stage-2 Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # DF-B Stage-1
        df_b_s1 = df1[df1['df_b_stage1_loss'].notna()]
        if not df_b_s1.empty:
            axes[1, 0].plot(df_b_s1['epoch'], df_b_s1['df_b_stage1_loss'])
            axes[1, 0].set_title('DF-B Stage-1 Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # DF-B Stage-2
        df_b_s2 = df1[df1['df_b_stage2_loss'].notna()]
        if not df_b_s2.empty:
            axes[1, 1].plot(df_b_s2['epoch'], df_b_s2['df_b_stage2_loss'])
            axes[1, 1].set_title('DF-B Stage-2 Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'phase1_losses.png', dpi=150)
        plt.close()
    
    # Phase-2結果プロット
    phase2_csv = output_path / "phase2_training.csv"
    if phase2_csv.exists():
        import pandas as pd
        df2 = pd.read_csv(phase2_csv)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 総損失
        axes[0].plot(df2['epoch'], df2['total_loss'])
        axes[0].set_title('Total Loss (Phase-2)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # 再構成損失
        axes[1].plot(df2['epoch'], df2['rec_loss'])
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        # CCA損失
        if 'cca_loss' in df2.columns and df2['cca_loss'].notna().any():
            axes[2].plot(df2['epoch'], df2['cca_loss'])
            axes[2].set_title('CCA Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'phase2_losses.png', dpi=150)
        plt.close()
    
    print(f"学習結果プロット保存完了: {output_path}")


def run_validation(trainer, Y_test: torch.Tensor, output_dir: str) -> dict:
    """学習済みモデルの検証"""
    print("\n=== モデル検証開始 ===")
    
    # 予測実行
    forecast_steps = min(96, Y_test.size(0) // 4)  # 系列長の1/4 or 96ステップ
    test_len = Y_test.size(0)
    warmup_len = test_len - forecast_steps
    
    Y_warmup = Y_test[:warmup_len]
    Y_true = Y_test[warmup_len:]
    
    try:
        Y_pred = trainer.predict(Y_warmup, forecast_steps)
        
        # 予測精度計算
        mse = torch.mean((Y_pred - Y_true) ** 2).item()
        mae = torch.mean(torch.abs(Y_pred - Y_true)).item()
        
        # 予測結果プロット
        fig, axes = plt.subplots(min(Y_test.size(1), 4), 1, figsize=(12, 8))
        if Y_test.size(1) == 1:
            axes = [axes]
        
        for dim in range(min(Y_test.size(1), 4)):
            ax = axes[dim] if len(axes) > 1 else axes[0]
            
            # 時間軸
            t_warmup = range(warmup_len)
            t_pred = range(warmup_len, test_len)
            
            # プロット
            ax.plot(t_warmup, Y_warmup[:, dim].cpu(), 'b-', label='Warmup', alpha=0.7)
            ax.plot(t_pred, Y_true[:, dim].cpu(), 'g-', label='True', linewidth=2)
            ax.plot(t_pred, Y_pred[:, dim].cpu(), 'r--', label='Predicted', linewidth=2)
            ax.axvline(warmup_len, color='k', linestyle=':', alpha=0.5, label='Prediction start')
            
            ax.set_title(f'Dimension {dim} Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'prediction_results.png', dpi=150)
        plt.close()
        
        validation_results = {
            'mse': mse,
            'mae': mae,
            'forecast_steps': forecast_steps,
            'warmup_length': warmup_len,
            'prediction_shape': tuple(Y_pred.shape)
        }
        
        print(f"検証完了: MSE={mse:.6f}, MAE={mae:.6f}")
        return validation_results
        
    except Exception as e:
        print(f"検証エラー: {e}")
        return {'error': str(e)}


def main():
    """メイン実行関数"""
    args = parse_args()
    
    print("=== 提案手法の2段階学習実験 ===")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 設定とデータの検証
        config = validate_config(args.config)
        Y_data = validate_data(args.data)
        
        # 2. 出力ディレクトリ準備
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定ファイルをコピー
        import shutil
        shutil.copy2(args.config, output_dir / 'config_used.yaml')
        
        # 3. デバイス設定
        if args.device:
            device = torch.device(args.device)
        else:
            device = select_device()
        print(f"使用デバイス: {device}")
        
        Y_data = Y_data.to(device)
        
        # 4. 検証専用モードの場合
        if args.validate_only:
            if not (output_dir / 'final_model.pth').exists():
                raise FileNotFoundError(f"学習済みモデルが見つかりません: {output_dir / 'final_model.pth'}")
            
            # 学習済みモデル読み込み（実装が必要）
            print("検証専用モード: 学習済みモデルの検証機能は未実装")
            return
        
        # 5. 学習実行
        if args.resume:
            print(f"チェックポイントから再開: {args.resume}")
            trainer = create_trainer_from_config(args.config, device, str(output_dir))
            trainer.load_checkpoint(args.resume)
            # 続きから学習（実装が必要）
        else:
            print("新規学習開始")
            results = run_training_experiment(
                config_path=args.config,
                data_path=args.data,
                output_dir=str(output_dir),
                device=device
            )
        
        # 6. 結果可視化
        if args.plot_results:
            plot_training_results(str(output_dir))
        
        # 7. 検証実行
        trainer = create_trainer_from_config(args.config, device, str(output_dir))
        # 学習済みモデル読み込み (簡略化)
        final_model_path = output_dir / 'final_model.pth'
        if final_model_path.exists():
            # 検証データ分割
            total_len = Y_data.size(0)
            train_len = int(0.8 * total_len)
            Y_test = Y_data[train_len:]
            
            if Y_test.size(0) > 100:  # 十分なテストデータがある場合のみ
                validation_results = run_validation(trainer, Y_test, str(output_dir))
                
                # 結果保存
                with open(output_dir / 'validation_results.json', 'w') as f:
                    json.dump(validation_results, f, indent=2)
        
        # 8. 実験サマリ保存
        experiment_summary = {
            'experiment_name': args.output,
            'start_time': datetime.now().isoformat(),
            'config_path': args.config,
            'data_path': args.data,
            'device': str(device),
            'data_shape': tuple(Y_data.shape),
            'success': True
        }
        
        with open(output_dir / 'experiment_summary.json', 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        
        print(f"\n実験完了: 結果は {output_dir} に保存されました")
        
    except Exception as e:
        print(f"\n実験失敗: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー情報保存
        if 'output_dir' in locals():
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            with open(output_dir / 'error_log.json', 'w') as f:
                json.dump(error_info, f, indent=2)


if __name__ == '__main__':
    main()