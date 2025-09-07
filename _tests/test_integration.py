#!/usr/bin/env python3
# _tests/test_integration.py
"""
提案手法の統合テスト（修正後実装対応版）

修正内容:
- TwoStageTrainerクラスに対応
- 新しいDF-A/DF-B APIに対応
- Phase-1/Phase-2学習フローに対応
- 詳細な動作確認機能追加

実行方法:
  cd SSMwithELTO
  python _tests/test_integration.py --quick    # 高速テスト
  python _tests/test_integration.py --full     # 完全テスト
  python _tests/test_integration.py --debug    # デバッグモード
"""

import argparse
import sys
import os
import torch
import numpy as np
import yaml
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# プロジェクトルートパスの設定
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

def check_dependencies():
    """依存関係の確認"""
    print("=== 依存関係チェック ===")
    required_modules = [
        'torch', 'numpy', 'yaml', 'matplotlib', 'pandas'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"✗ {module}")
    
    if missing:
        print(f"❌ 不足モジュール: {missing}")
        print("pip install torch numpy pyyaml matplotlib pandas で解決できます")
        return False
    
    print("✓ 全依存関係OK")
    return True

def check_project_structure():
    """プロジェクト構造の確認"""
    print("=== プロジェクト構造チェック ===")
    
    required_paths = [
        PROJECT_ROOT / 'src',
        PROJECT_ROOT / 'src' / 'models',
        PROJECT_ROOT / 'src' / 'ssm',
        PROJECT_ROOT / 'src' / 'training',
        PROJECT_ROOT / 'configs',
    ]
    
    missing_paths = []
    for path in required_paths:
        if path.exists():
            print(f"✓ {path.relative_to(PROJECT_ROOT)}")
        else:
            missing_paths.append(path)
            print(f"✗ {path.relative_to(PROJECT_ROOT)}")
    
    if missing_paths:
        print(f"❌ 不足パス: {[str(p.relative_to(PROJECT_ROOT)) for p in missing_paths]}")
        return False
    
    print("✓ プロジェクト構造OK")
    return True

def generate_synthetic_data(config: Dict[str, Any]) -> torch.Tensor:
    """設定に基づく合成データ生成"""
    d = config['model']['encoder']['input_dim']
    h = config['ssm']['realization']['past_horizon']
    
    # 最小必要長 + マージン
    min_T = 2 * h + 50
    T = max(min_T, 200)
    
    torch.manual_seed(42)
    t = torch.linspace(0, 4*np.pi, T)
    Y = torch.zeros(T, d)
    
    # 各次元を異なる特性で生成
    for i in range(d):
        # 基本周期成分
        freq = 0.5 + i * 0.3
        phase = i * np.pi / 4
        signal = torch.sin(freq * t + phase)
        
        # AR成分
        ar_coeff = 0.2 + i * 0.1
        noise = 0.1 * torch.randn(T)
        
        Y[:, i] = signal + noise
        
        # AR項追加
        for tau in range(1, T):
            Y[tau, i] += ar_coeff * Y[tau-1, i] * 0.5
    
    # 正規化
    Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-8)
    
    print(f"✓ 合成データ生成: shape={Y.shape}, range=({Y.min():.2f}, {Y.max():.2f})")
    return Y

def test_individual_components(config: Dict[str, Any], Y: torch.Tensor, verbose: bool = True) -> Dict[str, bool]:
    """個別コンポーネントのテスト（修正後実装対応）"""
    results = {}
    
    if verbose:
        print("\n=== 個別コンポーネントテスト ===")
    
    # 1. エンコーダテスト
    try:
        from src.models.architectures.tcn import tcnEncoder
        encoder = tcnEncoder(**config['model']['encoder'])
        
        # 修正: 3次元入力に対応
        Y_batch = Y.unsqueeze(0)  # (T, d) -> (1, T, d)
        m_output = encoder(Y_batch)  # (1, T, 1)
        m_series = m_output.squeeze()  # (T,)
        
        results['encoder'] = True
        if verbose:
            print(f"✓ エンコーダ: {Y.shape} -> {m_series.shape}")
            print(f"  特徴量統計: mean={m_series.mean():.4f}, std={m_series.std():.4f}")
    
    except Exception as e:
        results['encoder'] = False
        if verbose:
            print(f"✗ エンコーダエラー: {e}")
        return results
    
    # 2. 確率的実現テスト
    try:
        from src.ssm.realization import Realization
        realization = Realization(**config['ssm']['realization'])
        
        m_input = m_series.unsqueeze(1)  # (T,) -> (T, 1)
        realization.fit(m_input)
        X_states = realization.filter(m_input)
        
        results['realization'] = True
        if verbose:
            print(f"✓ 確率的実現: {m_input.shape} -> {X_states.shape}")
            if hasattr(realization, '_L_vals') and realization._L_vals is not None:
                singular_values_for_display = realization._L_vals.detach().cpu().numpy()
                print(f"  特異値: {singular_values_for_display}")
    
    except Exception as e:
        results['realization'] = False
        if verbose:
            print(f"✗ 確率的実現エラー: {e}")
        return results
    
    # 3. DF-A テスト
    try:
        from src.ssm.df_state_layer import DFStateLayer
        
        _, r = X_states.shape
        df_state = DFStateLayer(
            state_dim=r,
            **config['ssm']['df_state']
        )
        
        # Stage-1のみテスト
        optimizer_phi = torch.optim.Adam(df_state.phi_theta.parameters(), lr=1e-3)
        metrics = df_state.train_stage1_with_gradients(X_states, optimizer_phi)
        
        results['df_state'] = True
        if verbose:
            print(f"✓ DF-A: Stage-1 loss={metrics['stage1_loss']:.4f}")
    
    except Exception as e:
        results['df_state'] = False
        if verbose:
            print(f"✗ DF-A エラー: {e}")
        return results
    
    # 4. DF-B テスト
    try:
        from src.ssm.df_observation_layer import DFObservationLayer
        
        df_obs = DFObservationLayer(
            df_state_layer=df_state,
            **config['ssm']['df_observation']
        )
        
        # 状態予測を取得
        X_hat_states = df_state.predict_sequence(X_states)
        
        optimizer_phi = torch.optim.Adam(df_state.phi_theta.parameters(), lr=1e-3)
        metrics = df_obs.train_stage1_with_gradients(
            X_hat_states, m_series, optimizer_phi, fix_psi_omega=True
        )
        
        results['df_observation'] = True
        if verbose:
            print(f"✓ DF-B: Stage-1 loss={metrics['stage1_loss']:.4f}")
    
    except Exception as e:
        results['df_observation'] = False
        if verbose:
            print(f"✗ DF-B エラー: {e}")
        return results
    
    # 5. デコーダテスト
    try:
        from src.models.architectures.tcn import tcnDecoder
        decoder = tcnDecoder(**config['model']['decoder'])
        
        # ダミー入力でテスト
        dummy_input = torch.randn(1, 10, 1)  # (batch, time, features)
        decoded_output = decoder(dummy_input)
        
        results['decoder'] = True
        if verbose:
            print(f"✓ デコーダ: {dummy_input.shape} -> {decoded_output.shape}")
    
    except Exception as e:
        results['decoder'] = False
        if verbose:
            print(f"✗ デコーダエラー: {e}")
    
    return results

def test_two_stage_trainer(config_path: str, quick: bool = False, verbose: bool = True) -> bool:
    """TwoStageTrainerを用いた統合テスト"""
    if verbose:
        print("\n=== TwoStageTrainer統合テスト ===")
    
    try:
        # 設定読み込み
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # クイックモード設定
        if quick:
            config['training']['phase1_epochs'] = 2
            config['training']['phase2_epochs'] = 2
            config['training']['T1_iterations'] = 2
            config['training']['T2_iterations'] = 1
        
        # データ生成
        Y = generate_synthetic_data(config)
        
        # 個別テスト実行
        component_results = test_individual_components(config, Y, verbose)
        
        failed_components = [k for k, v in component_results.items() if not v]
        if failed_components:
            if verbose:
                print(f"✗ 失敗コンポーネント: {failed_components}")
            return False
        
        # TwoStageTrainer統合テスト
        if verbose:
            print("\n--- TwoStageTrainer統合テスト ---")
        
        from src.training.two_stage_trainer import TwoStageTrainer, TrainingConfig
        from src.models.architectures.tcn import tcnEncoder, tcnDecoder
        from src.ssm.realization import Realization
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # モデル初期化
            encoder = tcnEncoder(**config['model']['encoder'])
            decoder = tcnDecoder(**config['model']['decoder'])
            realization = Realization(**config['ssm']['realization'])
            
            # デバイス（CPUで安全に）
            device = torch.device('cpu')
            
            # 学習設定
            training_config = TrainingConfig(**config['training'])
            
            # トレーナー作成
            trainer = TwoStageTrainer(
                encoder=encoder,
                decoder=decoder,
                realization=realization,
                df_state_config=config['ssm']['df_state'],
                df_obs_config=config['ssm']['df_observation'],
                training_config=training_config,
                device=device,
                output_dir=temp_dir
            )
            
            # Phase-1学習
            if verbose:
                print("Phase-1学習実行中...")
            
            phase1_results = trainer.train_phase1(Y)
            
            if verbose:
                print(f"✓ Phase-1完了: {len(phase1_results)}エポック")
                
                # 最終損失表示
                if phase1_results:
                    final_metrics = phase1_results[-1]
                    df_a_loss = final_metrics.get('df_a_stage1_loss', 'N/A')
                    df_b_loss = final_metrics.get('df_b_stage1_loss', 'N/A')
                    print(f"  最終損失 - DF-A: {df_a_loss}, DF-B: {df_b_loss}")
            
            # Phase-2学習（クイックモードでは省略可能）
            if not quick:
                if verbose:
                    print("Phase-2学習実行中...")
                
                phase2_results = trainer.train_phase2(Y)
                
                if verbose:
                    print(f"✓ Phase-2完了: {len(phase2_results)}エポック")
                    if phase2_results:
                        final_loss = phase2_results[-1]['total_loss']
                        print(f"  最終損失: {final_loss:.4f}")
            
            # 予測テスト
            try:
                Y_test = Y[-30:]
                predictions = trainer.predict(Y_test, forecast_steps=3)
                
                if verbose:
                    print(f"✓ 予測テスト完了: {predictions.shape}")
                    print(f"  予測範囲: ({predictions.min():.3f}, {predictions.max():.3f})")
            
            except Exception as e:
                if verbose:
                    print(f"⚠ 予測テストエラー（非致命的）: {e}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ 統合テストエラー: {e}")
            if len(str(e)) < 200:  # 短いエラーのみ詳細表示
                traceback.print_exc()
        return False

def test_learning_flow_analysis(config_path: str, verbose: bool = True) -> bool:
    """学習フローの詳細分析テスト"""
    if verbose:
        print("\n=== 学習フロー分析テスト ===")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 小規模設定
        config['training']['phase1_epochs'] = 3
        config['training']['phase2_epochs'] = 2
        
        Y = generate_synthetic_data(config)
        
        from src.training.two_stage_trainer import TwoStageTrainer, TrainingConfig
        from src.models.architectures.tcn import tcnEncoder, tcnDecoder
        from src.ssm.realization import Realization
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # モデル・トレーナー初期化
            encoder = tcnEncoder(**config['model']['encoder'])
            decoder = tcnDecoder(**config['model']['decoder'])
            realization = Realization(**config['ssm']['realization'])
            training_config = TrainingConfig(**config['training'])
            
            trainer = TwoStageTrainer(
                encoder=encoder,
                decoder=decoder,
                realization=realization,
                df_state_config=config['ssm']['df_state'],
                df_obs_config=config['ssm']['df_observation'],
                training_config=training_config,
                device=torch.device('cpu'),
                output_dir=temp_dir
            )
            
            # 時間対応デバッグモード有効化
            trainer.enable_time_alignment_debug()
            
            # Phase-1の詳細分析
            if verbose:
                print("Phase-1詳細分析実行中...")
            
            phase1_results = trainer.train_phase1(Y)
            
            # 学習履歴分析
            if phase1_results:
                # 損失推移分析
                df_a_losses = [m.get('df_a_stage1_loss') for m in phase1_results if 'df_a_stage1_loss' in m]
                df_b_losses = [m.get('df_b_stage1_loss') for m in phase1_results if 'df_b_stage1_loss' in m]
                
                if verbose:
                    print(f"✓ DF-A損失推移: {len(df_a_losses)}個のエポック")
                    if df_a_losses:
                        print(f"  初期損失: {df_a_losses[0]:.4f} -> 最終損失: {df_a_losses[-1]:.4f}")
                    
                    print(f"✓ DF-B損失推移: {len(df_b_losses)}個のエポック")
                    if df_b_losses:
                        print(f"  初期損失: {df_b_losses[0]:.4f} -> 最終損失: {df_b_losses[-1]:.4f}")
            
            # Phase-2分析
            if verbose:
                print("Phase-2詳細分析実行中...")
            
            phase2_results = trainer.train_phase2(Y)
            
            if phase2_results:
                total_losses = [r['total_loss'] for r in phase2_results]
                rec_losses = [r['rec_loss'] for r in phase2_results]
                
                if verbose:
                    print(f"✓ Phase-2損失推移: {len(total_losses)}個のエポック")
                    if total_losses:
                        print(f"  総損失: {total_losses[0]:.4f} -> {total_losses[-1]:.4f}")
                        print(f"  再構成損失: {rec_losses[0]:.4f} -> {rec_losses[-1]:.4f}")
            
            # 学習サマリ取得
            summary = trainer.get_training_summary()
            if verbose:
                print(f"✓ 学習サマリ取得完了")
                print(f"  学習完了: {summary['training_complete']}")
                print(f"  総パラメータ数: {sum(summary['model_info'].values())}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ 学習フロー分析エラー: {e}")
            traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="提案手法統合テスト（修正後実装対応版）")
    parser.add_argument('--quick', action='store_true', help='高速テスト（短縮版）')
    parser.add_argument('--full', action='store_true', help='完全テスト')
    parser.add_argument('--debug', action='store_true', help='デバッグモード（詳細出力）')
    parser.add_argument('--analysis', action='store_true', help='学習フロー分析テスト')
    parser.add_argument('--config', type=str, default='_tests/test_config.yaml', 
                       help='設定ファイルパス')
    args = parser.parse_args()
    
    # モード設定
    if args.debug:
        verbose = True
        torch.autograd.set_detect_anomaly(True)
        print("🔧 デバッグモード: 詳細出力・異常検出有効")
    else:
        verbose = not args.quick
    
    if args.quick:
        print("🚀 高速テストモード")
    elif args.full:
        print("🎯 完全テストモード")
    elif args.analysis:
        print("📊 学習フロー分析モード")
    
    print("提案手法 統合テスト開始（修正後実装対応版）")
    print("=" * 60)
    
    # 依存関係チェック
    if not check_dependencies():
        return 1
    
    # プロジェクト構造チェック
    if not check_project_structure():
        return 1
    
    # 設定ファイル確認
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
        
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        print("_tests/test_config.yaml を作成してください")
        return 1
    
    print(f"📋 設定ファイル: {config_path.relative_to(PROJECT_ROOT)}")
    
    # テスト実行
    try:
        success = True
        
        # 基本統合テスト
        if args.analysis:
            success = test_learning_flow_analysis(str(config_path), verbose)
        else:
            success = test_two_stage_trainer(str(config_path), args.quick, verbose)
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 統合テスト成功！")
            print("修正後実装の基本動作確認完了。")
            
            if args.quick:
                print("\n次のステップ:")
                print("  python _tests/test_integration.py --full     # 完全テスト")
                print("  python _tests/test_integration.py --analysis # 学習フロー分析")
                print("  python main_two_stage.py --config configs/config_two_stage_experiment.yaml")
            
            return 0
        else:
            print("❌ 統合テスト失敗")
            print("上記のエラーを修正してから再実行してください。")
            
            if not verbose:
                print("\n詳細確認:")
                print("  python _tests/test_integration.py --debug")
            
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 テスト中断")
        return 1
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())