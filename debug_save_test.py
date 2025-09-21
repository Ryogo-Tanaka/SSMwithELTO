#!/usr/bin/env python3
"""
保存・読み込み機能のデバッグテスト
"""

import torch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.evaluation.mode_decomposition import (
    SpectrumAnalyzer,
    TrainedModelSpectrumAnalysis,
    SpectrumResultsSaver
)
from src.ssm.df_state_layer import DFStateLayer


def test_simple_save_load():
    """簡単な保存・読み込みテスト"""
    print("=== 保存・読み込みデバッグテスト ===")

    # 簡単なテストデータ作成
    analyzer = SpectrumAnalyzer(0.1)
    V_A = torch.randn(4, 4)
    results = analyzer.analyze_spectrum(V_A)

    print(f"元のresultsキー: {list(results.keys())}")

    # V_Aを含む結果作成
    test_results = {
        'spectrum': results,
        'V_A': V_A,
        'test_value': 42
    }

    # 保存テスト
    test_path = "debug_test"

    print(f"\n--- JSON保存テスト ---")
    try:
        SpectrumResultsSaver.save_results(test_results, test_path, 'json')
        loaded_json = SpectrumResultsSaver.load_results(test_path + '.json', 'json')
        print(f"JSON読み込み成功")
        print(f"JSONキー: {list(loaded_json.keys())}")
        if 'spectrum' in loaded_json:
            print(f"spectrumキー: {list(loaded_json['spectrum'].keys())[:5]}...")
    except Exception as e:
        print(f"JSONエラー: {e}")

    print(f"\n--- NPZ保存テスト ---")
    try:
        SpectrumResultsSaver.save_results(test_results, test_path, 'npz')
        loaded_npz = SpectrumResultsSaver.load_results(test_path + '.npz', 'npz')
        print(f"NPZ読み込み成功")
        print(f"NPZキー: {list(loaded_npz.keys())}")

        # V_A関連キーの確認
        va_keys = [k for k in loaded_npz.keys() if 'V_A' in k]
        print(f"V_A関連キー: {va_keys}")

    except Exception as e:
        print(f"NPZエラー: {e}")

    # クリーンアップ
    Path(test_path + '.json').unlink(missing_ok=True)
    Path(test_path + '.npz').unlink(missing_ok=True)


def test_df_state_layer():
    """DFStateLayerの動作確認"""
    print("\n=== DFStateLayer動作確認 ===")

    try:
        # DFStateLayer作成
        df_layer = DFStateLayer(
            state_dim=3,
            feature_dim=5,
            lambda_A=1e-3,
            lambda_B=1e-3
        )

        print(f"DFStateLayer作成成功: {type(df_layer)}")
        print(f"state_dim: {df_layer.state_dim}, feature_dim: {df_layer.feature_dim}")

        # ダミーV_A設定
        with torch.no_grad():
            df_layer.V_A = torch.eye(5) + 0.1 * torch.randn(5, 5)
            df_layer.U_A = torch.randn(5, 3)
            df_layer._is_fitted = True

        # V_A取得テスト
        V_A = df_layer.get_transfer_operator()
        print(f"V_A取得成功: shape={V_A.shape}")

        # スペクトル分析テスト
        analyzer = SpectrumAnalyzer(0.1)
        spectrum = analyzer.analyze_spectrum(V_A)
        print(f"スペクトル分析成功: 固有値数={len(spectrum['eigenvalues_discrete'])}")

        return True

    except Exception as e:
        print(f"DFStateLayerエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_save_load()
    success = test_df_state_layer()

    if success:
        print("\n✅ 基本機能は正常動作")
    else:
        print("\n❌ 基本機能にエラーあり")