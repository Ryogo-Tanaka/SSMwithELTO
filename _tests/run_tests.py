#!/usr/bin/env python3
# _tests/run_tests.py
"""
テスト実行用ユーティリティスクリプト

使用例:
  python _tests/run_tests.py               # デフォルトテスト
  python _tests/run_tests.py --quick       # 高速テスト
  python _tests/run_tests.py --debug       # デバッグモード
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="提案手法テスト実行ユーティリティ")
    parser.add_argument('--quick', action='store_true', help='高速テスト（短縮版）')
    parser.add_argument('--full', action='store_true', help='完全テスト')
    parser.add_argument('--debug', action='store_true', help='デバッグモード（詳細出力）')
    parser.add_argument('--component', choices=['encoder', 'realization', 'df', 'integration'], 
                       help='特定コンポーネントのみテスト')
    args = parser.parse_args()
    
    # プロジェクトルートに移動
    project_root = Path(__file__).parent.parent
    
    # テストスクリプトのパス
    test_script = project_root / "_tests" / "test_integration.py"
    
    if not test_script.exists():
        print(f"エラー: テストスクリプトが見つかりません: {test_script}")
        return 1
    
    # 実行コマンド構築
    cmd = [sys.executable, str(test_script)]
    
    if args.quick:
        cmd.append('--quick')
    elif args.full:
        cmd.append('--full')
    
    if args.debug:
        cmd.append('--debug')
    
    print(f"実行コマンド: {' '.join(cmd)}")
    print("=" * 50)
    
    # テスト実行
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nテスト中断")
        return 1
    except Exception as e:
        print(f"実行エラー: {e}")
        return 1

if __name__ == "__main__":
    exit(main())