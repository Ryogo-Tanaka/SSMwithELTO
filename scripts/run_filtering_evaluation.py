#!/usr/bin/env python3
"""
タスク4統合実行スクリプト

DFIV Kalman Filterのフィルタリング状態推定・評価を統合実行。
- フィルタリング性能評価
- 推定手法比較 (Kalman vs 決定的)
- 不確実性定量化評価
- 結果の完全出力・保存

Usage:
    # 基本実行
    python scripts/run_filtering_evaluation.py \
        --model results/trained_model.pth \
        --data data/test.npz \
        --output results/task4_evaluation

    # 包括的評価
    python scripts/run_filtering_evaluation.py \
        --model results/trained_model.pth \
        --data data/test.npz \
        --output results/comprehensive_eval \
        --config configs/evaluation_config.yaml \
        --mode comprehensive

    # クイックテスト
    python scripts/run_filtering_evaluation.py \
        --model results/trained_model.pth \
        --data data/test.npz \
        --output results/quick_test \
        --mode quick
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# プロジェクト設定
sys.path.append(str(Path(__file__).parent.parent))

# 評価モジュール
from scripts.evaluate_filtering_performance import FilteringPerformanceEvaluator
from scripts.compare_estimation_methods import EstimationMethodComparator


class Task4EvaluationPipeline:
    """タスク4統合評価パイプライン"""
    
    def __init__(
        self, 
        model_path: str, 
        config_path: str,
        output_dir: str,
        device: str = 'auto'
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 出力ディレクトリ構造作成
        self._setup_output_structure()
        
        # 設定読み込み
        self.config = self._load_config()
        
        # 実験ログ初期化
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'device': self.device,
            'stages': []
        }
        
    def _setup_output_structure(self):
        """出力ディレクトリ構造の作成"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # サブディレクトリ作成
        subdirs = [
            'filtering_performance',    # フィルタリング性能結果
            'method_comparison',        # 手法比較結果
            'uncertainty_analysis',     # 不確実性分析結果
            'summary',                  # 統合サマリ
            'visualizations',           # 可視化結果
            'raw_data'                  # 生データ出力
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️  設定ファイル読み込みエラー: {e}")
            print("📝 デフォルト設定を使用")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定の取得"""
        return {
            'evaluation': {
                'experiment_name': f'task4_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'save_detailed_results': True,
                'create_visualizations': True,
                'data': {'test_split': 'test'}
            },
            'filtering': {'batch': {'return_likelihood': True}},
            'uncertainty_analysis': {'enabled': True},
            'visualization': {'enabled': True}
        }
    
    def run_comprehensive_evaluation(
        self, 
        data_path: str,
        mode: str = 'standard'
    ) -> Dict[str, Any]:
        """
        包括的評価の実行
        
        Args:
            data_path: 評価データパス
            mode: 評価モード ('quick', 'standard', 'comprehensive')
            
        Returns:
            統合評価結果
        """
        print(f"\n🚀 タスク4統合評価開始")
        print(f"📊 評価モード: {mode}")
        print(f"📁 モデル: {self.model_path}")
        print(f"📈 データ: {data_path}")
        print(f"📂 出力: {self.output_dir}")
        print("="*70)
        
        evaluation_results = {}
        
        # モード別設定調整
        adjusted_config = self._adjust_config_for_mode(mode)
        
        # Stage 1: フィルタリング性能評価
        filtering_results = self._run_filtering_performance_evaluation(
            data_path, adjusted_config
        )
        evaluation_results['filtering_performance'] = filtering_results
        
        # Stage 2: 推定手法比較
        comparison_results = self._run_method_comparison_evaluation(
            data_path, adjusted_config
        )
        evaluation_results['method_comparison'] = comparison_results
        
        # Stage 3: 統合分析・結果整理
        integrated_results = self._integrate_and_summarize_results(
            evaluation_results, adjusted_config
        )
        evaluation_results['integrated_analysis'] = integrated_results
        
        # Stage 4: 最終出力・保存
        self._save_comprehensive_results(evaluation_results, mode)
        
        # 最終サマリ出力
        self._print_comprehensive_summary(evaluation_results)
        
        return evaluation_results
    
    def _adjust_config_for_mode(self, mode: str) -> Dict[str, Any]:
        """モード別設定調整"""
        config = self.config.copy()
        
        if mode == 'quick':
            # クイックモード：高速化のための設定
            config['evaluation']['save_detailed_results'] = False
            config['evaluation']['create_visualizations'] = False
            config['evaluation']['data']['max_evaluation_length'] = 100
            config['uncertainty_analysis']['enabled'] = False
            config['visualization']['enabled'] = False
            
        elif mode == 'comprehensive':
            # 包括モード：最詳細設定
            config['evaluation']['save_detailed_results'] = True
            config['evaluation']['create_visualizations'] = True
            config['uncertainty_analysis']['enabled'] = True
            config['uncertainty_analysis']['temporal_analysis'] = {
                'trend_analysis': True,
                'volatility_analysis': True,
                'autocorr_analysis': True
            }
            config['visualization']['enabled'] = True
            config['output']['compression']['enabled'] = True
            
        # else: 'standard' - デフォルト設定をそのまま使用
        
        return config
    
    def _run_filtering_performance_evaluation(
        self, 
        data_path: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 1: フィルタリング性能評価の実行"""
        print(f"\n📊 Stage 1: フィルタリング性能評価")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # フィルタリング性能評価器を作成
            performance_evaluator = FilteringPerformanceEvaluator(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                output_dir=str(self.output_dir / 'filtering_performance'),
                device=self.device
            )
            
            # 評価実行
            experiment_name = config['evaluation'].get(
                'experiment_name', 'filtering_performance'
            )
            
            results = performance_evaluator.evaluate_comprehensive(
                data_path=data_path,
                experiment_name=experiment_name,
                data_split=config['evaluation']['data']['test_split'],
                save_detailed_results=config['evaluation']['save_detailed_results'],
                create_visualizations=config['evaluation']['create_visualizations']
            )
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.experiment_log['stages'].append({
                'stage': 1,
                'name': 'filtering_performance',
                'start_time': stage_start.isoformat(),
                'duration': stage_duration,
                'status': 'completed'
            })
            
            print(f"✅ Stage 1完了 ({stage_duration:.2f}秒)")
            return results
            
        except Exception as e:
            print(f"❌ Stage 1エラー: {e}")
            self.experiment_log['stages'].append({
                'stage': 1,
                'name': 'filtering_performance',
                'start_time': stage_start.isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            return {'error': str(e), 'status': 'failed'}
    
    def _run_method_comparison_evaluation(
        self, 
        data_path: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: 推定手法比較の実行"""
        print(f"\n🔍 Stage 2: 推定手法比較")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # 推定手法比較器を作成
            method_comparator = EstimationMethodComparator(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                output_dir=str(self.output_dir / 'method_comparison'),
                device=self.device
            )
            
            # 比較実行
            experiment_name = config['evaluation'].get(
                'experiment_name', 'method_comparison'
            ) + '_comparison'
            
            results = method_comparator.compare_methods(
                data_path=data_path,
                experiment_name=experiment_name,
                data_split=config['evaluation']['data']['test_split'],
                save_results=config['evaluation']['save_detailed_results']
            )
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.experiment_log['stages'].append({
                'stage': 2,
                'name': 'method_comparison',
                'start_time': stage_start.isoformat(),
                'duration': stage_duration,
                'status': 'completed'
            })
            
            print(f"✅ Stage 2完了 ({stage_duration:.2f}秒)")
            return results
            
        except Exception as e:
            print(f"❌ Stage 2エラー: {e}")
            self.experiment_log['stages'].append({
                'stage': 2,
                'name': 'method_comparison',
                'start_time': stage_start.isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            return {'error': str(e), 'status': 'failed'}
    
    def _integrate_and_summarize_results(
        self, 
        evaluation_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: 結果統合・分析"""
        print(f"\n📈 Stage 3: 結果統合・分析")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            integrated = {
                'summary_statistics': self._compute_summary_statistics(evaluation_results),
                'key_findings': self._extract_key_findings(evaluation_results),
                'performance_comparison': self._create_performance_comparison(evaluation_results),
                'recommendations': self._generate_recommendations(evaluation_results)
            }
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.experiment_log['stages'].append({
                'stage': 3,
                'name': 'integration',
                'start_time': stage_start.isoformat(),
                'duration': stage_duration,
                'status': 'completed'
            })
            
            print(f"✅ Stage 3完了 ({stage_duration:.2f}秒)")
            return integrated
            
        except Exception as e:
            print(f"❌ Stage 3エラー: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _compute_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """サマリ統計の計算"""
        summary = {}
        
        # フィルタリング性能統計
        if 'filtering_performance' in results:
            filtering = results['filtering_performance']
            if 'summary_metrics' in filtering:
                summary['filtering'] = filtering['summary_metrics']
        
        # 手法比較統計
        if 'method_comparison' in results:
            comparison = results['method_comparison']
            if 'summary' in comparison:
                summary['comparison'] = comparison['summary']
        
        return summary
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """主要発見事項の抽出"""
        findings = []
        
        # フィルタリング性能から
        if 'filtering_performance' in results:
            filtering = results['filtering_performance']
            if 'summary_metrics' in filtering:
                metrics = filtering['summary_metrics']
                
                # バッチフィルタリング結果
                if 'batch_filtering' in metrics:
                    batch = metrics['batch_filtering']
                    findings.append(f"バッチフィルタリングMSE: {batch.get('mse', 'N/A'):.6f}")
                    if 'coverage_95' in batch:
                        findings.append(f"95%信頼区間カバレッジ: {batch['coverage_95']:.4f}")
                
                # オンラインフィルタリング結果
                if 'online_filtering' in metrics:
                    online = metrics['online_filtering']
                    findings.append(f"オンライン平均ステップ時間: {online.get('avg_step_time', 'N/A'):.6f}秒")
        
        # 手法比較から
        if 'method_comparison' in results:
            comparison = results['method_comparison']
            if 'summary' in comparison and 'comparison_summary' in comparison['summary']:
                comp_summary = comparison['summary']['comparison_summary']
                
                if 'accuracy' in comp_summary:
                    for metric, result in comp_summary['accuracy'].items():
                        findings.append(f"{metric.upper()}最良手法: {result['best_method']}")
                        
                        if 'improvement' in result and 'kalman_vs_deterministic' in result['improvement']:
                            improvement = result['improvement']['kalman_vs_deterministic']
                            findings.append(f"Kalman {metric.upper()}改善率: {improvement:+.2f}%")
        
        return findings
    
    def _create_performance_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """性能比較表の作成"""
        comparison = {}
        
        # 各手法の主要指標を整理
        methods = {}
        
        # フィルタリング結果から抽出
        if 'filtering_performance' in results:
            filtering = results['filtering_performance']
            if 'summary_metrics' in filtering:
                metrics = filtering['summary_metrics']
                
                if 'batch_filtering' in metrics:
                    methods['batch_kalman'] = metrics['batch_filtering']
                    
                if 'online_filtering' in metrics:
                    methods['online_kalman'] = metrics['online_filtering']
        
        # 手法比較結果から抽出
        if 'method_comparison' in results:
            comparison_data = results['method_comparison']
            if 'summary' in comparison_data:
                summary = comparison_data['summary']
                for method_name, method_data in summary.items():
                    if method_name != 'comparison_summary':
                        methods[method_name] = method_data
        
        comparison['methods'] = methods
        
        # 最良手法の特定
        if methods:
            comparison['best_methods'] = self._identify_best_methods(methods)
        
        return comparison
    
    def _identify_best_methods(self, methods: Dict[str, Any]) -> Dict[str, str]:
        """最良手法の特定"""
        best = {}
        
        # 各指標で最良手法を特定
        metrics_to_compare = ['mse', 'mae', 'rmse']  # 小さいほど良い
        
        for metric in metrics_to_compare:
            metric_values = {}
            for method_name, method_data in methods.items():
                if metric in method_data:
                    metric_values[method_name] = method_data[metric]
            
            if metric_values:
                best_method = min(metric_values, key=metric_values.get)
                best[metric] = best_method
        
        # 相関は大きいほど良い
        if any('correlation' in method_data for method_data in methods.values()):
            corr_values = {}
            for method_name, method_data in methods.items():
                if 'correlation' in method_data:
                    corr_values[method_name] = method_data['correlation']
            
            if corr_values:
                best_corr = max(corr_values, key=corr_values.get)
                best['correlation'] = best_corr
        
        return best
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        # 基本推奨事項
        recommendations.append("不確実性定量化が必要な場合はKalman手法を使用")
        recommendations.append("高速推論が優先される場合は決定的手法を検討")
        
        # 結果に基づく推奨事項
        if 'method_comparison' in results:
            comparison = results['method_comparison']
            if 'summary' in comparison and 'comparison_summary' in comparison['summary']:
                comp = comparison['summary']['comparison_summary']
                
                # 精度改善が見られる場合
                if 'accuracy' in comp:
                    for metric, result in comp['accuracy'].items():
                        if 'improvement' in result and 'kalman_vs_deterministic' in result['improvement']:
                            improvement = result['improvement']['kalman_vs_deterministic']
                            if improvement > 5:  # 5%以上改善
                                recommendations.append(f"Kalmanは{metric.upper()}で{improvement:.1f}%改善を実現")
        
        return recommendations
    
    def _save_comprehensive_results(self, results: Dict[str, Any], mode: str):
        """統合結果の保存"""
        print(f"\n💾 結果保存中...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 統合結果をJSON形式で保存
        comprehensive_results = {
            'experiment_info': {
                'mode': mode,
                'timestamp': timestamp,
                'model_path': str(self.model_path),
                'device': self.device,
                'total_duration': (datetime.now() - datetime.fromisoformat(self.experiment_log['start_time'])).total_seconds()
            },
            'experiment_log': self.experiment_log,
            'evaluation_results': results
        }
        
        # JSON保存
        json_path = self.output_dir / 'summary' / f'task4_comprehensive_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self._make_json_serializable(comprehensive_results), f, indent=2)
        
        # CSV サマリ保存
        csv_path = self.output_dir / 'summary' / f'task4_summary_{timestamp}.csv'
        self._save_results_csv(results, csv_path)
        
        print(f"📁 統合結果保存: {json_path}")
        print(f"📊 サマリCSV: {csv_path}")
    
    def _save_results_csv(self, results: Dict[str, Any], csv_path: Path):
        """結果をCSV形式で保存"""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'category', 'method', 'mse', 'mae', 'rmse', 'correlation',
                'processing_time', 'coverage_95', 'has_uncertainty'
            ])
            
            # フィルタリング性能結果
            if 'filtering_performance' in results and 'summary_metrics' in results['filtering_performance']:
                metrics = results['filtering_performance']['summary_metrics']
                
                for method_name, method_data in metrics.items():
                    if isinstance(method_data, dict):
                        writer.writerow([
                            'filtering_performance', method_name,
                            method_data.get('mse', ''),
                            method_data.get('mae', ''),
                            method_data.get('rmse', ''),
                            method_data.get('correlation', ''),
                            method_data.get('processing_time', ''),
                            method_data.get('coverage_95', ''),
                            'yes' if 'uncertainty' in method_name or 'kalman' in method_name else 'no'
                        ])
            
            # 手法比較結果
            if 'method_comparison' in results and 'summary' in results['method_comparison']:
                summary = results['method_comparison']['summary']
                
                for method_name, method_data in summary.items():
                    if method_name != 'comparison_summary' and isinstance(method_data, dict):
                        writer.writerow([
                            'method_comparison', method_name,
                            method_data.get('mse', ''),
                            method_data.get('mae', ''),
                            method_data.get('rmse', ''),
                            method_data.get('correlation', ''),
                            method_data.get('processing_time', ''),
                            '',  # coverage not available in method comparison
                            'yes' if method_data.get('has_uncertainty', False) else 'no'
                        ])
    
    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """包括的結果サマリの出力"""
        print(f"\n" + "="*70)
        print(f"🎉 タスク4統合評価完了")
        print(f"📊 実験名: {self.config['evaluation'].get('experiment_name', 'Task4 Evaluation')}")
        print("="*70)
        
        # 主要発見事項
        if 'integrated_analysis' in results and 'key_findings' in results['integrated_analysis']:
            findings = results['integrated_analysis']['key_findings']
            print(f"\n🔍 主要発見事項:")
            for i, finding in enumerate(findings, 1):
                print(f"   {i}. {finding}")
        
        # 推奨事項
        if 'integrated_analysis' in results and 'recommendations' in results['integrated_analysis']:
            recommendations = results['integrated_analysis']['recommendations']
            print(f"\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # 出力ファイル
        print(f"\n📁 出力ディレクトリ: {self.output_dir}")
        print(f"   ├── filtering_performance/  # フィルタリング性能詳細")
        print(f"   ├── method_comparison/      # 手法比較詳細")
        print(f"   ├── summary/                # 統合サマリ")
        print(f"   └── visualizations/         # 可視化結果")
        
        print("="*70)
    
    def _make_json_serializable(self, obj):
        """JSON対応形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="タスク4統合評価パイプライン")
    
    # 必須引数
    parser.add_argument('--model', required=True, help='学習済みモデルパス (.pth)')
    parser.add_argument('--data', required=True, help='評価データパス (.npz)')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')
    
    # オプション引数
    parser.add_argument(
        '--config', 
        default='configs/evaluation_config.yaml',
        help='評価設定ファイルパス'
    )
    parser.add_argument(
        '--mode',
        default='standard',
        choices=['quick', 'standard', 'comprehensive'],
        help='評価モード'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='計算デバイス (auto, cpu, cuda)'
    )
    
    args = parser.parse_args()
    
    # 引数検証
    required_files = [
        ('model', args.model),
        ('data', args.data),
        ('config', args.config)
    ]
    
    for name, filepath in required_files:
        if not Path(filepath).exists():
            print(f"❌ {name}ファイルが見つかりません: {filepath}")
            return 1
    
    print(f"🚀 タスク4統合評価パイプライン")
    print(f"📊 モード: {args.mode}")
    print(f"🖥️  デバイス: {args.device}")
    
    try:
        # パイプライン作成・実行
        pipeline = Task4EvaluationPipeline(
            model_path=args.model,
            config_path=args.config,
            output_dir=args.output,
            device=args.device
        )
        
        results = pipeline.run_comprehensive_evaluation(
            data_path=args.data,
            mode=args.mode
        )
        
        print(f"\n✅ タスク4統合評価完了！")
        print(f"📂 結果確認: {args.output}")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  ユーザーによる実行中断")
        return 130
    except Exception as e:
        print(f"\n❌ 評価中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())