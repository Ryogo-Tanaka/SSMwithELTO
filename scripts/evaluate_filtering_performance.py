#!/usr/bin/env python3
"""
フィルタリング性能評価スクリプト

DFIV Kalman Filterの状態推定性能を包括的に評価し、
結果をターミナル出力・数値保存する。

Usage:
    python scripts/evaluate_filtering_performance.py \
        --model_path results/trained_model.pth \
        --data_path data/test_data.npz \
        --output_dir results/filtering_evaluation \
        --config configs/inference_config.yaml
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import json

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent.parent))

from src.models.inference_model import InferenceModel
from src.evaluation.filtering_analysis import FilteringAnalyzer
from src.evaluation.uncertainly_evaluation import UncertaintyEvaluator
from src.utils.data_loader import load_experimental_data


class FilteringPerformanceEvaluator:
    """フィルタリング性能評価の統合クラス"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        output_dir: str,
        device: str = 'auto',
        config: dict = None
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # デバイス設定
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🖥️  使用デバイス: {self.device}")
        
        # 設定読み込み
        if config is not None:
            # 外部から設定が渡された場合（推奨）
            self.config = config
            print(f"📝 外部設定を使用")
        else:
            # ファイルから読み込み（フォールバック） - 明確な選択基準なし
            raise ValueError(
                "設定が指定されていません。run_filtering_evaluation.pyから適切な設定を渡してください。"
                "複数ドキュメントYAMLファイルからの自動選択は未実装です。"
            )

        # checkpoint['config']から真実の情報を取得してマージ（SSOT確立）
        self._merge_checkpoint_config()

        # 分析器の初期化
        self.filtering_analyzer = FilteringAnalyzer(str(self.output_dir), self.device)
        self.uncertainty_evaluator = UncertaintyEvaluator(str(self.output_dir))

        # モデル読み込み
        self.inference_model = None
        self._load_inference_model()

    def _merge_checkpoint_config(self):
        """
        Config優先、checkpoint fallbackでマージ

        優先順位:
          1. コマンドライン引数で指定されたYAML config (self.config)
          2. checkpoint['config']の保存情報

        目的:
        - 訓練時のYAML configを優先的に使用（完全な情報源）
        - checkpointには訓練時に保存された最小限の情報のみ含まれる
        - configにない情報のみcheckpointから補完
        """
        import torch

        print(f"📂 ConfigとCheckpointをマージ中（Config優先）...")

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            checkpoint_config = checkpoint.get('config', {})

            if not checkpoint_config:
                raise ValueError(
                    f"checkpoint['config']が存在しません。\n"
                    f"モデルファイルを確認してください: {self.model_path}"
                )

            # モデル構造情報のマージ
            if 'model' not in self.config:
                self.config['model'] = {}

            # encoder情報のマージ (config優先、checkpoint fallback)
            if 'encoder' in self.config.get('model', {}):
                # YAMLにencoder情報がある場合、それを基本とする
                base_encoder = self.config['model']['encoder'].copy()
                checkpoint_encoder = checkpoint_config.get('model', {}).get('encoder', {})

                # YAMLにない情報のみcheckpointから補完
                for key, value in checkpoint_encoder.items():
                    if key not in base_encoder:
                        base_encoder[key] = value
                        print(f"  ↳ encoder.{key}: checkpointから補完")

                self.config['model']['encoder'] = base_encoder
                encoder_type = self.config['model']['encoder'].get('type')
                print(f"✅ Encoder type: '{encoder_type}' (from config)")
            elif 'encoder' in checkpoint_config.get('model', {}):
                # YAMLにencoder情報がない場合のみcheckpointを使用
                self.config['model']['encoder'] = checkpoint_config['model']['encoder'].copy()
                encoder_type = self.config['model']['encoder'].get('type')
                print(f"⚠️  Encoder type: '{encoder_type}' (from checkpoint only - config missing)")
            else:
                raise ValueError(
                    "Encoder情報がYAML configにもcheckpointにも見つかりません。\n"
                    "設定ファイルを確認してください。"
                )

            # decoder情報のマージ (同様の方式)
            if 'decoder' in self.config.get('model', {}):
                base_decoder = self.config['model']['decoder'].copy()
                checkpoint_decoder = checkpoint_config.get('model', {}).get('decoder', {})

                for key, value in checkpoint_decoder.items():
                    if key not in base_decoder:
                        base_decoder[key] = value

                self.config['model']['decoder'] = base_decoder
                print(f"✅ Decoder config: from config")
            elif 'decoder' in checkpoint_config.get('model', {}):
                self.config['model']['decoder'] = checkpoint_config['model']['decoder'].copy()
                print(f"⚠️  Decoder config: from checkpoint only")

            # SSM設定のマージ (config優先)
            if 'ssm' not in self.config:
                self.config['ssm'] = {}

            if 'ssm' in checkpoint_config:
                for key in ['realization', 'df_state', 'df_observation']:
                    if key in checkpoint_config['ssm']:
                        if key not in self.config['ssm']:
                            # configにない場合のみcheckpointから取得
                            self.config['ssm'][key] = checkpoint_config['ssm'][key].copy()
                            print(f"  ↳ ssm.{key}: checkpointから補完")
                        else:
                            # configにある場合、checkpointの情報で不足分を補完
                            for subkey, subvalue in checkpoint_config['ssm'][key].items():
                                if subkey not in self.config['ssm'][key]:
                                    self.config['ssm'][key][subkey] = subvalue

            print(f"✅ Config優先マージ完了")

        except Exception as e:
            raise RuntimeError(f"ConfigとCheckpointのマージに失敗: {e}")

    def _load_inference_model(self):
        """推論モデルの読み込み"""
        try:
            print(f"📂 モデル読み込み: {self.model_path}")
            self.inference_model = InferenceModel(
                str(self.model_path), str(self.config_path)
            )
            print("✅ モデル読み込み完了")
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            raise
    
    def evaluate_comprehensive(
        self,
        data_path: str,
        experiment_name: str = None,
        data_split: str = 'test',
        save_detailed_results: bool = True,
        create_visualizations: bool = True
    ) -> dict:
        """
        包括的フィルタリング性能評価
        
        Args:
            data_path: データファイルパス
            experiment_name: 実験名
            data_split: 使用するデータ分割 ('test', 'val', 'all')
            save_detailed_results: 詳細結果保存するか
            create_visualizations: 可視化作成するか
            
        Returns:
            Dict: 評価結果
        """
        if experiment_name is None:
            experiment_name = f"filtering_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        print(f"\n🚀 包括的フィルタリング評価開始")
        print(f"📊 実験名: {experiment_name}")
        print(f"📁 データ: {data_path}")
        print("="*70)
        
        # 1. データ読み込み
        evaluation_data = self._load_evaluation_data(data_path, data_split)
        test_data = evaluation_data['observations']
        true_states = evaluation_data.get('true_states', None)
        
        print(f"📏 評価データ形状: {test_data.shape}")
        if true_states is not None:
            print(f"📏 真値状態形状: {true_states.shape}")
        else:
            print("⚠️  真値状態なし（推定精度評価は制限される）")
        
        # 2. 推論環境セットアップ
        self._setup_inference_environment(evaluation_data)
        
        # 3. フィルタリング分析実行
        filtering_results = self.filtering_analyzer.analyze_filtering_performance(
            self.inference_model,
            test_data,
            true_states,
            experiment_name=experiment_name,
            save_results=save_detailed_results,
            verbose=True
        )
        
        # 4. 不確実性詳細分析
        uncertainty_results = self._analyze_uncertainty_details(
            filtering_results, true_states, create_visualizations
        )
        
        # 5. 結果統合・保存
        complete_results = self._compile_final_results(
            filtering_results, uncertainty_results, experiment_name, evaluation_data
        )
        
        # 6. 最終サマリ出力
        self._print_final_summary(complete_results)
        
        # 7. 結果エクスポート
        if save_detailed_results:
            self._export_results(complete_results, experiment_name)
        
        return complete_results
    
    def _load_evaluation_data(self, data_path: str, data_split: str) -> dict:
        """評価データの読み込み"""
        print(f"\n📂 データ読み込み中...")

        try:
            # エンコーダタイプを取得（checkpoint由来、_merge_checkpoint_config()で設定済み）
            encoder_type = self.config.get('model', {}).get('encoder', {}).get('type')

            if encoder_type is None:
                raise ValueError(
                    "Encoder typeが取得できません。\n"
                    "checkpoint['config']['model']['encoder']['type']を確認してください。"
                )

            print(f"📝 エンコーダタイプ: {encoder_type}")

            # エンコーダタイプに応じたデータローダー選択
            if encoder_type == 'rkn':
                # 画像データ用（4D: T, H, W, C）
                from src.utils.data_loader import load_experimental_data_with_architecture
                import numpy as np

                dataset = load_experimental_data_with_architecture(
                    data_path=data_path,
                    config=self.config,
                    split=data_split if data_split != 'all' else 'test',
                    return_dataloaders=False
                )
                # Datasetから全データを取得
                if hasattr(dataset, 'data'):
                    observations = dataset.data

                    # numpy配列の場合はtorchに変換（DataLoader経由でない直接アクセス用）
                    if isinstance(observations, np.ndarray):
                        observations = torch.from_numpy(observations).float()
                        print(f"📊 numpy→torch変換完了: {observations.shape}, dtype={observations.dtype}")
                    elif isinstance(observations, torch.Tensor):
                        # 既にtorchの場合はそのまま
                        print(f"📊 torch.Tensor検出: {observations.shape}, dtype={observations.dtype}")
                    else:
                        raise TypeError(
                            f"未対応のデータ型: {type(observations)}\n"
                            f"期待される型: numpy.ndarray または torch.Tensor"
                        )
                else:
                    raise RuntimeError("Dataset does not have 'data' attribute")
                data_dict = {data_split: observations}

            elif encoder_type == 'time_invariant':
                # 1D系列データ用（2D: T, n）
                from src.utils.data_loader import load_experimental_data
                data_dict = load_experimental_data(data_path)

            # ========================================
            # 【拡張ポイント】新しいencoder typeを追加する場合
            # ========================================
            # elif encoder_type == 'transformer':
            #     # Transformerエンコーダ用（データ形状に応じて選択）
            #     # 例: 1D系列の場合
            #     from src.utils.data_loader import load_experimental_data
            #     data_dict = load_experimental_data(data_path)
            #
            # elif encoder_type == 'your_new_encoder':
            #     # 新しいエンコーダ用のデータローダー
            #     # データ形状に応じて適切なローダーを選択
            #     pass
            # ========================================

            else:
                raise ValueError(
                    f"未対応のencoder type: '{encoder_type}'\n"
                    f"現在対応しているのは 'rkn' と 'time_invariant' のみです。\n"
                    f"新しいencoder typeを追加する場合は、上記の【拡張ポイント】を参照してください。"
                )
            
            # 指定された分割を取得
            if data_split == 'test' and 'test' in data_dict:
                observations = data_dict['test']
            elif data_split == 'val' and 'val' in data_dict:
                observations = data_dict['val']
            elif data_split == 'all':
                # 全データを結合
                obs_list = []
                for key in ['train', 'val', 'test']:
                    if key in data_dict:
                        obs_list.append(data_dict[key])
                if obs_list:
                    observations = torch.cat(obs_list, dim=0)
                else:
                    observations = data_dict[list(data_dict.keys())[0]]
            else:
                # デフォルト：利用可能な最初のデータ
                observations = data_dict[list(data_dict.keys())[0]]
                print(f"⚠️  指定分割 '{data_split}' が見つからない。'{list(data_dict.keys())[0]}' を使用。")
            
            # デバイスに移動
            observations = observations.to(self.device)

            # データ形状の検証
            print(f"📏 データ形状: {observations.shape}")
            if encoder_type == 'rkn':
                if observations.ndim != 4:
                    raise ValueError(
                        f"rknEncoderには4D入力が必要: (T, H, W, C), "
                        f"got {observations.shape} (ndim={observations.ndim})"
                    )

            # 真値状態（もしあれば）
            true_states = None
            if 'true_states' in data_dict:
                true_states = data_dict['true_states'].to(self.device)
            elif f'{data_split}_states' in data_dict:
                true_states = data_dict[f'{data_split}_states'].to(self.device)

            evaluation_data = {
                'observations': observations,
                'true_states': true_states,
                'encoder_type': encoder_type,
                'metadata': {
                    'data_path': data_path,
                    'data_split': data_split,
                    'available_keys': list(data_dict.keys())
                }
            }

            print(f"✅ データ読み込み完了")
            return evaluation_data
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise
    
    def _setup_inference_environment(self, evaluation_data: dict):
        """推論環境のセットアップ"""
        print(f"\n⚙️  推論環境セットアップ中...")
        
        try:
            # キャリブレーションデータの準備（観測の一部を使用）
            observations = evaluation_data['observations']

            # past_horizonを考慮した最小キャリブレーションサイズを計算
            past_horizon = self.config.get('ssm', {}).get('realization', {}).get('past_horizon', 10)
            min_required = 2 * past_horizon + 1  # realization.filterに必要な最小サンプル数

            # 利用可能なデータに基づいてキャリブレーションサイズを決定
            total_samples = observations.size(0)

            # DEBUG: テストデータ不足の詳細ログ (Resolved in Step 7)
            # print(f"🔍 DEBUG - データサイズ分析:")
            # print(f"   観測データ総数: {total_samples}")
            # print(f"   past_horizon: {past_horizon}")
            # print(f"   必要最小サンプル: {min_required} (2*{past_horizon}+1)")
            # print(f"   データ分割: {total_samples} // 4 = {total_samples // 4}")

            if total_samples >= min_required:
                calibration_size = min(50, max(min_required, total_samples // 4))
                print(f"✅ 十分なデータ: キャリブレーション{calibration_size}サンプル使用")
            else:
                # データが不足している場合は全データを使用し、past_horizonを調整
                calibration_size = total_samples
                print(f"❌ 【根本原因】テストデータ不足: {total_samples}サンプル < 必要{min_required}")
                print(f"   → より大きなテストデータセット(推奨: >50サンプル)が必要")
                print(f"   → 一時対処: past_horizon={past_horizon}を調整することを推奨")

            calibration_data = observations[:calibration_size]
            
            # 推論セットアップ
            self.inference_model.setup_inference(
                calibration_data=calibration_data,
                method='data_driven'
            )
            
            print(f"✅ 推論環境セットアップ完了")
            print(f"📊 キャリブレーションデータ: {calibration_data.shape}")
            
        except Exception as e:
            print(f"❌ 推論環境セットアップエラー: {e}")
            raise
    
    def _analyze_uncertainty_details(
        self, 
        filtering_results: dict, 
        true_states: torch.Tensor, 
        create_visualizations: bool
    ) -> dict:
        """不確実性詳細分析"""
        if not filtering_results.get('batch_filtering', {}).get('success', False):
            print("⚠️  バッチフィルタリング失敗のため、不確実性分析をスキップ")
            return {}
        
        print(f"\n🎲 不確実性詳細分析中...")
        
        try:
            batch_result = filtering_results['batch_filtering']
            predictions = batch_result['estimated_states']
            covariances = batch_result['covariances']
            
            # 標準偏差抽出
            uncertainties = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))
            
            # 不確実性評価実行
            uncertainty_analysis = self.uncertainty_evaluator.evaluate_uncertainty_quality(
                predictions,
                uncertainties,
                true_states,
                save_plots=create_visualizations,
                verbose=True
            )
            
            print(f"✅ 不確実性詳細分析完了")
            return uncertainty_analysis
            
        except Exception as e:
            print(f"❌ 不確実性分析エラー: {e}")
            return {'error': str(e)}
    
    def _compile_final_results(
        self, 
        filtering_results: dict, 
        uncertainty_results: dict, 
        experiment_name: str,
        evaluation_data: dict
    ) -> dict:
        """最終結果の統合"""
        return {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'config_path': str(self.config_path),
                'device': self.device,
                'data_info': evaluation_data['metadata']
            },
            'filtering_analysis': filtering_results,
            'uncertainty_analysis': uncertainty_results,
            'summary_metrics': self._extract_summary_metrics(filtering_results, uncertainty_results)
        }
    
    def _extract_summary_metrics(self, filtering_results: dict, uncertainty_results: dict) -> dict:
        """サマリメトリクスの抽出"""
        summary = {}
        
        # バッチフィルタリング主要指標
        if filtering_results.get('batch_filtering', {}).get('success', False):
            batch_metrics = filtering_results['batch_filtering']['metrics']
            
            if 'accuracy' in batch_metrics:
                acc = batch_metrics['accuracy']
                summary['batch_filtering'] = {
                    'mse': acc['mse'],
                    'mae': acc['mae'],
                    'rmse': acc['rmse'],
                    'correlation': acc['correlation']
                }
            
            if 'uncertainty' in batch_metrics:
                unc = batch_metrics['uncertainty']
                summary['batch_filtering'].update({
                    'mean_uncertainty': unc['mean_uncertainty'],
                    'coverage_95': unc.get('coverage_95', None)
                })
                
            summary['batch_filtering']['processing_time'] = filtering_results['batch_filtering']['processing_time']
        
        # オンラインフィルタリング主要指標
        if filtering_results.get('online_filtering', {}).get('success', False):
            online_metrics = filtering_results['online_filtering']['metrics']
            
            if 'accuracy' in online_metrics:
                acc = online_metrics['accuracy']
                summary['online_filtering'] = {
                    'mse': acc['mse'],
                    'mae': acc['mae'],
                    'rmse': acc['rmse'],
                    'correlation': acc['correlation']
                }
            
            summary['online_filtering']['total_time'] = filtering_results['online_filtering']['total_processing_time']
            summary['online_filtering']['avg_step_time'] = filtering_results['online_filtering']['average_step_time']
        
        # 不確実性主要指標
        if 'confidence_intervals' in uncertainty_results:
            ci_results = uncertainty_results['confidence_intervals']
            summary['uncertainty'] = {}
            for level in [68, 95]:
                key = f'confidence_{level}'
                if key in ci_results:
                    summary['uncertainty'][f'coverage_{level}'] = ci_results[key]['actual_coverage']
                    summary['uncertainty'][f'coverage_error_{level}'] = ci_results[key]['coverage_error']
        
        if 'calibration' in uncertainty_results:
            summary['uncertainty']['ece'] = uncertainty_results['calibration']['ece']
        
        return summary
    
    def _print_final_summary(self, results: dict):
        """最終結果サマリの出力"""
        print(f"\n" + "="*70)
        print(f"📊 フィルタリング性能評価 最終結果")
        print(f"🏷️  実験名: {results['experiment_info']['name']}")
        print(f"⏰ 実行時刻: {results['experiment_info']['timestamp']}")
        print("="*70)
        
        summary = results['summary_metrics']
        
        # バッチフィルタリング結果
        if 'batch_filtering' in summary:
            batch = summary['batch_filtering']
            print(f"\n📊 バッチフィルタリング性能:")
            print(f"   MSE:          {batch.get('mse', 'N/A'):.6f}")
            print(f"   MAE:          {batch.get('mae', 'N/A'):.6f}")
            print(f"   RMSE:         {batch.get('rmse', 'N/A'):.6f}")
            print(f"   相関係数:     {batch.get('correlation', 'N/A'):.4f}")
            print(f"   平均不確実性: {batch.get('mean_uncertainty', 'N/A'):.6f}")
            print(f"   95%カバレッジ:{batch.get('coverage_95', 'N/A'):.4f}")
            print(f"   処理時間:     {batch.get('processing_time', 'N/A'):.4f}秒")
        
        # オンラインフィルタリング結果
        if 'online_filtering' in summary:
            online = summary['online_filtering']
            print(f"\n📱 オンラインフィルタリング性能:")
            print(f"   MSE:          {online.get('mse', 'N/A'):.6f}")
            print(f"   MAE:          {online.get('mae', 'N/A'):.6f}")
            print(f"   RMSE:         {online.get('rmse', 'N/A'):.6f}")
            print(f"   相関係数:     {online.get('correlation', 'N/A'):.4f}")
            print(f"   総処理時間:   {online.get('total_time', 'N/A'):.4f}秒")
            print(f"   平均ステップ時間: {online.get('avg_step_time', 'N/A'):.6f}秒")
        
        # 不確実性結果
        if 'uncertainty' in summary:
            unc = summary['uncertainty']
            print(f"\n🎲 不確実性定量化品質:")
            print(f"   68%カバレッジ: {unc.get('coverage_68', 'N/A'):.4f}")
            print(f"   95%カバレッジ: {unc.get('coverage_95', 'N/A'):.4f}")
            print(f"   ECE:          {unc.get('ece', 'N/A'):.4f}")
        
        # 手法比較
        if 'batch_filtering' in summary and 'online_filtering' in summary:
            batch = summary['batch_filtering']
            online = summary['online_filtering']
            
            print(f"\n🔍 バッチ vs オンライン比較:")
            if 'mse' in batch and 'mse' in online:
                mse_diff = online['mse'] - batch['mse']
                print(f"   MSE差分:      {mse_diff:+.6f} (オンライン - バッチ)")
            
            if 'processing_time' in batch and 'total_time' in online:
                speed_ratio = online['total_time'] / batch['processing_time']
                print(f"   速度比:       {speed_ratio:.2f}x (オンライン/バッチ)")
        
        print("="*70)
        print(f"✅ 評価完了")
    
    def _export_results(self, results: dict, experiment_name: str):
        """結果のエクスポート"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で詳細結果保存
        json_path = self.output_dir / f"{experiment_name}_{timestamp}_complete_results.json"
        with open(json_path, 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)
        
        # サマリをCSV形式で保存
        self._export_summary_csv(results, experiment_name, timestamp)
        
        print(f"\n📁 結果エクスポート完了:")
        print(f"   詳細結果: {json_path}")
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def _export_summary_csv(self, results: dict, experiment_name: str, timestamp: str):
        """サマリをCSV形式でエクスポート"""
        import csv
        
        csv_path = self.output_dir / f"{experiment_name}_{timestamp}_summary.csv"
        summary = results['summary_metrics']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'experiment_name', 'method', 'mse', 'mae', 'rmse', 'correlation',
                'mean_uncertainty', 'coverage_95', 'processing_time', 'avg_step_time'
            ])
            
            # バッチフィルタリング
            if 'batch_filtering' in summary:
                batch = summary['batch_filtering']
                writer.writerow([
                    experiment_name, 'batch',
                    batch.get('mse', ''), batch.get('mae', ''), batch.get('rmse', ''),
                    batch.get('correlation', ''), batch.get('mean_uncertainty', ''),
                    batch.get('coverage_95', ''), batch.get('processing_time', ''), ''
                ])
            
            # オンラインフィルタリング
            if 'online_filtering' in summary:
                online = summary['online_filtering']
                writer.writerow([
                    experiment_name, 'online',
                    online.get('mse', ''), online.get('mae', ''), online.get('rmse', ''),
                    online.get('correlation', ''), '', '', 
                    online.get('total_time', ''), online.get('avg_step_time', '')
                ])
        
        print(f"   サマリCSV: {csv_path}")
    
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
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="DFIV Kalman Filter フィルタリング性能評価")
    
    parser.add_argument(
        '--model_path', 
        required=True,
        help='学習済みモデルファイルパス (.pth)'
    )
    parser.add_argument(
        '--data_path',
        required=True, 
        help='評価データファイルパス (.npz)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='結果出力ディレクトリ'
    )
    parser.add_argument(
        '--config',
        default='configs/inference_config.yaml',
        help='推論設定ファイルパス'
    )
    parser.add_argument(
        '--experiment_name',
        default=None,
        help='実験名（デフォルト：自動生成）'
    )
    parser.add_argument(
        '--data_split',
        default='test',
        choices=['test', 'val', 'all'],
        help='使用するデータ分割'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='計算デバイス (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='可視化をスキップ'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='クイック評価（詳細保存なし）'
    )
    
    args = parser.parse_args()
    
    # 引数検証
    if not Path(args.model_path).exists():
        print(f"❌ モデルファイルが見つかりません: {args.model_path}")
        return
    
    if not Path(args.data_path).exists():
        print(f"❌ データファイルが見つかりません: {args.data_path}")
        return
    
    if not Path(args.config).exists():
        print(f"❌ 設定ファイルが見つかりません: {args.config}")
        return
    
    # 評価実行
    try:
        evaluator = FilteringPerformanceEvaluator(
            model_path=args.model_path,
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device
        )
        
        results = evaluator.evaluate_comprehensive(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            data_split=args.data_split,
            save_detailed_results=not args.quick,
            create_visualizations=not args.no_visualization
        )
        
        print(f"\n🎉 フィルタリング性能評価完了！")
        
    except Exception as e:
        print(f"\n❌ 評価中にエラーが発生: {e}")
        raise


if __name__ == "__main__":
    main()