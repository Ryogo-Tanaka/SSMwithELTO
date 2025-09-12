# src/config/inference_config_loader.py
"""
推論設定の読み込み・管理

設定ファイルベースで推論パラメータを管理。
環境別設定、設定検証、デフォルト値フォールバックを提供。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

class InferenceConfigLoader:
    """推論設定読み込みクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._find_config_path(config_path)
        self._config_cache = None
    
    def _find_config_path(self, config_path: Optional[str]) -> Path:
        """設定ファイルパスの探索"""
        if config_path:
            return Path(config_path)
        
        # デフォルト探索順序
        search_paths = [
            Path("configs/inference_settings.yaml"),
            Path("../configs/inference_settings.yaml"),
            Path(__file__).parent.parent.parent / "configs/inference_settings.yaml"
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"推論設定ファイルが見つかりません。探索パス: {search_paths}")
    
    def load_config(self, environment: str = "inference_defaults") -> Dict[str, Any]:
        """設定読み込み"""
        if self._config_cache is None:
            with open(self.config_path, 'r') as f:
                self._config_cache = yaml.safe_load(f)
        
        # ベース設定取得
        base_config = self._config_cache.get("inference_defaults", {}).copy()
        
        # 環境別オーバーライド適用
        if environment != "inference_defaults" and environment in self._config_cache:
            env_overrides = self._config_cache[environment]
            base_config = self._deep_merge(base_config, env_overrides)
        
        # 設定検証
        validated_config = self._validate_config(base_config)
        
        return validated_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の深いマージ"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定検証とデフォルト値補完"""
        # 必須セクションの確認
        required_sections = ["noise_estimation", "initialization", "numerical"]
        for section in required_sections:
            if section not in config:
                warnings.warn(f"設定に {section} セクションがありません。デフォルト値を使用します。")
                config[section] = self._get_default_section(section)
        
        # 数値範囲の検証
        noise_config = config.get("noise_estimation", {})
        if noise_config.get("gamma_Q", 0) <= 0:
            warnings.warn("gamma_Q は正の値である必要があります。1e-6に設定します。")
            noise_config["gamma_Q"] = 1e-6
            
        if noise_config.get("gamma_R", 0) <= 0:
            warnings.warn("gamma_R は正の値である必要があります。1e-6に設定します。")
            noise_config["gamma_R"] = 1e-6
        
        return config
    
    def _get_default_section(self, section_name: str) -> Dict[str, Any]:
        """デフォルトセクション設定"""
        defaults = {
            "noise_estimation": {
                "method": "residual_based",
                "gamma_Q": 1e-6,
                "gamma_R": 1e-6
            },
            "initialization": {
                "method": "data_driven",
                "n_init_samples": 50
            },
            "numerical": {
                "condition_threshold": 1e12,
                "min_eigenvalue": 1e-8,
                "jitter": 1e-6
            }
        }
        return defaults.get(section_name, {})

# コンビニエンス関数
def load_inference_config(
    config_path: Optional[str] = None, 
    environment: str = "inference_defaults"
) -> Dict[str, Any]:
    """推論設定の簡単読み込み"""
    loader = InferenceConfigLoader(config_path)
    return loader.load_config(environment)