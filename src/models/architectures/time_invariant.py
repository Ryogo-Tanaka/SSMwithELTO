# src/models/architectures/time_invariant.py

"""
時不変性保証付きエンコーダー・デコーダーアーキテクチャ

【設定ファイル使用例】
```yaml
model:
  encoder:
    type: "time_invariant"           # 必須: このアーキテクチャを選択
    input_dim: 7                     # 必須: 観測次元 n
    output_dim: 16                   # 必須: 特徴量次元 m
    architecture: "mlp"              # 推奨: "mlp" | "resnet"
    hidden_dims: [64, 32]            # オプション: 隠れ層次元リスト
    activation: "GELU"               # オプション: "GELU" | "ReLU" | "Tanh"
    dropout: 0.1                     # オプション: ドロップアウト率
    normalize_input: true            # オプション: 入力正規化
    normalize_output: true           # オプション: 出力正規化
    track_running_stats: true        # オプション: 統計量追跡
    momentum: 0.1                    # オプション: BatchNorm慣性
    eps: 1e-5                        # オプション: 数値安定化

  decoder:
    type: "time_invariant"           # 必須: このアーキテクチャを選択
    input_dim: 16                    # 必須: 特徴量次元 m（エンコーダーと一致）
    output_dim: 7                    # 必須: 観測次元 n（元の観測と一致）
    architecture: "mlp"              # 推奨: "mlp" | "resnet"
    hidden_dims: [32, 64]            # オプション: 隠れ層次元リスト
    activation: "GELU"               # オプション: 活性化関数
    dropout: 0.1                     # オプション: ドロップアウト率
```

【重要な設定パラメータ】
- input_dim/output_dim: 必須、次元整合性の確保
- normalize_input/output: 時不変性・弱定常性のために推奨
- track_running_stats: 推論時統計量一貫性のために推奨
- architecture: "mlp"が標準、"resnet"は複雑なデータ用
"""

import math
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class time_invariantEncoder(nn.Module):
    """
    時不変性保証付きエンコーダー u_η: R^n → R^m

    定式化要求：
    - 時不変性：全時点で同一パラメータ η を共有
    - 弱定常性：E[u_η(y_t)] = const（時間に依存しない平均）
    - 次元圧縮：観測 y_t ∈ R^n を特徴量 m_t ∈ R^m に変換

    実装特徴：
    1. 入力正規化：y_t → (y_t - μ_y) / σ_y
    2. 時不変変換：m_t = u_η(normalized_y_t)
    3. 出力正規化：m_t → (m_t - μ_m) / σ_m
    4. 統計量管理：推論時の一貫性保証
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: str = "mlp",
        hidden_dims: Optional[List[int]] = None,
        activation: str = "GELU",
        dropout: float = 0.0,
        normalize_input: bool = True,
        normalize_output: bool = True,
        track_running_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_dim: 入力次元 n
            output_dim: 出力次元 m
            architecture: 内部アーキテクチャ（"mlp", "resnet"）
            hidden_dims: 隠れ層次元リスト
            activation: 活性化関数
            dropout: ドロップアウト率
            normalize_input: 入力正規化を行うか
            normalize_output: 出力正規化を行うか
            track_running_stats: 統計量を追跡するか
            momentum: 統計量更新の慣性
            eps: 数値安定化パラメータ
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

        # デフォルト隠れ層設定（出力次元に応じた自動設定）
        if hidden_dims is None:
            if output_dim <= 4:
                hidden_dims = [64, 32]
            elif output_dim <= 16:
                hidden_dims = [128, 64]
            else:
                hidden_dims = [256, 128, 64]

        # 活性化関数
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU()

        # 入力正規化層
        if self.normalize_input:
            self.input_norm = nn.BatchNorm1d(input_dim, momentum=momentum, eps=eps)

        # コアネットワーク
        if architecture == "mlp":
            self.core_net = self._build_mlp(input_dim, output_dim, hidden_dims, dropout)
        elif architecture == "resnet":
            self.core_net = self._build_resnet(input_dim, output_dim, hidden_dims, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Supported: ['mlp', 'resnet']")

        # 出力正規化層
        if self.normalize_output:
            self.output_norm = nn.BatchNorm1d(output_dim, momentum=momentum, eps=eps)

        # 統計量（推論時の一貫性用）
        if track_running_stats:
            self.register_buffer('input_mean', torch.zeros(input_dim))
            self.register_buffer('input_var', torch.ones(input_dim))
            self.register_buffer('output_mean', torch.zeros(output_dim))
            self.register_buffer('output_var', torch.ones(output_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # 初期化
        self._initialize_weights()

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """標準MLPアーキテクチャ"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _build_resnet(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """ResNet風アーキテクチャ（残差接続付き）"""
        layers = []
        prev_dim = input_dim

        # 入力射影
        if len(hidden_dims) > 0:
            first_hidden = hidden_dims[0]
            layers.append(nn.Linear(input_dim, first_hidden))
            prev_dim = first_hidden

        # 残差ブロック
        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:  # 最初のブロックはスキップ（すでに射影済み）
                layers.append(ResidualBlock(prev_dim, hidden_dim, self.activation, dropout))
                prev_dim = hidden_dim

        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初期化（活性化関数に応じた調整）
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        時不変性保証付き前向き計算

        Args:
            y: [B, T, n] または [B, n] または [T, n]

        Returns:
            m: [B, T, m] または [B, m] または [T, m]
        """
        original_shape = y.shape
        is_single_step = len(original_shape) == 1  # [n]
        is_no_batch = len(original_shape) == 2 and y.size(0) != 1  # [T, n]

        # 形状統一化: [B, T, n]
        if is_single_step:
            y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, n]
        elif is_no_batch:
            y = y.unsqueeze(0)  # [1, T, n]
        elif len(original_shape) == 2:
            y = y.unsqueeze(1)  # [B, 1, n]

        B, T, n = y.shape

        if n != self.input_dim:
            raise ValueError(f"入力次元不一致: expected {self.input_dim}, got {n}")

        # 時間軸に沿った独立処理（時不変性保証）
        y_flat = y.view(-1, n)  # [B*T, n]

        # 入力正規化
        if self.normalize_input:
            if self.training:
                y_flat = self.input_norm(y_flat)
            else:
                # 推論時は保存された統計量を使用
                if self.track_running_stats:
                    # GPU/CPUデバイス整合性を確保
                    input_mean = self.input_mean.to(y_flat.device)
                    input_var = self.input_var.to(y_flat.device)
                    y_flat = (y_flat - input_mean) / torch.sqrt(input_var + self.eps)
                else:
                    y_flat = self.input_norm(y_flat)

        # コア変換（GPUデバイス整合性を確保）
        if hasattr(self.core_net, 'to'):
            self.core_net = self.core_net.to(y_flat.device)
        m_flat = self.core_net(y_flat)  # [B*T, m]

        # 出力正規化
        if self.normalize_output:
            if self.training:
                m_flat = self.output_norm(m_flat)
            else:
                # 推論時は保存された統計量を使用
                if self.track_running_stats:
                    # GPU/CPUデバイス整合性を確保
                    output_mean = self.output_mean.to(m_flat.device)
                    output_var = self.output_var.to(m_flat.device)
                    m_flat = (m_flat - output_mean) / torch.sqrt(output_var + self.eps)
                else:
                    m_flat = self.output_norm(m_flat)

        # 形状復元
        m = m_flat.view(B, T, self.output_dim)

        # 統計量更新（学習時のみ）
        if self.training and self.track_running_stats:
            self._update_statistics(y_flat, m_flat)

        # 元の形状に復元
        if is_single_step:
            return m.squeeze(0).squeeze(0)  # [m]
        elif is_no_batch:
            return m.squeeze(0)  # [T, m]
        elif len(original_shape) == 2:
            return m.squeeze(1)  # [B, m]
        else:
            return m  # [B, T, m]

    def _update_statistics(self, y_flat: torch.Tensor, m_flat: torch.Tensor):
        """統計量の指数移動平均更新"""
        with torch.no_grad():
            # 入力統計量
            input_mean_batch = y_flat.mean(dim=0)
            input_var_batch = y_flat.var(dim=0, unbiased=False)

            # 出力統計量
            output_mean_batch = m_flat.mean(dim=0)
            output_var_batch = m_flat.var(dim=0, unbiased=False)

            # 指数移動平均更新
            n = self.num_batches_tracked.item()
            momentum = self.momentum if n > 0 else 1.0

            # GPU/CPUデバイス整合性を確保した統計量更新
            input_mean_batch = input_mean_batch.to(self.input_mean.device)
            input_var_batch = input_var_batch.to(self.input_var.device)
            output_mean_batch = output_mean_batch.to(self.output_mean.device)
            output_var_batch = output_var_batch.to(self.output_var.device)

            self.input_mean.mul_(1 - momentum).add_(input_mean_batch, alpha=momentum)
            self.input_var.mul_(1 - momentum).add_(input_var_batch, alpha=momentum)
            self.output_mean.mul_(1 - momentum).add_(output_mean_batch, alpha=momentum)
            self.output_var.mul_(1 - momentum).add_(output_var_batch, alpha=momentum)

            self.num_batches_tracked += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """正規化統計量の取得"""
        if not self.track_running_stats:
            return {}

        return {
            'input_mean': self.input_mean.clone(),
            'input_var': self.input_var.clone(),
            'output_mean': self.output_mean.clone(),
            'output_var': self.output_var.clone(),
            'num_batches_tracked': self.num_batches_tracked.clone()
        }

    def load_statistics(self, stats: Dict[str, torch.Tensor]):
        """統計量の読み込み"""
        if not self.track_running_stats:
            return

        for key, value in stats.items():
            if hasattr(self, key):
                getattr(self, key).copy_(value)

    def verify_time_invariance(self, y1: torch.Tensor, y2: torch.Tensor, tol: float = 1e-6) -> bool:
        """
        時不変性の検証

        Args:
            y1, y2: 同じ値の観測（異なる時刻）
            tol: 許容誤差

        Returns:
            bool: 時不変性が保証されているか
        """
        with torch.no_grad():
            self.eval()
            m1 = self.forward(y1)
            m2 = self.forward(y2)

            max_diff = torch.max(torch.abs(m1 - m2)).item()
            return max_diff < tol


class time_invariantDecoder(nn.Module):
    """
    簡素化デコーダー g_α: R^m → R^n

    定式化変更：
    - 旧：スカラー m_t → 遅延埋め込み → 2パス処理
    - 新：多変量 m_t ∈ R^m → 直接的DNN → y_t ∈ R^n

    数学的根拠：
    多変量特徴量 m_t ∈ R^m は瞬時の情報を十分含むため、
    時間遅延情報の明示的埋め込みは不要
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: str = "mlp",
        hidden_dims: Optional[List[int]] = None,
        activation: str = "GELU",
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            input_dim: 入力特徴量次元 m
            output_dim: 出力観測次元 n
            architecture: アーキテクチャタイプ（"mlp", "resnet"）
            hidden_dims: 隠れ層次元リスト
            activation: 活性化関数
            dropout: ドロップアウト率
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture

        # デフォルト隠れ層設定（逆圧縮なので大きめ）
        if hidden_dims is None:
            # エンコーダーの逆順序で設定
            if input_dim <= 4:
                hidden_dims = [32, 64]
            elif input_dim <= 16:
                hidden_dims = [64, 128]
            else:
                hidden_dims = [64, 128, 256]

        # 活性化関数
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU()

        # ネットワーク構築
        if architecture == "mlp":
            self.net = self._build_mlp(input_dim, output_dim, hidden_dims, dropout)
        elif architecture == "resnet":
            self.net = self._build_resnet(input_dim, output_dim, hidden_dims, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Supported: ['mlp', 'resnet']")

        # 初期化
        self._initialize_weights()

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """標準MLPアーキテクチャ"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        # 出力層（活性化なし）
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _build_resnet(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """ResNet風アーキテクチャ"""
        layers = []
        prev_dim = input_dim

        # 入力射影
        if len(hidden_dims) > 0:
            first_hidden = hidden_dims[0]
            layers.append(nn.Linear(input_dim, first_hidden))
            prev_dim = first_hidden

        # 残差ブロック
        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:
                layers.append(ResidualBlock(prev_dim, hidden_dim, self.activation, dropout))
                prev_dim = hidden_dim

        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        多変量特徴量からの観測再構成

        Args:
            m: [B, T, m] または [B, m] または [T, m] 特徴量

        Returns:
            y: [B, T, n] または [B, n] または [T, n] 再構成観測
        """
        original_shape = m.shape
        is_single_step = len(original_shape) == 1  # [m]
        is_no_batch = len(original_shape) == 2 and m.size(0) != 1  # [T, m]

        # 形状統一化: [B, T, m]
        if is_single_step:
            m = m.unsqueeze(0).unsqueeze(0)  # [1, 1, m]
        elif is_no_batch:
            m = m.unsqueeze(0)  # [1, T, m]
        elif len(original_shape) == 2:
            m = m.unsqueeze(1)  # [B, 1, m]

        B, T, m_dim = m.shape

        if m_dim != self.input_dim:
            raise ValueError(f"入力次元不一致: expected {self.input_dim}, got {m_dim}")

        # 時間軸に沿った独立処理（GPUデバイス整合性を確保）
        m_flat = m.view(-1, m_dim)  # [B*T, m]
        if hasattr(self.net, 'to'):
            self.net = self.net.to(m_flat.device)
        y_flat = self.net(m_flat)   # [B*T, n]

        # 形状復元
        y = y_flat.view(B, T, self.output_dim)

        # 元の形状に復元
        if is_single_step:
            return y.squeeze(0).squeeze(0)  # [n]
        elif is_no_batch:
            return y.squeeze(0)  # [T, n]
        elif len(original_shape) == 2:
            return y.squeeze(1)  # [B, n]
        else:
            return y  # [B, T, n]


class ResidualBlock(nn.Module):
    """残差ブロック"""

    def __init__(self, input_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim)  # 残差接続のため入力次元に戻す
        )

        # ショートカット接続（次元が異なる場合）
        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, input_dim)  # 恒等写像維持
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)