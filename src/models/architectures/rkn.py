# src/models/architectures/rkn.py
"""
RKN画像用アーキテクチャ - Factory Pattern準拠
rknEncoder: 画像(H,W,C) → 潜在表現(100次元)
rknDecoder: 潜在表現(100次元) → 画像(H,W,C)
rkn_targetDecoder: 潜在表現 → 状態平均(d_state次元)

形状処理: 単一(H,W,C), 時系列(T,H,W,C), バッチ(B,T,H,W,C)
"""

import math
from typing import Optional, List, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class rknEncoder(nn.Module):
    """
    画像エンコーダ: (H,W,C) → 潜在(100次元)
    Conv(5×5,32)→MaxPool→Conv(3×3,64)→MaxPool→FC(200)→FC(100)
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        hidden: int = 200,
        conv_channels: Tuple[int, int] = (32, 64),
        activation: str = "relu",
        normalize_input: bool = False,
        normalize_output: bool = False,
        track_running_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_resolution: 入力画像解像度 (H,W,C)
            feature_dim: 潜在次元
            hidden: FC隠れ次元
            conv_channels: CNN層チャネル数
            activation: 活性化関数
            normalize_input/output: 正規化フラグ
            track_running_stats: 統計量追跡
            momentum: BatchNorm慣性
            eps: 数値安定化
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

        H, W, C = input_resolution
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # CNN層
        self.conv1 = nn.Conv2d(C, conv_channels[0], kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # FC層
        conv_h, conv_w = H // 4, W // 4
        conv_output_size = conv_h * conv_w * conv_channels[1]
        self.fc1 = nn.Linear(conv_output_size, hidden)
        self.fc2 = nn.Linear(hidden, feature_dim)

        # 正規化層
        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(C, momentum=momentum, eps=eps)
        if self.normalize_output:
            self.output_norm = nn.BatchNorm1d(feature_dim, momentum=momentum, eps=eps)

        # 統計量
        if track_running_stats:
            self.register_buffer('input_mean', torch.zeros(C, H, W))
            self.register_buffer('input_var', torch.ones(C, H, W))
            self.register_buffer('output_mean', torch.zeros(feature_dim))
            self.register_buffer('output_var', torch.ones(feature_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        時不変性保証付きエンコーディング
        Args:
            o: (H,W,C) | (T,H,W,C) | (B,T,H,W,C)
        Returns:
            z: (feature_dim,) | (T,feature_dim) | (B,T,feature_dim)
        """
        original_shape = o.shape
        is_single_step = len(original_shape) == 3
        is_no_batch = len(original_shape) == 4

        # 形状統一: (B,T,H,W,C)
        if is_single_step:
            o = o.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            o = o.unsqueeze(0)
        elif len(original_shape) == 4:
            o = o.unsqueeze(1)

        B, T, H, W, C = o.shape
        expected_H, expected_W, expected_C = self.input_resolution
        if H != expected_H or W != expected_W or C != expected_C:
            raise ValueError(f"入力形状不一致: expected {self.input_resolution}, got ({H},{W},{C})")

        # 時間軸独立処理
        o_flat = o.view(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T,C,H,W)

        # 入力正規化
        if self.normalize_input:
            if self.training:
                o_flat = self.input_norm(o_flat)
            else:
                if self.track_running_stats:
                    input_mean = self.input_mean.to(o_flat.device)
                    input_var = self.input_var.to(o_flat.device)
                    o_flat = (o_flat - input_mean.unsqueeze(0)) / torch.sqrt(input_var.unsqueeze(0) + self.eps)
                else:
                    o_flat = self.input_norm(o_flat)

        # CNN処理
        x = self.activation(self.conv1(o_flat))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        # FC処理
        x = x.reshape(x.size(0), -1)
        x = self.activation(self.fc1(x))
        z_flat = self.fc2(x)

        # 出力正規化
        if self.normalize_output:
            if self.training:
                z_flat = self.output_norm(z_flat)
            else:
                if self.track_running_stats:
                    output_mean = self.output_mean.to(z_flat.device)
                    output_var = self.output_var.to(z_flat.device)
                    z_flat = (z_flat - output_mean) / torch.sqrt(output_var + self.eps)
                else:
                    z_flat = self.output_norm(z_flat)

        z = z_flat.view(B, T, self.feature_dim)

        # 統計量更新
        if self.training and self.track_running_stats:
            self._update_statistics(o_flat, z_flat)

        # 形状復元
        if is_single_step:
            return z.squeeze(0).squeeze(0)
        elif is_no_batch:
            return z.squeeze(0)
        elif len(original_shape) == 4:
            return z.squeeze(1)
        else:
            return z

    def _update_statistics(self, o_flat: torch.Tensor, z_flat: torch.Tensor):
        """統計量の指数移動平均更新"""
        with torch.no_grad():
            input_mean_batch = o_flat.mean(dim=0)
            input_var_batch = o_flat.var(dim=0, unbiased=False)
            output_mean_batch = z_flat.mean(dim=0)
            output_var_batch = z_flat.var(dim=0, unbiased=False)

            n = self.num_batches_tracked.item()
            momentum = self.momentum if n > 0 else 1.0

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
        """正規化統計量取得"""
        if not self.track_running_stats:
            return {}
        return {
            'input_mean': self.input_mean.clone(),
            'input_var': self.input_var.clone(),
            'output_mean': self.output_mean.clone(),
            'output_var': self.output_var.clone(),
            'num_batches_tracked': self.num_batches_tracked.clone()
        }

    def verify_time_invariance(self, o1: torch.Tensor, o2: torch.Tensor, tol: float = 1e-6) -> bool:
        """時不変性検証"""
        with torch.no_grad():
            self.eval()
            z1 = self.forward(o1)
            z2 = self.forward(o2)
            max_diff = torch.max(torch.abs(z1 - z2)).item()
            return max_diff < tol


class rknDecoder(nn.Module):
    """
    画像デコーダ: 潜在(100次元) → (H,W,C)
    FC(200)→FC(S)→Reshape→Upsample×2→Conv(3×3,64)→Upsample×2→Conv(5×5,32)→Conv(1×1,C)
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        grid: Optional[Tuple[int, int, int]] = None,
        hidden: int = 200,
        upsample_mode: str = "nearest",
        conv_channels: Tuple[int, int, int] = (64, 32, 1),
        activation: str = "relu",
        output_activation: str = "sigmoid",
        **kwargs
    ):
        """
        Args:
            input_resolution: 出力画像解像度 (H,W,C)
            feature_dim: 入力潜在次元
            grid: 中間グリッドサイズ (H',W',C')
            hidden: FC隠れ次元
            upsample_mode: アップサンプリング方法
            conv_channels: 逆畳み込み層チャネル数
            activation: 活性化関数
            output_activation: 出力活性化関数
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.upsample_mode = upsample_mode

        H, W, C = input_resolution
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # 出力活性化
        if output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Sigmoid()

        # グリッドサイズ自動設定
        if grid is None:
            grid = (H // 4, W // 4, conv_channels[0])
        self.grid = grid

        grid_h, grid_w, grid_c = grid
        grid_size = grid_h * grid_w * grid_c

        # FC層
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, grid_size)

        # Upsample + Conv層
        self.upsample1 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv1 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        時不変デコーディング
        Args:
            z: (feature_dim,) | (T,feature_dim) | (B,T,feature_dim)
        Returns:
            o: (H,W,C) | (T,H,W,C) | (B,T,H,W,C)
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2

        # 形状統一: (B,T,feature_dim)
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            z = z.unsqueeze(0)
        elif len(original_shape) == 2:
            z = z.unsqueeze(1)

        B, T, feature_dim = z.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"潜在特徴次元不一致: expected {self.feature_dim}, got {feature_dim}")

        # 時間軸独立処理
        z_flat = z.view(B * T, feature_dim)

        # FC処理
        x = self.activation(self.fc1(z_flat))
        x = self.activation(self.fc2(x))

        # Reshape to grid
        grid_h, grid_w, grid_c = self.grid
        x = x.view(-1, grid_c, grid_h, grid_w)

        # Upsample + Conv処理
        x = self.upsample1(x)
        x = self.activation(self.conv1(x))
        x = self.upsample2(x)
        x = self.output_activation(self.conv2(x))

        # (B*T,C,H,W) → (B*T,H,W,C)
        x = x.permute(0, 2, 3, 1)

        # 形状復元
        H, W, C = self.input_resolution
        o = x.view(B, T, H, W, C)

        if is_single_step:
            return o.squeeze(0).squeeze(0)
        elif is_no_batch:
            return o.squeeze(0)
        elif len(original_shape) == 2:
            return o.squeeze(1)
        else:
            return o


class rkn_targetDecoder(nn.Module):
    """
    ターゲット予測デコーダ: 潜在(d次元) → 状態平均(d_state次元)
    z_t → FC(hidden) → ReLU → FC(d_state) → output_activation
    """

    def __init__(
        self,
        feature_dim: int = 100,
        state_dim: int = 8,
        hidden: int = 50,
        activation: str = "relu",
        output_activation: str = "linear",
        **kwargs
    ):
        """
        Args:
            feature_dim: 入力潜在次元
            state_dim: 出力状態次元
            hidden: FC隠れ次元
            activation: 活性化関数
            output_activation: 出力活性化関数
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.hidden = hidden

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # 出力活性化
        if output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Identity()

        # FC層
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, state_dim)

        self._initialize_weights()

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        時不変状態推定
        Args:
            z: (feature_dim,) | (T,feature_dim) | (B,T,feature_dim)
        Returns:
            μ̂_s: (state_dim,) | (T,state_dim) | (B,T,state_dim)
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2

        # 形状統一: (B,T,feature_dim)
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            z = z.unsqueeze(0)
        elif len(original_shape) == 2:
            z = z.unsqueeze(1)

        B, T, feature_dim = z.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"潜在特徴次元不一致: expected {self.feature_dim}, got {feature_dim}")

        # 時間軸独立処理
        z_flat = z.view(B * T, feature_dim)

        # FC処理
        x = self.activation(self.fc1(z_flat))
        x = self.fc2(x)
        mu_s_flat = self.output_activation(x)

        # 形状復元
        mu_s = mu_s_flat.view(B, T, self.state_dim)

        if is_single_step:
            return mu_s.squeeze(0).squeeze(0)
        elif is_no_batch:
            return mu_s.squeeze(0)
        elif len(original_shape) == 2:
            return mu_s.squeeze(1)
        else:
            return mu_s


def make_state_decoder(cfg: Dict[str, Any]) -> rkn_targetDecoder:
    """
    状態デコーダファクトリ関数
    Args:
        cfg: 設定辞書 (feature_dim, state_dim, hidden)
    Returns:
        rkn_targetDecoder インスタンス
    """
    feature_dim = cfg.get('feature_dim', 100)
    state_dim = cfg.get('state_dim')
    hidden = cfg.get('hidden', 50)

    if state_dim is None:
        raise ValueError("state_dim は必須パラメータです")

    return rkn_targetDecoder(
        feature_dim=feature_dim,
        state_dim=state_dim,
        hidden=hidden,
        **{k: v for k, v in cfg.items() if k not in ['feature_dim', 'state_dim', 'hidden']}
    )
