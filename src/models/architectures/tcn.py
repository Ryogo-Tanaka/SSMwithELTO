# tcn.py
# Temporal Convolutional Network based encoder/decoder
# - Encoder: (B, T, d) -> (B, T, p)  [p=1: "stationary" scalar feature series]
# - Decoder: (B, T, 1) -> (B, T, n)  [Takens window + Stat/Trend two-path]
import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Low-level building blocks
# --------------------------
class _CausalConv1dZeroMean(nn.Module):
    """
    Causal dilated Conv1d with zero-mean kernels along the time axis:
        sum_k W[:, :, k] = 0
    -> DC / low-frequency suppression (high-pass property).
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        assert kernel_size >= 2, "kernel_size >= 2 is recommended for high-pass"
        self.c_in = c_in
        self.c_out = c_out
        self.k = kernel_size
        self.d = dilation

        # learnable raw weights; we subtract kernel-axis mean on-the-fly
        self.weight_raw = nn.Parameter(torch.empty(c_out, c_in, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(c_out))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight_raw, a=math.sqrt(5))

    def zero_mean_weight(self) -> torch.Tensor:
        # shape: [C_out, C_in, K]
        return self.weight_raw - self.weight_raw.mean(dim=2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        # left padding only (causal)
        pad_left = self.d * (self.k - 1)
        x = F.pad(x, (pad_left, 0))
        # enforce sum_k W = 0 by reparameterization
        w = self.zero_mean_weight()
        return F.conv1d(x, w, self.bias, stride=1, padding=0, dilation=self.d)


class _ResidualTCNBlock(nn.Module):
    """
    [Causal dilated conv (ΣW=0)] -> Act -> (Dropout) -> LayerNorm -> Residual
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 activation: str = "GELU", dropout: float = 0.0):
        super().__init__()
        self.conv = _CausalConv1dZeroMean(
            c_in=channels, c_out=channels,
            kernel_size=kernel_size, dilation=dilation, bias=True
        )
        self.act = getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(channels)  # applied on last dim (can accept [B,C] or [B,T,C])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        y = self.conv(x)                  # [B, C, T]
        y = self.act(y)
        y = self.dropout(y)
        # LayerNorm expects features in the last dim -> transpose to [B, T, C]
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        return x + y                      # residual add

    # ---- one-step update for stateful inference ----
    @torch.no_grad()
    def step_infer(self,
                   h_prev_t: torch.Tensor,        # [B, C] : h^{(ℓ-1)}_t  (current time, for residual)
                   buf_prev: torch.Tensor,        # [B, C, R_prev] : ring buffer of h^{(ℓ-1)}
                   ptr_prev: int                  # write pointer of buf_prev (points to next write index)
                   ) -> torch.Tensor:
        """
        Compute one-step output h^{(ℓ)}_t using only the previous-layer ring buffer.
        This runs under no_grad by default; remove decorator if you need grad.
        """
        B, C, R_prev = buf_prev.shape
        K = self.conv.k
        d = self.conv.d
        # gather taps: indices are (ptr_prev - 1 - k*d) % R_prev
        idxs = [ (ptr_prev - 1 - k * d) % R_prev for k in range(K) ]
        taps = torch.stack([buf_prev[:, :, i] for i in idxs], dim=-1)  # [B, C, K]

        # weights: [C_out, C_in, K] with zero-mean along K (DC cut)
        w = self.conv.zero_mean_weight()                               # [C, C, K]
        b = self.conv.bias                                             # [C] or None

        # matrix multiply over taps: sum_k ( tap_k @ w[:,:,k]^T )
        # taps: [B,C,K], w: [C,C,K] -> out: [B,C]
        out = torch.zeros(B, C, device=taps.device, dtype=taps.dtype)
        for k in range(K):
            out += taps[:, :, k] @ w[:, :, k].transpose(0, 1)          # [B,C] @ [C,C]^T -> [B,C]
        if b is not None:
            out = out + b

        # activation + layernorm + residual
        out = self.act(out)
        # LayerNorm can accept [B,C] directly (normalized_shape=C)
        out = self.ln(out)
        h_cur = h_prev_t + out
        return h_cur


# --------------------------
# Encoder
# --------------------------
class tcnEncoder(nn.Module):
    """
    Residual-TCN encoder
      (B, T, d) --permute--> (B, d, T)
        --1x1 Conv(d->C)--> (B, C, T)
        --[Residual TCN x L]--> (B, C, T)
        --1x1 Conv(C->p)--> (B, p, T) --permute--> (B, T, p)
    Args:
      input_dim:   d
      output_dim:  p (default=1: scalar feature series m_t)
      channels:    C (hidden channels)
      layers:      L (number of residual blocks)
      kernel_size: K (FIR length; recommend 3)
      activation:  "GELU" or "ReLU" etc.
      dropout:     dropout prob inside residual block
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 channels: int = 64,
                 layers: int = 6,
                 kernel_size: int = 3,
                 activation: str = "GELU",
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels
        self.layers = layers
        self.kernel_size = kernel_size

        self.in_proj  = nn.Conv1d(input_dim, channels, kernel_size=1)
        blocks = []
        for ell in range(layers):
            dilation = 2 ** ell  # 1,2,4,8,...
            blocks.append(_ResidualTCNBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation=activation,
                dropout=dropout
            ))
        self.tcn = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(channels, output_dim, kernel_size=1)

        # small init on out_proj for stable early-phase training
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # ---------- stateless forward (batch) ----------
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B, T, d]  ->  u: [B, T, p]
        """
        x = y.transpose(1, 2)        # [B, d, T]
        h = self.in_proj(x)          # [B, C, T]
        h = self.tcn(h)              # [B, C, T]
        u = self.out_proj(h)         # [B, p, T]
        return u.transpose(1, 2)     # [B, T, p]

    # ---------- helpers ----------
    @torch.no_grad()
    def receptive_field(self) -> int:
        """Total receptive field length R_F = 1 + (K-1)(2^L - 1)."""
        K = self.kernel_size
        L = self.layers
        return 1 + (K - 1) * (2 ** L - 1)

    @torch.no_grad()
    def init_state(self,
                   batch_size: int,
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
        """
        Initialize ring buffers for stateful one-step inference.
        Returns a dict with:
          - 'bufs': list of length L; bufs[ℓ] is [B, C, R_ℓ] storing h^{(ℓ)} history
          - 'ptrs': list of length L; current write positions (int)
        Note: layer ℓ uses buffer of previous layer (ℓ-1).
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        B = batch_size
        C = self.channels
        K = self.kernel_size
        bufs: List[torch.Tensor] = []
        ptrs: List[int] = []
        # level 0..L-1 buffers (store h^{(0)}..h^{(L-1)})
        for ell in range(1, self.layers + 1):
            d = 2 ** (ell - 1)
            R = d * (K - 1)
            # R could be 0 if K=1; enforce at least 1
            R = max(R, 1)
            bufs.append(torch.zeros(B, C, R, device=device, dtype=dtype))
            ptrs.append(0)
        return {"bufs": bufs, "ptrs": ptrs}

    @torch.no_grad()
    def forward_step(self,
                     y_t: torch.Tensor,        # [B, d]
                     state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        One-step stateful inference.
          y_t:  [B, d]
          returns (m_t: [B, p], state)
        """
        assert y_t.dim() == 2, "y_t must be [B, d]"
        B, d_in = y_t.shape
        assert d_in == self.input_dim, f"expected d={self.input_dim}, got {d_in}"
        bufs: List[torch.Tensor] = state["bufs"]
        ptrs: List[int] = state["ptrs"]
        C = self.channels

        # 1) input 1x1 conv to obtain h^{(0)}_t
        # in_proj is Conv1d (in_channels=d, out=C) -> accept [B,d,1]
        h0 = self.in_proj(y_t.unsqueeze(-1)).squeeze(-1)    # [B, C]

        # push h0 into buffer level 0 (used by layer 1)
        if bufs:
            buf0 = bufs[0]
            ptr0 = ptrs[0]
            buf0[:, :, ptr0] = h0
            ptrs[0] = (ptr0 + 1) % buf0.shape[-1]

        h_prev = h0
        # 2) per-layer update using ring buffers of previous layer
        for ell, block in enumerate(self.tcn, start=1):
            # previous-layer buffer index = ell-1
            buf_prev = bufs[ell - 1]
            ptr_prev = ptrs[ell - 1]
            # compute current layer output h^{(ℓ)}_t
            h_cur = block.step_infer(h_prev_t=h_prev, buf_prev=buf_prev, ptr_prev=ptr_prev)
            # push h^{(ℓ)}_t into its own buffer (unless this is the top layer)
            if ell < self.layers:
                buf_this = bufs[ell]
                ptr_this = ptrs[ell]
                buf_this[:, :, ptr_this] = h_cur
                ptrs[ell] = (ptr_this + 1) % buf_this.shape[-1]
            h_prev = h_cur

        # 3) final 1x1 conv to project to p
        u_t = self.out_proj(h_prev.unsqueeze(-1)).squeeze(-1)   # [B, p]
        # state mutated in-place (bufs/ptrs updated)
        return u_t, state


# --------------------------
# Decoder
# --------------------------
def _takens_window(u: torch.Tensor, window: int, tau: int) -> torch.Tensor:
    """
    Make Takens-style delay embedding (left-zero-padded).
      u: [B, T, 1]  ->  Z: [B, T, window]
      Z[t] = [u_t, u_{t-τ}, ..., u_{t-(window-1)τ}]
    """
    B, T, P = u.shape
    assert P == 1, "Decoder expects scalar feature series u: [B,T,1]."
    # left pad with (window-1)*tau zeros
    pad = (window - 1) * tau
    if pad > 0:
        u_pad = F.pad(u.transpose(1, 2), (pad, 0)).transpose(1, 2)  # [B, T+pad, 1]
    else:
        u_pad = u
    # gather indices
    cols = []
    for i in range(window):
        shift = i * tau
        cols.append(u_pad[:, pad - shift: pad - shift + T, 0])  # [B, T]
    Z = torch.stack(cols, dim=-1)  # [B, T, window]
    return Z


class tcnDecoder(nn.Module):
    """
    Takens-window + two-path (Stat/Trend) decoder:
      Input:  u  [B, T, 1]
      Steps:
        Z = Takens(u; window, tau)        # [B, T, l]
        S = MovingAverage(u; K_ma)        # [B, T, 1]
        Stat-path:  Z -> MLP(l->128->d_h)
        Trend-path: S -> GRU(1->h_gru) -> Linear(h_gru->d_h)
        Merge: H = H_S + H_T
        Output: y_hat = Linear(d_h -> n)  # [B, T, n]
    Args:
      output_dim:  n   (dimension of reconstructed observation)
      window:      l   (Takens window length)
      tau:         τ   (delay step)
      hidden:      d_h (merged hidden size before final linear)
      ma_kernel:   K_ma (moving-average window for trend cue)
      gru_hidden:  hidden size of the GRU in trend path
      activation:  "GELU" or "ReLU" etc.
      dropout:     dropout prob in Stat-path MLP
    """
    def __init__(self,
                 output_dim: int,
                 window: int = 8,
                 tau: int = 1,
                 hidden: int = 64,
                 ma_kernel: int = 64,
                 gru_hidden: int = 64,
                 activation: str = "GELU",
                 dropout: float = 0.0):
        super().__init__()
        self.output_dim = output_dim
        self.window = window
        self.tau = tau
        self.ma_kernel = ma_kernel
        self.hidden = hidden
        self.gru_hidden = gru_hidden

        # Stat-path MLP: l -> 128 -> hidden
        Act = getattr(nn, activation) if hasattr(nn, activation) else nn.GELU
        self.stat_mlp = nn.Sequential(
            nn.Linear(window, 128),
            Act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, hidden),
            Act(),
        )

        # Trend-path: GRU over MA signal
        self.trend_gru = nn.GRU(input_size=1, hidden_size=gru_hidden,
                                num_layers=1, batch_first=True)
        self.trend_proj = nn.Linear(gru_hidden, hidden)

        # output: hidden -> n
        self.out = nn.Linear(hidden, output_dim)

        # register moving-average kernel as a buffer (non-trainable) for batch forward
        if ma_kernel <= 1:
            w = torch.ones(1, 1, 1)
        else:
            w = torch.ones(1, 1, ma_kernel) / float(ma_kernel)
        self.register_buffer("_ma_kernel", w)

    # ---------- stateless forward (batch) ----------
    def _moving_average(self, u: torch.Tensor) -> torch.Tensor:
        """
        Causal moving average over time axis with kernel length K_ma.
          u: [B, T, 1] -> s: [B, T, 1]
        """
        B, T, _ = u.shape
        K = self._ma_kernel.shape[-1]
        # causal left padding
        u1 = F.pad(u.transpose(1, 2), (K - 1, 0)).transpose(1, 2)  # [B, T+K-1, 1]
        s = F.conv1d(u1.transpose(1, 2), self._ma_kernel, padding=0).transpose(1, 2)
        return s  # [B, T, 1]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: [B, T, 1]  ->  y_hat: [B, T, n]
        """
        # Takens window
        Z = _takens_window(u, self.window, self.tau)          # [B, T, l]
        # Trend cue by causal moving average
        S = self._moving_average(u)                           # [B, T, 1]

        # Stat-path (time-wise MLP)
        Hs = self.stat_mlp(Z)                                 # [B, T, hidden]

        # Trend-path (GRU over time)
        Ht_seq, _ = self.trend_gru(S)                         # [B, T, gru_hidden]
        Ht = self.trend_proj(Ht_seq)                          # [B, T, hidden]

        # Merge and output
        H = Hs + Ht                                           # [B, T, hidden]
        y_hat = self.out(H)                                   # [B, T, n]
        return y_hat

    # ---------- stateful inference (one-step API) ----------
    @torch.no_grad()
    def init_state(self,
                   batch_size: int,
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
        """
        Initialize state for one-step decoding:
          - Z ring buffer of length Lz = (l-1)*tau + 1  (store last Lz samples of m)
          - ptrZ pointer (int)
          - MA buffer of length K_ma (for exact rolling sum), ptrMA and running sum S
          - GRU hidden state h (shape [1,B,gru_hidden])
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        B = batch_size
        l = self.window
        tau = self.tau
        Lz = (l - 1) * tau + 1
        Kma = max(int(self.ma_kernel), 1)

        state = {
            "Z": torch.zeros(B, Lz, 1, device=device, dtype=dtype),
            "ptrZ": 0,
            "MA_buf": torch.zeros(B, Kma, 1, device=device, dtype=dtype),
            "ptrMA": 0,
            "MA_sum": torch.zeros(B, 1, device=device, dtype=dtype),  # running sum over MA_buf
            "gru_h": torch.zeros(1, B, self.gru_hidden, device=device, dtype=dtype),
        }
        return state

    @torch.no_grad()
    def forward_step(self,
                     u_t: torch.Tensor,       # [B, 1]
                     state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        One-step decoding:
          u_t: [B, 1]  ->  y_hat_t: [B, n]
        """
        assert u_t.dim() == 2 and u_t.size(-1) == 1, "u_t must be [B, 1]"
        B = u_t.shape[0]
        # 1) Takens window update
        Z = state["Z"]             # [B, Lz, 1]
        ptrZ = state["ptrZ"]
        Lz = Z.shape[1]
        Z[:, ptrZ, 0] = u_t[:, 0]
        # build z_t by striding tau backwards over Z
        l = self.window
        tau = self.tau
        idxs = [(ptrZ - i * tau) % Lz for i in range(l)]
        z_t = torch.stack([Z[:, idx, 0] for idx in idxs], dim=-1)   # [B, l]
        state["ptrZ"] = (ptrZ + 1) % Lz

        # 2) moving average (difference update with fixed-size buffer)
        Kma = max(int(self.ma_kernel), 1)
        MA_buf = state["MA_buf"]     # [B, Kma, 1]
        ptrMA = state["ptrMA"]
        popped = MA_buf[:, ptrMA, 0].clone()      # [B]
        MA_buf[:, ptrMA, 0] = u_t[:, 0]
        state["ptrMA"] = (ptrMA + 1) % Kma
        MA_sum = state["MA_sum"][:, 0]            # [B]
        MA_sum = MA_sum + u_t[:, 0] - popped      # [B]
        state["MA_sum"][:, 0] = MA_sum
        s_t = (MA_sum / float(Kma)).unsqueeze(-1).unsqueeze(-1)     # [B,1,1] for GRU

        # 3) Stat-path (MLP on z_t)
        Hs = self.stat_mlp(z_t)      # [B, hidden]

        # 4) Trend-path (GRU on s_t)
        h_prev = state["gru_h"]      # [1, B, H]
        out_seq, h_new = self.trend_gru(s_t, h_prev)   # out_seq: [B,1,H]
        state["gru_h"] = h_new
        Ht = self.trend_proj(out_seq[:, 0, :])   # [B, hidden]

        # 5) Merge and output
        H = Hs + Ht
        y_hat_t = self.out(H)        # [B, n]
        return y_hat_t, state
