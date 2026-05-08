"""CNN-LSTM-CNN baseline for quad-link image prediction.

Architecture:
    image (B, T, 48, 48, 1)
    -> cnn_imageEncoder -> features (B, T, F)
    -> nn.LSTM (input=F, hidden=F, num_layers=1) -> hidden states (B, T, F)
    -> cnn_imageDecoder -> predicted images (B, T, 48, 48, 1)

Training: teacher forcing on next-step prediction.
    For input y[0..L-1], the LSTM produces hidden states h[0..L-1].
    h[t] is interpreted as the predicted feature for time t+1.
    Loss = MSE(decoder(h[0..L-2]), y[1..L-1]).

Inference: free rollout from a context window of length C.
    rollout(y_ctx) -> predicted next H frames (B, H, 48, 48, 1).
    Step 1 (h=1) corresponds to y[C], h=2 to y[C+1], ..., h=H to y[C+H-1].
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.architectures.cnn_image import cnn_imageEncoder, cnn_imageDecoder


class CnnLstmCnn(nn.Module):
    """CNN-LSTM-CNN next-step image predictor."""

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        lstm_hidden: int = 100,
        lstm_layers: int = 1,
        encoder_hidden: int = 200,
        decoder_hidden: int = 200,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.encoder = cnn_imageEncoder(
            input_resolution=input_resolution,
            feature_dim=feature_dim,
            hidden=encoder_hidden,
        )
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Project LSTM output back to feature_dim for the decoder.
        if lstm_hidden == feature_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(lstm_hidden, feature_dim)
        self.decoder = cnn_imageDecoder(
            input_resolution=input_resolution,
            feature_dim=feature_dim,
            hidden=decoder_hidden,
        )

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, T, H, W, C) -> (B, T, F)."""
        return self.encoder(y.contiguous())

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        """e: (B, T, F) -> (B, T, H, W, C)."""
        return self.decoder(e.contiguous())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward.

        Args:
            y: (B, T, H, W, C) input frames.

        Returns:
            y_pred: (B, T, H, W, C). y_pred[:, t] is the predicted next frame
                given y[:, :t+1] (so loss uses y_pred[:, :-1] vs y[:, 1:]).
        """
        e = self.encoder(y.contiguous())  # (B, T, F)
        h, _ = self.lstm(e)  # (B, T, hidden)
        e_pred = self.proj(h).contiguous()  # (B, T, F)
        y_pred = self.decoder(e_pred)
        return y_pred

    @torch.no_grad()
    def rollout(
        self,
        y_ctx: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Free rollout from context y_ctx for `horizon` steps.

        Args:
            y_ctx: (B, C, H, W, C_img) context frames.
            horizon: number of frames to predict after the context.

        Returns:
            y_pred: (B, horizon, H, W, C_img). y_pred[:, h-1] corresponds
                to the prediction for time index C+h-1 (h = 1..horizon).
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        e_ctx = self.encoder(y_ctx.contiguous())
        h_seq, hidden = self.lstm(e_ctx)
        e_pred = self.proj(h_seq[:, -1:, :].contiguous()).contiguous()

        preds = [e_pred]
        for _ in range(horizon - 1):
            h_seq, hidden = self.lstm(e_pred, hidden)
            e_pred = self.proj(h_seq).contiguous()
            preds.append(e_pred)

        e_preds = torch.cat(preds, dim=1).contiguous()  # (B, horizon, F)
        y_preds = self.decoder(e_preds)
        return y_preds.clamp(0.0, 1.0)
