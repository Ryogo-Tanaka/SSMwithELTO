#!/usr/bin/env python3
# eval.py

import os
import argparse
import datetime
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.encoder import build_encoder
from src.models.decoder import build_decoder
from src.ssm.realization import build_realization
from src.ssm.observation import CMEObservation
from src.utils.gpu_utils import select_device


def load_data(path: str) -> np.ndarray:
    """.npz/.npy ファイルから特徴量行列 X (T×d) を読み込む"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        if "X" not in data:
            raise ValueError(f".npz must contain key 'X'; found {list(data.keys())}")
        return data["X"]
    elif ext == ".npy":
        return np.load(path)
    else:
        raise ValueError("Unsupported data format: " + ext)


def main():
    # -----------------------------
    # 引数パース
    # -----------------------------
    p = argparse.ArgumentParser(description="Evaluate model vs sequence length")
    p.add_argument("--config", "-c", required=True, help="path to config yaml")
    p.add_argument("--min-length", type=int, required=True,
                   help="minimum sequence length to evaluate")
    p.add_argument("--max-length", type=int, required=True,
                   help="maximum sequence length to evaluate")
    p.add_argument("--step", type=int, default=10,
                   help="step size for sequence length sweep")
    args = p.parse_args()

    # -----------------------------
    # 設定ロード
    # -----------------------------
    cfg = yaml.safe_load(open(args.config, "r"))
    # データ読み込み
    X_np = load_data(cfg["training"]["data_path"])  # shape (T, d)

    # デバイス設定
    device = select_device(prefer_memory=True)

    # モジュール構築
    enc_cfg = cfg["model"]["encoder"]
    dec_cfg = cfg["model"]["decoder_y2d"]
    real_cfg = cfg["ssm"]["realization"]
    obs_cfg  = cfg["ssm"]["observation"]

    # 辞書→Namespace 変換 (build_* が .type 属性を期待する)
    from types import SimpleNamespace
    enc_ns  = SimpleNamespace(**enc_cfg)
    dec_ns  = SimpleNamespace(**dec_cfg)
    real_ns = SimpleNamespace(**real_cfg)
    obs_ns  = SimpleNamespace(**obs_cfg)

    encoder    = build_encoder(enc_ns).to(device)
    decoder    = build_decoder(dec_ns).to(device)
    realization= build_realization(real_ns)
    observation= CMEObservation(
        kernel=obs_ns.kernel,
        gamma=obs_ns.gamma,
        reg_lambda=obs_ns.reg_lambda,
        approx=obs_ns.approx,
        approx_rank=obs_ns.approx_rank
    )

    # 評価用ループ
    lengths = list(range(args.min_length, args.max_length + 1, args.step))
    results = {"length": [], "mse": [], "sv_sum": [], "objective": []}

    h = real_cfg["h"]
    sv_weight = cfg["training"]["svd_weight"]

    for L in lengths:
        if L <= 2*h:
            # Hankel を組めない系列長はスキップ
            continue

        # データ切り出し & テンソル化
        X = torch.from_numpy(X_np[:L]).to(device)  # (L, d)

        # 1) エンコード
        Y = encoder(X)                              # (L, p)

        # 2) サブスペース同定
        realization.fit(Y)
        X_state = realization.filter(Y)             # (L-2h+1, r)

        # 3) CME デコード
        observation.fit(X_state, Y[: X_state.size(0)])
        Y_pred = observation.decode(X_state)        # (L-2h+1, p)

        # 4) デコーダを通して再構成
        X_rec = decoder(Y_pred)                     # (L-2h+1, d)

        # 5) １ステップ予測誤差 (MSE)
        X_eff = X[h-1 : -(h)]                       # (L-2h+1, d)
        mse = torch.mean((X_rec - X_eff) ** 2).item()

        # 6) 特異値和
        sv_sum = realization.singular_value_sum().item()

        # 7) 目的関数
        obj = mse + sv_weight * sv_sum

        # ログに保存
        results["length"].append(L)
        results["mse"].append(mse)
        results["sv_sum"].append(sv_sum)
        results["objective"].append(obj)

        print(f"Length={L:3d}  MSE={mse:.4e}  SVsum={sv_sum:.4e}  Obj={obj:.4e}")

    # -----------------------------
    # プロット
    # -----------------------------
    out_dir = cfg["visualization"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 目的関数
    plt.figure()
    plt.plot(results["length"], results["objective"], marker="o")
    plt.title("Objective vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("MSE + weight·SVsum")
    plt.grid(True)
    fname = f"objective_{ts}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

    # MSE
    plt.figure()
    plt.plot(results["length"], results["mse"], marker="o")
    plt.title("One-step MSE vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("MSE")
    plt.grid(True)
    fname = f"mse_{ts}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

    # 特異値和
    plt.figure()
    plt.plot(results["length"], results["sv_sum"], marker="o")
    plt.title("Singular Value Sum vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Sum of Singular Values")
    plt.grid(True)
    fname = f"sv_sum_{ts}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

    print("Plots saved to", out_dir)


if __name__ == "__main__":
    main()
