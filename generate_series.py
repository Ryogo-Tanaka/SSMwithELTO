#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6次元の時系列データ（label='Y'）を生成して data/*.npz に保存するスクリプト。
デフォルトは「結合 Stuart–Landau ×3（連続時間, 6D）」。

出力:
  npz: {'Y': (T,6) ndarray, 't': (T,) ndarray}

使い方例:
  python scripts/generate_series.py --model stuart_landau --T 5000 --dt 0.01 --noise-std 1e-3 --seed 42
"""

import os
import argparse
import numpy as np


# ===== コア：汎用RK4 =====
def rk4_step(f, x, dt, noise_std=0.0, rng=None):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if noise_std > 0:
        if rng is None:
            x_next += noise_std * np.random.randn(*x.shape)
        else:
            x_next += noise_std * rng.standard_normal(size=x.shape)
    return x_next


# ===== モデル定義 =====
def make_series(model="stuart_landau", T=2000, dt=0.01, seed=0, noise_std=1e-3):
    """
    Returns:
      Y: shape (T,6)  # ラベル 'Y' 用の系列
      t: shape (T,)   # 時間（離散のときも便宜的に n*dt）
    """
    rng = np.random.default_rng(seed)

    if model.lower() in ["stuart_landau", "sl", "sl3"]:
        # state: [Re z1, Im z1, Re z2, Im z2, Re z3, Im z3]
        mu, c, kappa = 1.0, 0.1, 0.005
        omegas = np.array([1.00, 1.05, 0.95], dtype=float)

        def f(x):
            z = x.reshape(3, 2)
            zc = z[:, 0] + 1j * z[:, 1]  # complex
            out = []
            s = np.sum(zc)
            for k in range(3):
                coup = kappa * (s - 3 * zc[k])
                dz = (mu + 1j * omegas[k]) * zc[k] - (1 + 1j * c) * (np.abs(zc[k]) ** 2) * zc[k] + coup
                out.extend([np.real(dz), np.imag(dz)])
            return np.array(out, dtype=float)

        x = rng.normal(scale=0.1, size=6)

        Y = np.zeros((T, 6), dtype=float)
        t = np.arange(T) * dt
        for n in range(T):
            Y[n] = x
            x = rk4_step(f, x, dt, noise_std=noise_std, rng=rng)
        return Y, t

    elif model.lower() in ["henon", "henon3"]:
        # 離散時間Hénon×3（参考: 実験で離散が良いときに）
        # state: [x1,y1,x2,y2,x3,y3]
        a, b = 1.4, 0.3

        def step(x):
            X = x.reshape(3, 2)
            Xn = np.zeros_like(X)
            for k in range(3):
                xk, yk = X[k]
                xn = 1 - a * (xk ** 2) + yk
                yn = b * xk
                Xn[k] = [xn, yn]
            out = Xn.reshape(-1)
            if noise_std > 0:
                out += noise_std * rng.standard_normal(6)
            return out

        x = rng.normal(scale=0.1, size=6)
        Y = np.zeros((T, 6), dtype=float)
        t = np.arange(T) * dt  # 便宜的に n*dt
        for n in range(T):
            Y[n] = x
            x = step(x)
        return Y, t

    else:
        raise ValueError(f"Unknown model: {model}")


# ===== メイン：保存処理 =====
def main():
    parser = argparse.ArgumentParser(description="Generate 6D time-series (label 'Y') and save to data/*.npz")
    parser.add_argument("--model", type=str, default="stuart_landau",
                        help="stuart_landau | henon")
    parser.add_argument("--T", type=int, default=2000, help="sequence length (time steps)")
    parser.add_argument("--dt", type=float, default=0.01, help="time step")
    parser.add_argument("--noise-std", type=float, default=1e-3, help="Gaussian noise std added each step")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--outdir", type=str, default="data", help="output directory")
    parser.add_argument("--outfile", type=str, default="", help="custom output filename (optional)")
    args = parser.parse_args()

    # 生成
    Y, t = make_series(model=args.model, T=args.T, dt=args.dt, seed=args.seed, noise_std=args.noise_std)

    # 出力先
    os.makedirs(args.outdir, exist_ok=True)
    if args.outfile:
        out_path = os.path.join(args.outdir, args.outfile)
    else:
        tag = "sl3" if args.model.lower() in ["stuart_landau", "sl", "sl3"] else "henon3"
        # ファイル名に dt は安全のため小数点を置換
        dt_s = str(args.dt).replace(".", "p")
        out_path = os.path.join(args.outdir, f"{tag}_T{args.T}_dt{dt_s}_seed{args.seed}.npz")

    # 保存
    np.savez_compressed(out_path, Y=Y, t=t)
    print(f"[OK] saved: {out_path}")
    print(f"  Y shape: {Y.shape}, t shape: {t.shape}")
    print("  keys: 'Y', 't'")


if __name__ == "__main__":
    main()
