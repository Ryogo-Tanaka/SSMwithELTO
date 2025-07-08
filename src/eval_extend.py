# src/eval_extend.py

import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from ssm.realization import Realization, RealizationError
from ssm.observation import CMEObservation
from models.architectures._mlp import _mlpEncoder, _mlpDecoder


def parse_args():
    """
    コマンドライン引数をパースする関数。

    --config       : YAML 設定ファイルパス
    --min-length   : シーケンス長の最小値
    --max-length   : シーケンス長の最大値
    --step         : シーケンス長の刻み幅
    --train-ratio  : 訓練データの比率 (0.0～1.0)
    --reg-term     : 正則化項の計算方法 ('sum' または 'squared')
    --sv-weight    : 特異値正則化の基準重み
    --epochs       : 各長さで学習するエポック数
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True, help="YAML 設定ファイル")
    p.add_argument("--min-length",  type=int, default=20,  help="最小シーケンス長")
    p.add_argument("--max-length",  type=int, default=100, help="最大シーケンス長")
    p.add_argument("--step",        type=int, default=10,  help="刻み幅")
    p.add_argument("--train-ratio", type=float, default=0.6,help="訓練データ比率")
    p.add_argument("--reg-term",    choices=["sum","squared"], default="sum",
                   help="正則化項: sum=特異値和, squared=(1−σ)^2")
    # p.add_argument("--sv-weight",   type=float, default=0.1,
    #                help="正則化重みの基準値")
    p.add_argument("--epochs",      type=int,   default=50,
                   help="各シーケンス長で実行するエポック数")
    return p.parse_args()


def load_data(path):
    """
    .npz ファイルの 'Y' キーから観測行列を読み込む。

    戻り値:
      Tensor Y: shape (T, d)
    """
    data = np.load(path)
    return torch.from_numpy(data["Y"].astype(np.float32))


def build_loaders(Y, train_ratio, batch_size, seed):
    """
    観測データ Y を訓練用とテスト用に分割し、
    DataLoader を生成する。

    引数:
      Y           : Tensor (T, d) の観測時系列
      train_ratio : 訓練データの割合 (0～1)
      batch_size  : ミニバッチサイズ
      seed        : 分割シャッフルの乱数シード

    戻り値:
      train_loader, test_loader (両方とも shuffle=False)
    """
    T, d = Y.shape
    n_train = int(train_ratio * T)
    Y_tr, Y_te = Y[:n_train], Y[n_train:]
    ds_tr = TensorDataset(Y_tr, Y_tr)
    ds_te = TensorDataset(Y_te, Y_te)
    # loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)
    # loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, drop_last=True)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader_tr, loader_te


def evaluate_model(loader, encoder, decoder, real, obs, sv_weight, device):
    """
    モデルを固定したまま評価する関数。

    引数:
      loader     : DataLoader (shuffle=False)
      encoder    : _mlpEncoder インスタンス (学習済み)
      decoder    : _mlpDecoder インスタンス (学習済み)
      real       : Realization インスタンス
      obs        : CMEObservation インスタンス
      sv_weight  : 正則化重み
      reg_term   : 'sum' , 'squared', 'abs', ''bounded' ->削除
      device     : torch.device

    戻り値:
      mse_avg: 平均 MSE
      reg_avg: 平均 正則化項
    """
    mse_loss = torch.nn.MSELoss()
    encoder.eval(); decoder.eval()
    total_mse = 0.0
    total_reg = 0.0
    count = 0
    # sv_histoy = []
    with torch.no_grad():
        for Y_batch, _ in loader:
            Yb = Y_batch.to(device)
            T = Yb.size(0)
            Yf = encoder(Yb)
            real.fit(Yf)
            Xs = real.filter(Yf)
            obs.fit(Xs, Yf)
            Yp = obs.decode()
            Y_pred = decoder(Yp)
            Yb_eff = Yb[real.h+1 : T-real.h + 2, :]  #h が　h+1 に-> one stepで
            mse = torch.norm(Y_pred - Yb_eff, dim=1).mean().item()
            reg = real.singular_value_reg(sv_weight).item()
            # sv_history.append(real._L_vals.detach().cpu().numpy())

            total_mse += mse  # 実験高速化のため、出力はバッチごとの誤差の平均にする
            total_reg += reg
            # sv_history = np.stack(sv_history, axis=0)
            count += 1
    return total_mse / count, total_reg / count  # remove sv_history


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    # 観測データ読み込み
    Y_full = load_data(cfg["training"]["data_path"])
    # print("Y_full length:", Y_full.shape[0])  #debag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {k: [] for k in ["lengths","train_mse","test_mse",
                                "train_reg","test_reg","diff_abs"]}

    # 正則化重みモード: 'fixed' or 'length_dependent'
    mode = cfg["training"].get("sv_weight_mode", "fixed")

    # 結果プロットのためのフォルダ作成
    ep = args.epochs
    svw_label = f"{mode}"
    out_root = cfg["visualization"]["output_dir"]
    outdir = os.path.join(out_root, f"ep{ep}_sv_{svw_label}_{cfg["ssm"]["realization"]["reg_type"]}")
    os.makedirs(outdir, exist_ok=True)

    # 損失ログ用ファイル作成
    log_path = os.path.join(outdir, "training_loss_log.txt")
    log_file = open(log_path, "w")
    # ヘッダ行（任意で消してもOK）
    log_file.write("L,epoch,step,loss,mse,reg\n")

    # 特異値履歴用サブフォルダ
    sv_hst_dir = os.path.join(outdir, "sv_hst")
    os.makedirs(sv_hst_dir, exist_ok=True)

    _flag = 0
    
    # シーケンス長ごとのループ
    for L in range(args.min_length, args.max_length + 1, args.step):
        Y_seq = Y_full[:L].to(device)
        # print("Y_seq length:", Y_seq.shape[0])

        # バッチサイズを L//3 に設定（最低1）
        # batch_size = max(1, L // 3)
        batch_size = int((args.train_ratio * L) / 3)
        # print(f"batch_size : {batch_size}")  # debag
        train_loader, test_loader = build_loaders(
            Y_seq, args.train_ratio, batch_size, cfg["training"]["seed"]
        )

        # sv_weight の決定
        if mode == "fixed":
            sv_weight = cfg["training"]["sv_weight"]
        elif mode == "L1":  # length_dependent
            sv_weight = 1.0 / L
        elif mode == "L2":
            sv_weight = 1.0 / (L ** 2)

        # モデル＆オプティマイザ初期化（エポック前）
        d = Y_seq.size(1)
        encoder = _mlpEncoder(input_dim=d,
                              hidden_sizes=cfg["model"]["encoder_hidden"],
                              output_dim=cfg["model"]["feature_dim"],
                              activation=cfg["model"]["activation"]).to(device)  # 変更: relu -> tanh
        decoder = _mlpDecoder(input_dim=cfg["model"]["feature_dim"],
                              output_dim=cfg["model"]["feature_dim"],
                              hidden_sizes=cfg["model"]["decoder_hidden"],
                              activation=cfg["model"]["activation"]).to(device)
        real = Realization(**cfg["ssm"]["realization"])
        # h = int(cfg["ssm"]["realization"]["past_horizon"])
        obs  = CMEObservation(**cfg["ssm"]["observation"])
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=cfg["training"]["lr"]
        )

        sv_history = []
        # 学習ループ
        for epoch in range(args.epochs):
            encoder.train(); decoder.train()
            for step, (Y_batch, _ ) in enumerate(train_loader):
                Yb = Y_batch.to(device)
                # print(f'Yb.shape ; {Yb.shape}')
                T = Yb.size(0)
                Yf = encoder(Yb)
                # print(f'Yf.shape ; {Yf.shape}')
                if T - 2 * real.h + 2 <= 0:
                    print(f'too short for realization: batchsize={T}; h={real.h}')
                    continue
                else:
                    real.fit(Yf)
                sv_history.append(real._L_vals.detach().cpu().numpy())
                Xs = real.filter(Yf)
                obs.fit(Xs, Yf)
                Yp = obs.decode()
                Y_pred = decoder(Yp)
                Yb_eff = Yb[real.h+1 : T-real.h + 2, :]
                # if epoch == 0:
                #     print(f'shape of Xr : {Xr.shape}')
                #     print(f'shape of Yb_dff : {Yb_eff.shape}')
                    # print(f'first sample of Xr : {Xr[0,:]}')
                    # print(f'first sample of Yb_eff : {Yb_eff[0,:]}')
                mse = torch.norm(Y_pred - Yb_eff, dim=1).mean()
                reg = real.singular_value_reg(sv_weight)
                loss = mse + reg
                
                #debag_ログ書き込み
                # print(f"[L={L}][epoch={epoch}][step={step}] loss={loss.item():.6f}")
                log_file.write(
                    f"{L},{epoch},{step},{loss.item():.6f},"
                    f"{mse.item():.6f},{reg.item():.6f}\n"
                )
                
                optimizer.zero_grad()
                loss.backward()
                
                #debag
                if _flag == 0:
                    # 学習ステップ直後に勾配ノルムをプリント
                    for name, param in encoder.named_parameters():
                        if param.grad is not None:
                            print(f"Encoder : {name} grad norm = {param.grad.norm().item():.6f}")
                    for name, param in encoder.named_parameters():
                        if param.grad is not None:
                            print(f"Decoder : {name} grad norm = {param.grad.norm().item():.6f}")
                    _flag = 1
                            
                optimizer.step()
                # #debag
                # print(f"[Epoch {epoch}] mse={mse.item():.4f}  reg={reg.item():.4f}  loss={(mse+reg).item():.4f}")

            
            # 学習過程特異値プロット
            # numpy 配列にまとめる
            singular_history = np.stack(sv_history, axis=0)  # shape: (steps, rank)
            plt.figure(figsize=(6, 4))
            for i in range(singular_history.shape[1]):
                plt.plot(singular_history[:, i], label=f"σ_{i}")
            plt.xlabel("Training step")
            plt.ylabel("Singular values")
            plt.title(f"Singular values during training (L={L})")
            plt.legend(ncol=2, fontsize="small")
            plt.grid(True)
            # y=1 の水平点線を入れる
            plt.axhline(y=1, linestyle=':', linewidth=1)
            # 保存先は先に作った sv_hst_dir
            plt.savefig(os.path.join(sv_hst_dir, f"singular_values_L{L}.png"), dpi=150)
            plt.close()

        #debag
        print(f'Y_pred : {Y_pred[0 : 4, :]}')
        print(f'Yb_eff : {Yb_eff[0 : 4, :]}\n\n')


        # 訓練データでの評価
        tm, tr = evaluate_model(
            train_loader, encoder, decoder, real, obs,
            sv_weight, device
        )
        # テストデータでの評価
        te_m, te_r = evaluate_model(
            test_loader, encoder, decoder, real, obs,
            sv_weight, device
        )

        results["lengths"].append(L)
        results["train_mse"].append(tm)
        results["train_reg"].append(tr)
        results["test_mse"].append(te_m)
        results["test_reg"].append(te_r)
        results["diff_abs"].append(abs(te_m - (tm + tr)))


    #debag_ログファイルを閉じる
    log_file.close()
    
    # コマンドライン引数を text ファイルに保存
    args_dict = vars(args)
    with open(os.path.join(outdir, "run_args.txt"), "w") as f:
        for key, val in args_dict.items():
            f.write(f"{key}: {val}\n")

    for key in results:
        plt.figure()
        plt.plot(results["lengths"], results[key], marker="o")
        plt.xlabel("Sequence Length")
        plt.ylabel(key)
        plt.title(f"{key} vs Length")
        plt.grid(True)
        plt.savefig(os.path.join(outdir, f"{key}.png"))
        plt.close()


if __name__ == "__main__":
    main()
