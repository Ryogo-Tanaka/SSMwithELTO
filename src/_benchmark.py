#!/usr/bin/env python3
import argparse
import yaml
import json
import os
import importlib                     # 動的インポート用
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt, csv, os
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss, acf
import datetime as dt

from utils.gpu_utils import select_device  # GPU 選択を外部ユーティリティに委譲
from ssm.realization import Realization, RealizationError
from ssm.observation import CMEObservation


def get_class(class_path: str):
    module_name, cls_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def build_loaders(Y: np.ndarray, seq_len: int, batch_size: int, stride: int = 1, shuffle: bool = False):
    """
    Y (T, d) から長さ seq_len の部分時系列を
    stride (= スライド幅) で重複許可して切り出し、
    DataLoader を返す。

    Args:
      Y:          NumPy array, shape (T, d)
      seq_len:    部分時系列の長さ L
      batch_size: ミニバッチサイズ
      stride:     間隔ステップ。1 なら完全重複スライド
      shuffle:    True なら各エポックでランダムにサンプルを抽出
    Returns:
      DataLoader yielding (batch_Y, batch_Y), with batch_Y.shape == (B, L, d)
    """
    # NumPy→Tensor
    tensor = torch.tensor(Y, dtype=torch.float32)       # (T, d)

    # unfold で (num_windows, L, d) のテンソルを作る
    #   dimension=0 を滑らせ、window=seq_len, step=stride
    windows = tensor.unfold(0, seq_len, stride)        # → shape (N, L, d)
    windows = windows.contiguous()                     # メモリを連続化
    if windows.size(1) == Y.shape[1] and windows.size(2) == seq_len:  # bug?
        windows = windows.permute(0, 2, 1).contiguous()
        # print(f"    permuted to correct shape: {tuple(windows.shape)}")

    # DataLoader 用の Dataset
    ds = TensorDataset(windows, windows)               # (入力, 教師) を同じに

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def plot_results(result_dir):
    """
    train_log.csv と benchmark_results.json の sv_history から
    ① loss 推移
    ② MSE 推移
    ③ 各特異値推移 (+ y=1 に点線グリッド)
    をプロットし、PNG で保存する。
    """
    import csv, json

    # --- ①② train_log.csv の読み込み ---
    log_path = os.path.join(result_dir, 'train_log.csv')
    epochs = []
    steps  = []
    loss_list = []
    mse_list  = []
    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            steps.append(int(row['step']))
            loss_list.append(float(row['loss']))
            mse_list.append(float(row['mse']))

    # ① loss 推移
    plt.figure()                                                    # ディスティンクトプロット
    plt.plot(loss_list)                                             # 色指定せずデフォルト
    plt.xlabel('Iteration')  
    plt.ylabel('Loss')  
    plt.title('Training Loss')  
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))         # 保存
    plt.close()

    # ② MSE 推移
    plt.figure()
    plt.plot(mse_list)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Training MSE')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'mse_curve.png'))
    plt.close()

    # --- ③ benchmark_results.json の sv_history 読み込み ---
    json_path = os.path.join(result_dir, 'benchmark_results.json')
    with open(json_path) as f:
        all_results = json.load(f)
    # 単一 config キーを取る
    key = next(iter(all_results))
    sv_hist = all_results[key]['sv_history']                        # list of list
    sv_arr = np.array(sv_hist)                                      # shape (n_steps, n_svs)

    # 各特異値の推移
    plt.figure()
    for i in range(sv_arr.shape[1]):
        plt.plot(sv_arr[:, i])                                      # 各列をプロット
    # y=1 に点線グリッド
    plt.axhline(1.0, linestyle='--')                                # 水平点線
    plt.grid(axis='y', linestyle=':')                               # y 軸方向に点線グリッド
    plt.xlabel('Iteration')
    plt.ylabel('Singular values')
    plt.title('Singular value trajectories')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'sv_history.png'))
    plt.close()
    print("func. plot_results ended.")


def plot_and_save_state_variables(X: torch.Tensor, save_dir: str):
    """
    X: torch.Tensor of shape (T, d)
    save_dir: 既に存在する親ディレクトリのパス
    → save_dir/states/ 以下に図を保存します
    """
    # NumPy array に変換
    X_np = X.detach().cpu().numpy()
    T, d = X_np.shape

    # 保存先ディレクトリを作成
    states_dir = os.path.join(save_dir, 'states')
    os.makedirs(states_dir, exist_ok=True)

    # 時刻軸
    time_full    = np.arange(T)
    t10          = T // 10
    time_partial = np.arange(t10)

    for dim in range(d):
        # ① 全期間プロット
        fig = plt.figure(figsize=(8, 3))
        plt.plot(time_full, X_np[:, dim])
        plt.title(f'Variable {dim} — Full Horizon (0 to {T-1})')
        plt.xlabel('Time step')
        plt.ylabel(f'X[:, {dim}]')
        plt.grid(True)
        plt.tight_layout()

        # ファイルに保存
        fname_full = os.path.join(states_dir, f'dim_{dim:02d}_full.png')
        fig.savefig(fname_full)
        plt.close(fig)  # メモリ解放

        # ② 最初の T/10 プロット
        fig = plt.figure(figsize=(8, 3))
        plt.plot(time_partial, X_np[:t10, dim])
        plt.title(f'Variable {dim} — First {t10} Steps (0 to {t10-1})')
        plt.xlabel('Time step')
        plt.ylabel(f'X[:, {dim}]')
        plt.grid(True)
        plt.tight_layout()

        # ファイルに保存
        fname_partial = os.path.join(states_dir, f'dim_{dim:02d}_partial.png')
        fig.savefig(fname_partial)
        plt.close(fig)

    print(f'Saved all plots to {states_dir}/')


def diagnose_time_series(Y: torch.Tensor,
                         result_dir: str | Path,
                         win_roll: int = 200,
                         L_list  = (64,128,256,512),
                         max_lag: int = 100):
    """
    Y : (T,p)  CPU tensor (中心化済み推奨)
    result_dir : 実験ごとの出力フォルダ
    """
    # ----------------- フォルダ準備 ------------------------
    out_dir = Path(result_dir).expanduser().resolve() / "diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"# Diagnose generated {dt.datetime.now().isoformat()}")
    T, p = Y.shape
    log(f"Series length T={T},  dims p={p}")

    # 1. Running mean -------------------------------------
    t_axis = torch.arange(1, T+1)
    cum_sum = torch.cumsum(Y, dim=0) / t_axis.view(-1,1)

    plt.figure(figsize=(6,3))
    for j in range(p):
        plt.plot(cum_sum[:,j], label=f"dim{j}")
    for j in range(p):
        plt.axhline(Y[:,j].mean().item(), ls='--', lw=0.7)
    plt.title("Running mean"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"running_mean.png", dpi=200)
    plt.close()
    log("Saved running_mean.png")

    # 2. RMS of segment means ------------------------------
    rms_rows=[]
    for L in L_list:
        M = T//L
        seg_mean = Y[:M*L].view(M, L, p).mean(1)
        rms = ((seg_mean - Y.mean(0)).pow(2).sum(1)).sqrt().mean().item()
        rms_rows.append((L, rms))
        log(f"RMS segment mean (L={L}) = {rms:.4f}")

    with open(out_dir/"rms_segment.csv","w",newline="") as f:
        csv.writer(f).writerows([("L","RMS")]+rms_rows)
    log("Saved rms_segment.csv")

    # 3. Rolling mean/var (dim0示例) ------------------------
    roll_mu = torch.nn.functional.avg_pool1d(Y.T.unsqueeze(0), win_roll, stride=1)[0].T
    roll_var= torch.nn.functional.avg_pool1d((Y**2).T.unsqueeze(0), win_roll, stride=1)[0].T - roll_mu**2
    plt.figure(figsize=(6,3))
    plt.plot(roll_mu[:,0], label='μ(t)'); plt.plot(roll_var[:,0], label='σ²(t)')
    plt.title(f"Rolling (w={win_roll}) dim0"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"rolling_mean_var.png", dpi=200); plt.close()
    log(f"Saved rolling_mean_var.png (w={win_roll})")

    # 4. Block covariance Frobenius -----------------------
    M_blocks = 5
    Lb = T//M_blocks
    full_cov = torch.cov(Y.T)
    frob = 0.
    for m in range(M_blocks):
        cov_m = torch.cov(Y[m*Lb:(m+1)*Lb].T)
        frob += torch.norm(cov_m - full_cov, p='fro')
    frob /= M_blocks
    rel_frob = frob / torch.norm(full_cov, p='fro')
    log(f"Avg Frobenius diff (M={M_blocks}) = {frob:.4f}  (relative {rel_frob:.3%})")

    # 5. ADF / KPSS (dim0) --------------------------------
    Y_np = Y.numpy()
    adf_p = adfuller(Y_np[:,0])[1]
    try:
        kpss_p = kpss(Y_np[:,0], nlags="auto")[1]
    except ValueError:
        kpss_p = float('nan')   # ショートシリーズで失敗することがある
    log(f"ADF  p={adf_p:.4f}  (p<0.05 ⇒ 定常寄り)")
    log(f"KPSS p={kpss_p:.4f} (p>0.05 ⇒ 定常寄り)")

    # 6. ACF & τ_int --------------------------------------
    ac = acf(Y_np[:,0], nlags=max_lag, fft=True)
    tau_int = 1 + 2*ac[1:].sum()
    log(f"Integrated autocorr time τ_int≈{tau_int:.1f} (max_lag={max_lag})")
    plt.figure(figsize=(6,3))
    plt.stem(range(max_lag+1), ac) ; plt.title("ACF dim0")
    plt.tight_layout(); plt.savefig(out_dir/"acf_dim0.png", dpi=200); plt.close()
    log("Saved acf_dim0.png")

    # --------------- ログを txt に保存 --------------------
    with open(out_dir/"diagnose.txt","w") as f:
        f.write("\n".join(log_lines))
    print(f"[diagnose] results written under {out_dir}")


def train_one_sequence(Y_train: np.ndarray, cfg: dict, device: torch.device, result_dir: str):
    """
    与えられた時系列 Y_train でモデルを訓練し、
    (encoder, realization, observation, decoder, sv_history) を返却する
    """
    # set dataloader
    seq_len = cfg['train']['seq_len']
    stride = cfg['train'].get('stride', 1)
    batch_size = cfg['train'].get('batchsize', 1)
    shuffle = cfg['train']['shuffle']
    train_loader = build_loaders(Y_train, seq_len, batch_size, stride, shuffle)
    # tensor = torch.tensor(Y_train, dtype=torch.float32)
    # ds_train = TensorDataset(tensor)
    # train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)

    # model initialization
    from models.encoder import build_encoder
    from models.decoder import build_decoder

    encoder = build_encoder(cfg['model']['encoder']).to(device)
    decoder = build_decoder(cfg['model']['decoder']).to(device)
    real = Realization(**cfg['ssm']['realization'])
    obs = CMEObservation(**cfg['ssm']['observation'])
    
    # EncoderClass = get_class(cfg['model']['encoder_class'])
    # DecoderClass = get_class(cfg['model']['decoder_class'])
    # encoder = EncoderClass(**cfg['model']['encoder']).to(device)
    # decoder = DecoderClass(**cfg['model']['decoder']).to(device)
    # real = Realization(**cfg['ssm']['realization'])
    # obs = CMEObservation(**cfg['ssm']['observation'])

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=cfg['train']['lr']
    )

    # --- ログ準備 ---
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'train_log.csv')  # UPDATED: 保存先を result_dir 配下に
    sv_history = []
    with open(log_path, 'w') as log_file:
        log_file.write('epoch,step,loss,mse,reg\n')
        epochs = cfg['train']['epochs']
        sv_weight = cfg['train']['sv_weight']

        # --- 訓練ループ ---
        epochs = cfg['train']['epochs']
        max_step = cfg['train']['max_step']
        sv_weight = cfg['train']['sv_weight']
        for epoch in range(epochs):
            _err_count = 0
            encoder.train()
            decoder.train()
            last_loss = None
            last_mse  = None
            last_reg  = None
            for step, (Y_batch, _) in enumerate(train_loader):
                if step >= max_step:
                    break
                elif _err_count >= 40:
                    break
                else:
                    # Y_batch: (batch_size, seq_len, d)
                    for Yb in Y_batch:
                        Yb = Yb.to(device)  # (T, d)
                        # print(f'Yb:{Yb.shape}')
                        T = Yb.size(0)
                        Yf = encoder(Yb)  # (T, p)
            
                        if T - 2 * real.h + 2 <= 0:
                            print(f"too short for realization: batchsize={T}; h={real.h}")
                            
                            continue
                        try:
                            real.fit(Yf)
                        except Exception as e:
                            print(f"[Warning] real.fit failed at epoch={epoch}, step={step}: {e}")
                            _err_count += 2

                            #for debug
                            debug_dir = os.path.join(result_dir, "debug")
                            os.makedirs(debug_dir, exist_ok=True)
                            # H_debug = real.H.detach().cpu().numpy()
                            # npy_path = os.path.join(debug_dir, f"H_epoch{epoch}_step{step}.npy")
                            # np.save(npy_path, H_debug)
                            # csv_path = os.path.join(debug_dir, f"H_debug_epoch{epoch}_step{step}.csv")
                            # np.savetxt(csv_path, H_debug, delimiter=',',
                            #                header=','.join([f"col{i}" for i in range(H_debug.shape[1])]),comments='')
                            # _evls = real._Spp_eigvals
                            # if _evls.ndim == 1:
                            #     _evls = _evls.reshape(-1, 1)
                            # S_pp_eigvals = _evls.detach().cpu().numpy()
                            # csv_path_evls = os.path.join(debug_dir, f"Spp_evls_epoch{epoch}_step{step}.csv")
                            # np.savetxt(csv_path_evls, S_pp_eigvals, delimiter=',',
                            #                header=','.join([f"col{i}" for i in range(H_debug.shape[1])]),comments='')
                            continue

                        # # for debug
                        debug_dir = os.path.join(result_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        # # npy_path = os.path.join(debug_dir, f"H_epoch{epoch}_step{step}.npy")
                        # H_debug = real.H.detach().cpu().numpy()
                        # # np.save(npy_path, H_debug)
                        # csv_path = os.path.join(debug_dir, f"H_debug_epoch{epoch}_step{step}.csv")
                        # np.savetxt(csv_path, H_debug, delimiter=',',
                        #                header=','.join([f"col{i}" for i in range(H_debug.shape[1])]),comments='')
                        # _evls = real._Spp_eigvals
                        # if _evls.ndim == 1:
                        #     _evls = _evls.reshape(-1, 1)
                        # S_pp_eigvals = _evls.detach().cpu().numpy()
                        # csv_path_evls = os.path.join(debug_dir, f"Spp_evls_epoch{epoch}_step{step}.csv")
                        # np.savetxt(csv_path_evls, S_pp_eigvals, delimiter=',',
                        #                header=','.join([f"col{i}" for i in range(S_pp_eigvals.shape[1])]),comments='')
                        
                        sv_history.append(real._L_vals.detach().cpu().numpy())
            
                        Xs = real.filter(Yf)                   # 状態系列
                        obs.fit(Xs, Yf)                        # 不変観測モデル訓練
                        # Yp = obs.decode()                      # 一括特徴予測
                        try:
                            Yp = obs.decode() 
                        except Exception as e:
                            print(f"[Warning] obs.decode failed at epoch={epoch}, step={step}: {e}")
                            
                            # # debug
                            _evls_obs = obs.eigvals
                            if _evls_obs.ndim == 1:
                                _evls_obs = _evls_obs.reshape(-1, 1)
                            K_eigvals = _evls_obs.detach().cpu().numpy()
                            csv_path_evls_obs = os.path.join(debug_dir, f"K_evls_epoch{epoch}_step{step}.csv")
                            np.savetxt(csv_path_evls_obs, K_eigvals, delimiter=',',
                                           header=','.join([f"col{i}" for i in range(K_eigvals.shape[1])]),comments='')
                            _err_count += 2

                            # print(obs.eigvals)
                            # K_np = obs.K_past.cpu().detach().numpy()
                            # debug_dir = os.path.join(result_dir, "debug")
                            # os.makedirs(debug_dir, exist_ok=True)
                            # npy_path = os.path.join(debug_dir, f"K_past_epoch{epoch}_step{step}.npy")
                            # np.save(npy_path, K_np)
                            
                            continue

                        # #debug
                        # _evls_obs = obs.eigvals
                        # if _evls_obs.ndim == 1:
                        #     _evls_obs = _evls_obs.reshape(-1, 1)
                        # K_eigvals = _evls_obs.detach().cpu().numpy()
                        # csv_path_evls_obs = os.path.join(debug_dir, f"K_evls_epoch{epoch}_step{step}.csv")
                        # np.savetxt(csv_path_evls_obs, K_eigvals, delimiter=',',
                        #                header=','.join([f"col{i}" for i in range(K_eigvals.shape[1])]),comments='')
                        
                        Y_pred = decoder(Yp)                   # 出力空間に変換
            
                        Yb_eff = Yb[real.h+1 : T-real.h+2, :]
                        diff = Y_pred - Yb_eff
                        time_dims = tuple(range(1, diff.dim()))
                        norms = torch.norm(diff, p='fro', dim=time_dims)  # (forecast_steps,)
                        mse  = norms.mean()
                        # mse = torch.norm(Y_pred - Yb_eff, dim=1).mean()
                        reg = real.singular_value_reg(sv_weight)
                        loss = mse + reg
            
                        # UPDATED: stepごとのログ書き込み
                        log_file.write(f"{epoch},{step},{loss.item():.6f},{mse.item():.6f},{reg.item():.6f}\n")
            
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        last_loss = loss
                        last_mse = mse
                        last_reg = reg

                #todo: early stopping
            if last_loss is not None:
                print(f"[Epoch {epoch+1}/{epochs}] loss={loss.item():.4f}, mse={mse.item():.4f}, reg={reg.item():.4f}")
            else:
                print(f"[Epoch {epoch+1}/{epochs}] skipped all batch...")

    log_file.close()
    return encoder, real, obs, decoder, sv_history


def evaluate_validation(Y_valid: np.ndarray, components: tuple, cfg: dict, device: torch.device) -> dict:
    """
    Args:
      Y_valid: 時系列データ (T × d の NumPy 配列)
      components: (encoder, real, obs, decoder) のタプル
      cfg:        実験設定辞書（YAML から読み込んだもの）
      device:     torch.device('cuda' or 'cpu')
    Returns:
      {'mse': float, 'reg': float}
    """
    encoder, real, obs, decoder = components
    encoder.eval()
    decoder.eval()

    
    eval_cfg = cfg.get('eval', {})
    seq_len  = eval_cfg['seq_len']                            # 必須
    stride   = eval_cfg.get('stride', seq_len)                # デフォルトで非重複
    batch_size = eval_cfg['batch_size']
    
    valid_loader = build_loaders(Y_valid, seq_len, batch_size, stride, shuffle=False)
    # tensor = torch.tensor(Y_valid, dtype=torch.float32)
    # loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    metrics = {'mse': [], 'reg': []}
    horizon   = cfg['ssm']['realization']['past_horizon']       # UPDATED: horizon を cfg から参照
    sv_weight = cfg['train']['sv_weight']                  # UPDATED: sv_weight を cfg から参照

    with torch.no_grad():
        _err_count = 0
        for (Y_batch, _) in valid_loader:
            for Yb in Y_batch:
                Yb = Yb.to(device)             # (T_b, d)
                T = Yb.size(0)
                Yf = encoder(Yb)               # (T_b, p)
                try:
                    real.fit(Yf)
                except Exception as e:
                    # print(f"[Warning] real.fit failed in validation: {e}")
                    _err_count += 1
                    if _err_count >=30:
                        print("real.fit failed in validation")
                        return {
                            'mse': float('nan'),
                            'reg': float('nan')
                        }
                    continue
                Xs = real.filter(Yf)           # (T_b-2h+1, r)
                obs.fit(Xs, Yf)
                Yp = obs.decode()              # (T_b-2h+1, p)
                Y_pred = decoder(Yp)           # (T_b-2h+1, d)
    
                # 時間合わせのスライス
                Yb_eff = Yb[real.h+1 : T-real.h+2, :]  # (T_b-2h+1, d)
    
                # 損失計算
                diff = Y_pred - Yb_eff
                time_dims = tuple(range(1, diff.dim()))
                norms = torch.norm(diff, p='fro', dim=time_dims)  # (forecast_steps,)
                mse  = norms.mean().item()
                # mse = torch.mean((Y_pred - Yb_eff)**2).item()              # UPDATED: 平均二乗誤差
                reg = real.singular_value_reg(sv_weight).item()            # UPDATED: 正則化項
    
                metrics['mse'].append(mse)
                metrics['reg'].append(reg)

    return {
        'mse': float(np.mean(metrics['mse'])),
        'reg': float(np.mean(metrics['reg']))
    }  #todo: バッチ部分書き換え。

def evaluate_forecast(Y_pred: np.ndarray, components: tuple, cfg: dict, device: torch.device) -> dict:
    """
    Args:
      Y_pred:      時系列データ (T × d の NumPy 配列)
      components:      (encoder, real, obs, decoder) のタプル
      cfg:             実験設定辞書（YAML から読み込んだもの）
      device:          torch.device('cuda' or 'cpu')
    Returns:
      {'mse_forecast': float, 'forecast_steps': int}
    """

    encoder, real, obs, decoder = components
    encoder.eval()
    decoder.eval()

    # load config
    horizon       = cfg['ssm']['realization']['past_horizon']
    sv_weight     = cfg['train']['sv_weight']
    pred_cfg      = cfg.get('pred', {})
    warmup        = pred_cfg.get('warmup', horizon + 1)
    forecast_steps = pred_cfg.get('forecast_steps', None)

    n_test = Y_pred.shape[0]
    # --- デフォルト or clamp for forecast_steps ---
    max_steps = max(0, n_test - warmup)
    if forecast_steps is None:
        forecast_steps = max_steps
    else:
        forecast_steps = min(forecast_steps, max_steps)

    # errors = []
    with torch.no_grad():
        # --- 初期状態推定 ---
        Y_init = torch.tensor(
            Y_pred[:warmup],
            dtype=torch.float32,
            device=device
        )
        # print(f'debag: Y_init.shape:{Y_init.shape}')
        # T = Y_init.size(0)
        Yf_init = encoder(Y_init)  # (init_len, p)
        # print(f'debag: Yf_init.shape:{Yf_init.shape}')
        try:
            real.fit(Yf_init)
        except Exception as e:
            print(f"[Warning] real.fit failed in test: {e}")
            return {
                'mse_forecast': float('nan'),
                'forecast_steps': 0
            }
        Xs_init = real.filter(Yf_init)
        obs.fit(Xs_init, Yf_init)

        # --- 逐次予測 ---
        preds = []
        for _ in range(forecast_steps + obs.h - 1):
            yf_rec = obs.decode_pred()                             # (p,)
            y_pred = decoder(yf_rec.unsqueeze(0)).squeeze(0)       # (d,)
            preds.append(y_pred)

        # --- 真値と比較 ---
        preds_tensor = torch.stack(preds, dim=0)
        Y_true_tensor = torch.tensor(
            Y_pred[warmup : warmup + forecast_steps],
            dtype = preds_tensor.dtype,
            device = preds_tensor.device
        )
        preds_tensor = preds_tensor[obs.h-1:, :]
        diff = preds_tensor - Y_true_tensor
        # --- Frobenius norm over all non-batch dims, then average over time ---
        time_dims = tuple(range(1, diff.dim()))
        norms         = torch.norm(diff, p='fro', dim=time_dims)  # (forecast_steps,)
        mse_forecast  = norms.mean().item()
        
        # if diff.dim() == 2:
        #     mse_forecast = diff.norm(p=2, dim=1).mean().item()
        # elif diff.dim() >= 3:
        #     #Frobenious norm
        #     _flatten = diff.view(diff.size(0), -1)
        #     mse_forecast = _flatten.norm(p=2, dim=1).mean().item()

    return {
        'mse_forecast': float(mse_forecast),
        'forecast_steps': forecast_steps
    }

def run_benchmark(config_path, data_dir, device, result_dir):
    """
    設定ファイルでベンチマーク実験を実行し、
    結果とログは result_dir 以下に保存
    """
    os.makedirs(result_dir, exist_ok=True)
    cfg_name = os.path.basename(config_path)                     # ex. 'ETTh_benchmark.yaml'
    snapshot_name = os.path.splitext(cfg_name)[0] + '_snapshot.txt'
    snapshot_path = os.path.join(result_dir, snapshot_name)
    with open(config_path, 'r') as fin, open(snapshot_path, 'w') as fout:
        fout.write(fin.read())
    
    all_results = {}
    
    cfg = yaml.safe_load(open(config_path))
    train_path = os.path.join(data_dir, cfg['data']['train_file'])
    val_path   = os.path.join(data_dir, cfg['data']['val_file'])
    pred_path  = os.path.join(data_dir, cfg['data']['pred_file'])

    Y_train = np.load(train_path)['arr_0']
    # print("DEBUG: Y_train.shape =", Y_train.shape)
    # print(f"DEBUG: Y_train[:4] ={Y_train[:4]}")
    Y_val   = np.load(val_path)['arr_0']
    Y_pred  = np.load(pred_path)['arr_0']

    enc, real, obs, dec, sv_hist = train_one_sequence(
        Y_train, cfg, device, result_dir
    )

    plot_and_save_state_variables(real.X_state_torch, save_dir = result_dir)
    print('trained state variables plot ended!')

    
    # debug: diagnose ergodicity
    _encoder = enc
    _encoder.eval()
    Y_val_torch = torch.from_numpy(Y_val).float().to(device)
    _Yf = _encoder(Y_val_torch)
    _Yf_c = _Yf.detach().cpu()
    diagnose_time_series(_Yf_c, result_dir= result_dir,
                     win_roll=200,
                     L_list=(64,128,256,512),
                     max_lag=100)
    
    
    val_metrics = evaluate_validation(Y_val, (enc, real, obs, dec), cfg, device)
    pred_metrics = evaluate_forecast(Y_pred, (enc, real, obs, dec), cfg, device)

    # NumPy→リスト変換してから保存 (JSON シリアライズ用)
    all_results[os.path.basename(config_path)] = {
        'sv_history': [arr.tolist() for arr in sv_hist],  # ← ここを変換
        'validation': val_metrics,                        # float や dict 内の値は OK
        'forecast': pred_metrics
    }

    

    out_path = os.path.join(result_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config ファイルのパス')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='用途別 .npz データ格納ディレクトリ')
    parser.add_argument('--device', type=str, default=None,
                        help='使用デバイス。未指定時は select_device() が自動選択')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='実験名。results/{exp-name}/ に出力')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else select_device()
    print(f"Using device: {device}")

    # 保存先ディレクトリ
    result_dir = os.path.join('results', args.exp_name)
    
    run_benchmark(args.config, args.data_dir, device, result_dir)

    plot_results(result_dir)

if __name__ == '__main__':
    main()

#呼び出し例
# python src/benchmark.py \
#   --config configs/ETTh_benchmark.yaml \
#   --data-dir data/ \
#   --device cuda \
#   --exp-name ETTh_experiment

