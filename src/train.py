# src/train.py

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import build_loaders            # data_loader.py にて npz→DataLoader を統一
from models.encoder import MLPEncoder
from models.decoder import MLPDecoder
from ssm.realization import Realization
from ssm.observation import CMEObservation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,   required=True)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--sv-weight',  type=str,   default='fixed_abs')   # UPDATED: eval_extend.py と同じキーを受け取る
    parser.add_argument('--save-path',  type=str,   default='model.pth')  
    return parser.parse_args()

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- モデルの初期化 --- 
    encoder = MLPEncoder(**cfg['model']['encoder']).to(device)      # UPDATED: MLPEncoder を外部モジュール化
    decoder = MLPDecoder(**cfg['model']['decoder']).to(device)      # UPDATED: MLPDecoder を外部モジュール化
    real    = Realization(**cfg['ssm']['realization']).to(device)   # UPDATED: Realization の fit/filter メソッドを呼べるように
    obs     = CMEObservation(**cfg['ssm']['observation']).to(device)# UPDATED: CMEObservation の fit/decode メソッドを呼べるように

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(real.parameters()) +
        list(obs.parameters()),
        lr=args.lr
    )

    # --- データ読み込み と シーケンス長ループ ---
    Y_full = np.load(cfg['data']['file'])['arr_0']  # UPDATED: npz のキーを 'arr_0' と仮定
    for L in range(cfg['train']['min_length'],
                   cfg['train']['max_length'] + 1,
                   cfg['train']['step']):
        # CHANGED: eval_extend.py と同じ batch_size 計算式
        batch_size = int(cfg['train']['train_ratio'] * L // 3)
        train_loader, _ = build_loaders(
            Y_full[:L],
            cfg['train']['train_ratio'],
            batch_size=batch_size,
            seed=cfg['train']['seed']
        )

        # --- 訓練ループ --- 
        for epoch in range(args.epochs):
            for Yb, _ in train_loader:
                Yb = Yb.to(device)                     # (T_b, d)
                Yf = encoder(Yb)                       # (T_b, p)

                real.fit(Yf)                           # UPDATED: 過去／未来スペクトル学習
                Xs = real.filter(Yf)                   # UPDATED: 1-step フィルタリング

                obs.fit(Xs, Yf)                        # UPDATED: 状態→特徴の写像を推定
                Yp = obs.decode()                      # UPDATED: 予測特徴系列を取得
                Y_pred = decoder(Yp)                   # (T_b-2h+2, p) にマッピング

                # UPDATED: eval_extend.py と同じ時間合わせスライス
                h = cfg['ssm']['realization']['horizon']
                Yb_eff = Yb[h+1 : Yb.size(0)-h+1, :]

                mse = torch.mean((Y_pred - Yb_eff)**2)  # MSE
                reg = real.singular_value_reg(cfg['sv-weight'])  # UPDATED: 特異値正則化

                loss = mse + reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # UPDATED: Epoch ごとのプログレス表示
            print(f"[L={L}] Epoch {epoch+1}/{args.epochs} → "
                  f"loss={loss.item():.4f}, mse={mse.item():.4f}, reg={reg.item():.4f}")

    # --- モデル保存 ---
    torch.save({
        'encoder':     encoder.state_dict(),
        'decoder':     decoder.state_dict(),
        'realization': real.state_dict(),
        'observation': obs.state_dict()
    }, args.save_path)
    print(f"モデルを保存しました: {args.save_path}")

if __name__ == '__main__':
    main()



# # src/train.py

# import os
# import argparse
# import yaml
# import torch
# import numpy as np
# from types import SimpleNamespace
# from torch.optim import Adam
# from torch.nn import MSELoss

# # データローダー
# from src.data_loader import build_dataloaders

# # モデルファクトリ
# from src.models.encoder import build_encoder
# from src.models.decoder import build_decoder
# from src.ssm.realization import build_realization
# from src.ssm.observation import CMEObservation


# def load_config(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def main():
#     #――― 引数パース ―――
#     parser = argparse.ArgumentParser(description="Train SSMwithELTO model")
#     parser.add_argument(
#         "--config", "-c", type=str, default="configs/demo.yaml",
#         help="Path to training config (YAML)")
#     args = parser.parse_args()
#     cfg = load_config(args.config)

#     #――― 乱数シード & デバイス設定 ―――
#     seed = cfg.get("seed", 42)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     #――― ディレクトリ準備 ―――
#     out_dir = cfg["visualization"]["output_dir"]
#     os.makedirs(out_dir, exist_ok=True)
#     ckpt_dir = cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     #――― データローダー ―――
#     # train / val / test が必要なら split を変えて３回呼び出しても良い
#     train_loader = build_dataloaders(
#         file_path=cfg["training"]["data_path"],
#         batch_size=cfg["training"]["batch_size"],
#         split="train",
#         train_ratio=cfg["training"].get("train_ratio", 0.7),
#         val_ratio=cfg["training"].get("val_ratio", 0.15),
#         test_ratio=cfg["training"].get("test_ratio", 0.15),
#         seed=seed,
#         num_workers=cfg["training"].get("num_workers", 4),
#         pin_memory=cfg["training"].get("pin_memory", True),
#     )

#     #――― モデル構築 ―――
#     # 1) Encoder d → p
#     from types import SimpleNamespace
#     enc_cfg = SimpleNamespace(**cfg["model"]["encoder"])
#     encoder = build_encoder(enc_cfg).to(device)
    
#     # 2) Decoder y→d
#     dec_cfg = SimpleNamespace(**cfg["model"]["decoder_y2d"])
#     decoder = build_decoder(dec_cfg).to(device)

#     # 3) SSM Realization
#     ssm_cfg = SimpleNamespace(**cfg["ssm"]["realization"])
#     real = build_realization(ssm_cfg)

#     # 4) CME Observation
#     obs_cfg = cfg["ssm"]["observation"]
#     observation = CMEObservation(
#         kernel=obs_cfg.get("kernel", "rbf"),
#         gamma=obs_cfg.get("gamma", None),
#         reg_lambda=obs_cfg.get("reg_lambda", 1e-3),
#         approx=obs_cfg.get("approx", False),
#         approx_rank=obs_cfg.get("approx_rank", None),
#     )

#     #――― 最適化 & 損失 ―――
#     lr = cfg["training"]["lr"]
#     optimizer = Adam(
#         list(encoder.parameters()) + list(decoder.parameters()),
#         lr=lr
#     )
#     mse_loss = MSELoss()

#     # ハイパーパラメータ
#     epochs       = cfg["training"]["epochs"]
#     svd_weight   = cfg["training"]["svd_weight"]
#     # svd_reg_type = cfg["training"]["svd_reg_type"]
#     h            = cfg["ssm"]["realization"]["h"]

#     #――― 学習ループ ―――
#     #todo
#     for epoch in range(1, epochs + 1):
#         encoder.train()
#         decoder.train()
#         epoch_loss = 0.0

#         for X_batch, _ in train_loader:
#             # X_batch: Tensor[batch_size, d]  あるいは [T, d] depending on Dataset
#             X = X_batch.to(device)
            
#             # 1) Encoder
#             Y = encoder(X)

#             # 2) Realization: スキップ処理を追加
#             #    シーケンス長 T が 2*h より小さいと N=T-2h+1≤0 になるので除外
#             T = Y.size(0)
#             if T <= 2 * real.h:
#                 # too short to build past/future Hankels → skip this batch
#                 continue
#             real.fit(Y)
#             X_state = real.filter(Y)

#             # 3) CME fit + decode → Y_cme
#             observation.fit(X_state, Y)
#             Y_cme = observation.decode(X_state)

#             # 4) Decoder → X_rec
#             X_rec = decoder(Y_cme)

#             # 5) 再構成誤差：中央部分を切り出し
#             #    X shape [T, d] の場合
#             X_eff = X[h+1 : T-h + 1, :]
#             loss_rec = mse_loss(X_rec, X_eff)

#             # 6) 特異値和による正則化
#             loss_svd = real.singular_value_reg()
#             loss = loss_rec + svd_weight * loss_svd

#             # 7) バックプロパゲーション
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(train_loader)
#         print(f"[Epoch {epoch}/{epochs}]  Loss = {avg_loss:.4f}")

#         # 必要に応じて検証ループやモデル保存を追加

#     #――― モデル保存 ―――
#     torch.save({
#         "encoder": encoder.state_dict(),
#         "decoder": decoder.state_dict(),
#     }, os.path.join(ckpt_dir, "model_final.pth"))
#     print("Training complete. Models saved to", ckpt_dir)


# if __name__ == "__main__":
#     main()
