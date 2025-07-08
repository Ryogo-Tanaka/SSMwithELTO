#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def preprocess_dataset(fname):
    """
    Reads CSV from data/preprocessed/{fname}.csv, sorts by time index,
    normalizes features, splits into train/val/test, and saves each as an .npz.
    """
    in_path = f"data/preprocessed/{fname}.csv"
    df = pd.read_csv(in_path)
    
    # Identify and parse the datetime column
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=date_col, inplace=True)
    
    # Extract feature values (drop datetime)
    features = df.drop(columns=[date_col]).values  # shape: (T, 7)
    T = features.shape[0]
    
    # Compute split indices (60% train, 20% val, 20% test)
    train_end = int(0.6 * T)
    val_end = int(0.8 * T)
    
    splits = {
        'train': features[:train_end],
        'val':   features[train_end:val_end],
        'test':  features[val_end:]
    }
    
    # Ensure output directory exists
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Save each split as an .npz file
    for split_name, arr in splits.items():
        out_path = os.path.join(out_dir, f"{fname}_{split_name}.npz")
        # UPDATED: キーワードなしで渡す → np.load(...)[ 'arr_0' ] で読めるように
        np.savez(out_path, arr)
        print(f"Saved {split_name} split for {fname}: {out_path} (shape={arr.shape})")
    
    # # Save each split as an .npz file
    # for split_name, arr in splits.items():
    #     out_path = os.path.join(out_dir, f"{fname}_{split_name}.npz")
    #     np.savez(out_path, data=arr)
    #     print(f"Saved {split_name} split for {fname}: {out_path} (shape={arr.shape})")

if __name__ == "__main__":
    for fname in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        preprocess_dataset(fname)
