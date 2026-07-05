import pandas as pd
import os

path = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti/1-2_aligned_to_pre.csv"
if os.path.exists(path):
    df = pd.read_csv(path, nrows=2)
    print("1-2_aligned_to_pre.csv columns:")
    print(df.columns.tolist())
else:
    print(f"Path does not exist: {path}")
    
pre_path = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti/1-2_pillars_pre.csv"
if os.path.exists(pre_path):
    df_pre = pd.read_csv(pre_path, nrows=2)
    print("1-2_pillars_pre.csv columns:")
    print(df_pre.columns.tolist())
else:
    print(f"Pre path does not exist: {pre_path}")
