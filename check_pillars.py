import pandas as pd
import glob
import numpy as np

files = glob.glob(r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\*\intensity_summary.csv')
files += glob.glob(r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\*_foranti\intensity_summary.csv')

for f in files:
    df = pd.read_csv(f)
    if 'matched_pillars' in df.columns:
        p = df['matched_pillars'].values
        print(f"{f.split(chr(92))[-2]}: min={np.min(p)}, max={np.max(p)}, median={np.median(p)}")
