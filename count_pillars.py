import sys
import os

# Dynamically add the virtual environment's site-packages to sys.path
venv_site_packages = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Lib\site-packages"
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)
else:
    print(f"Warning: venv path not found: {venv_site_packages}")

import pandas as pd

target_dir = r"G:\マイドライブ\実験データ\1.plasmon_analyzer"
files = ['results.csv', 'results_min1.csv', 'ground_truth.csv']

for f in files:
    path = os.path.join(target_dir, f)
    if os.path.exists(path):
        print(f"Reading {f}...")
        try:
            df = pd.read_csv(path)
            print(f"{f}: {len(df):,} pillars")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"{f} does not exist at {path}")
