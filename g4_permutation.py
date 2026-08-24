import os
import glob
import time
import json
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import argparse

# Stage 3 script focusing purely on validation
def run_g4_permutation_test(paired_csv_dir, conc_map, iterations=1000):
    print("==================================================")
    print(" STAGE 3: G-4 Label Permutation (Null Calibration)")
    print("==================================================")
    
    # Load all paired CSVs
    csv_files = glob.glob(os.path.join(paired_csv_dir, "*.csv"))
    if not csv_files:
        print("No paired CSV files found.")
        return
        
    data = []
    for f in csv_files:
        basename = os.path.basename(f)
        series = basename.split('_')[2].replace('.csv', '') # e.g. test_paired_10_1.csv -> 10-1 -> wait, name is 'test_paired_10-1.csv'
        # More robust extraction assuming format like "paired_{series}.csv"
        # Let's extract series safely
        for k in conc_map.keys():
            if k in basename:
                series = k
                break
        else:
            continue
            
        df = pd.read_csv(f)
        median_delta = df['delta_I'].median()
        trimmed_delta = scipy.stats.trim_mean(df['delta_I'], 0.1)
        
        data.append({
            'series': series,
            'true_conc': conc_map[series],
            'median_delta': median_delta,
            'trimmed_delta': trimmed_delta
        })
        
    df_agg = pd.DataFrame(data)
    
    # Remove 0 M (blank) for log-linear regression
    df_fit = df_agg[df_agg['true_conc'] > 0].copy()
    if len(df_fit) < 3:
        print("Not enough non-zero concentration points for fitting.")
        return
        
    log_c = np.log10(df_fit['true_conc'].values)
    y_true = df_fit['trimmed_delta'].values
    
    # 1. Calculate true R^2
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_c, y_true)
    true_r2 = r_value**2
    print(f"True R^2 (Trimmed Delta): {true_r2:.4f}")
    
    # 2. Monte Carlo Permutation
    null_r2s = []
    np.random.seed(42) # For reproducibility
    for _ in range(iterations):
        y_shuffled = np.random.permutation(y_true)
        _, _, r, _, _ = scipy.stats.linregress(log_c, y_shuffled)
        null_r2s.append(r**2)
        
    null_r2s = np.array(null_r2s)
    p_95 = np.percentile(null_r2s, 95)
    
    # 3. Calculate p-value
    # How many null R^2 are >= true R^2?
    p_val = np.sum(null_r2s >= true_r2) / iterations
    
    print("\n--- Null Calibration Results ---")
    print(f"Null R^2 Median     : {np.median(null_r2s):.4f}")
    print(f"Null R^2 95th Perc. : {p_95:.4f}")
    print(f"True R^2            : {true_r2:.4f}")
    print(f"Permutation p-value : {p_val:.4f}")
    
    if true_r2 > p_95:
        print(">>> RESULT: SIGNIFICANT (Signal is stronger than noise)")
    else:
        print(">>> RESULT: NOT SIGNIFICANT (Cannot distinguish from noise)")
        
    # 4. Plot Null Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(null_r2s, bins=30, alpha=0.7, color='gray', density=True, label='Null R² Distribution')
    plt.axvline(np.median(null_r2s), color='black', linestyle='--', label=f'Median: {np.median(null_r2s):.2f}')
    plt.axvline(p_95, color='red', linestyle='--', label=f'95% Limit: {p_95:.2f}')
    plt.axvline(true_r2, color='blue', linewidth=2, label=f'True R²: {true_r2:.2f}')
    
    plt.title("G-4 Label Permutation Test (Null Calibration)")
    plt.xlabel("R²")
    plt.ylabel("Density")
    plt.legend()
    
    out_png = os.path.join(paired_csv_dir, "G4_permutation_test.png")
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved permutation plot to {out_png}")

if __name__ == "__main__":
    # Test dictionary mapping series to concentration
    test_map = {
        '1-1': 1e-9,
        '1-2': 1e-10,
        '1-3': 1e-11,
        '1-4': 1e-12,
        '1-5': 1e-13,
        '1-6': 1e-14,
        '1-7': 1e-15,
        '1-8': 0.0
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing paired CSVs")
    args = parser.parse_args()
    
    run_g4_permutation_test(args.dir, test_map)
