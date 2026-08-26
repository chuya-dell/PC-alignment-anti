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
def run_g4_permutation_test(summary_csv_path, conc_map, iterations=1000):
    print("==================================================")
    print(" STAGE 3: G-4 Label Permutation (Null Calibration)")
    print("==================================================")
    
    if not os.path.exists(summary_csv_path):
        print(f"Summary CSV not found: {summary_csv_path}")
        return
        
    df = pd.read_csv(summary_csv_path)
    
    # Extract condition series from series_id (e.g., '10-1' -> '10')
    # Actually, the user's notes say the series number (e.g., '10') maps to a concentration
    # Let's map it based on the first part of the dash. '10-1' -> '10'
    df['cond_id'] = df['series_id'].apply(lambda x: str(x).split('-')[0])
    
    data = []
    for cond_id, group in df.groupby('cond_id'):
        if cond_id not in conc_map:
            continue
            
        true_conc = conc_map[cond_id]
        
        # Aggregate the field-of-views for this condition (e.g., 10-1 through 10-8)
        # We can take the mean of the trimmed_deltas across the FOVs
        agg_trimmed = group['trimmed_delta'].mean()
        
        data.append({
            'cond_id': cond_id,
            'true_conc': true_conc,
            'trimmed_delta': agg_trimmed
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
    
    # Save plot relative to summary CSV
    out_dir = os.path.dirname(summary_csv_path)
    out_png = os.path.join(out_dir, "G4_permutation_test.png")
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved permutation plot to {out_png}")

if __name__ == "__main__":
    test_map = {
        '0': 0.0,
        '1': 1e-9,
        '2': 1e-10,
        '3': 1e-11,
        '4': 1e-12,
        '5': 1e-13,
        '8': 1e-14,
        '10': 1e-15,
        '11': 1e-11,
        '12': 1e-12,
        '13': 1e-14,
        '14': 0.0
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to image_level_summary.csv")
    args = parser.parse_args()
    
    run_g4_permutation_test(args.summary_csv, test_map)
