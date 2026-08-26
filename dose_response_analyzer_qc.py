# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
import cv2
import matplotlib.pyplot as plt
from qc_rules import get_position_status, get_sample_metadata

def analyze_and_plot(assay_type, exp_dir, out_png):
    # Best Params
    th = 15
    bl = 7
    pct = 99.9
    tr = 0.2
    
    exp_dir = os.path.abspath(exp_dir)
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "**", "*-0.tif"), recursive=True))
    
    image_stats = []
    
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        
        parts = base_prefix.split('-')
        if len(parts) < 2:
            continue
            
        sample_id = int(parts[0])
        pos_id = int(parts[1])
        
        # Check QC Rules
        is_valid, status_reason = get_position_status(assay_type, sample_id, pos_id)
        if not is_valid:
            print(f"Skipping {base_prefix}: {status_reason}")
            continue
            
        meta = get_sample_metadata(assay_type, sample_id)
        if not meta:
            continue
            
        true_conc = meta.get('conc', 0.0)
        run_id = meta.get('run_id', 1)
        sample_status = meta.get('status', 'INCLUDE')
        
        post_path = os.path.join(os.path.dirname(pre_path), f"{base_prefix}-1.tif")
        has_post = os.path.exists(post_path)
        
        pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        h, w = pre_img.shape
        
        if has_post:
            post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            post_warped = pre_img # Dummy
            
        top_hat = cv2.morphologyEx(pre_img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th, th)))
        blurred = cv2.GaussianBlur(top_hat, (bl, bl), 0)
        bg_local = cv2.boxFilter(blurred, -1, (21, 21))
        local_contrast = blurred - bg_local
        
        pos_contrast = local_contrast[local_contrast > 0]
        thresh_val = np.percentile(pos_contrast, pct) if len(pos_contrast) > 0 else 0.01
        
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(blurred, kernel_dil)
        peaks_mask = (blurred == dilated) & (local_contrast >= thresh_val)
        num_peaks, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
        
        if num_peaks <= 1:
            continue
            
        xi = np.clip(np.round(centroids[1:, 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(centroids[1:, 1]).astype(np.int32), 0, h - 1)
        
        pre_blur = cv2.blur(pre_img, (3, 3))
        pre_ints = pre_blur[yi, xi]
        
        if has_post:
            post_blur = cv2.blur(post_warped, (3, 3))
            post_ints = post_blur[yi, xi]
            valid = post_ints > 0
            pre_ints, post_ints = pre_ints[valid], post_ints[valid]
            if len(pre_ints) == 0: continue
            delta = post_ints - pre_ints
            trimmed_delta = scipy.stats.trim_mean(delta, tr)
        else:
            # Single image mode: Delta = Peak Intensity - Global Background Mean
            bg_mask = blurred < np.percentile(blurred, 90)
            bg_mean = np.mean(pre_img[bg_mask])
            valid = pre_ints > 0
            pre_ints = pre_ints[valid]
            if len(pre_ints) == 0: continue
            delta = pre_ints - bg_mean
            trimmed_delta = scipy.stats.trim_mean(delta, tr)
        
        image_stats.append({
            'series_id': base_prefix,
            'sample_id': sample_id,
            'pos_id': pos_id,
            'true_conc': true_conc,
            'run_id': run_id,
            'sample_status': sample_status,
            'trimmed_delta': trimmed_delta,
            'num_pillars': num_peaks - 1
        })
            
    df = pd.DataFrame(image_stats)
    if len(df) == 0:
        print("No valid data points found.")
        return
        
    csv_out = out_png.replace('.png', '.csv')
    df.to_csv(csv_out, index=False)
    print(f"Saved raw QC data to {csv_out}")
    
    # Aggregation uses MEDIAN and IQR
    def iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)
        
    df_agg = df.groupby(['true_conc', 'run_id', 'sample_status', 'sample_id']).agg(
        n_pos=('trimmed_delta', 'count'),
        median_delta=('trimmed_delta', 'median'),
        iqr_delta=('trimmed_delta', iqr),
        mean_pillars=('num_pillars', 'mean')
    ).reset_index()
    
    agg_csv = out_png.replace('.png', '_aggregated.csv')
    df_agg.to_csv(agg_csv, index=False)
    
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Plot Include points
    df_inc = df_agg[(df_agg['sample_status'] == 'INCLUDE') & (df_agg['true_conc'] > 0)]
    if len(df_inc) > 0:
        log_c = np.log10(df_inc['true_conc'].values)
        y_true = df_inc['median_delta'].values
        y_err = df_inc['iqr_delta'].values / 2.0  # approximate error bar from IQR
        
        # Fit ONLY the INCLUDE points
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_c, y_true)
        
        plt.errorbar(log_c, y_true, yerr=y_err, fmt='o', color='blue', markersize=8, capsize=4, label='Data (Median ± IQR/2)')
        
        x_line = np.linspace(min(log_c), max(log_c), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=f'Fit (R² = {r_value**2:.3f})')
        
    # Plot Flagged points
    df_flag = df_agg[(df_agg['sample_status'] == 'QC_FLAGGED') & (df_agg['true_conc'] > 0)]
    if len(df_flag) > 0:
        plt.errorbar(np.log10(df_flag['true_conc']), df_flag['median_delta'], yerr=df_flag['iqr_delta']/2.0, 
                     fmt='x', color='orange', markersize=8, capsize=4, label='QC Flagged (Excluded from fit)')
                     
    # Plot blanks
    df_blank = df_agg[df_agg['true_conc'] == 0]
    if not df_blank.empty:
        # separate by status
        blank_inc = df_blank[df_blank['sample_status'] == 'INCLUDE']
        blank_flag = df_blank[df_blank['sample_status'] == 'QC_FLAGGED']
        
        min_log_c = np.log10(df_inc['true_conc'].min()) if not df_inc.empty else -15
        blank_x = min_log_c - 1.5
        
        if not blank_inc.empty:
            plt.errorbar([blank_x]*len(blank_inc), blank_inc['median_delta'], yerr=blank_inc['iqr_delta']/2.0,
                         fmt='s', color='gray', markersize=8, capsize=4, label='Blank (INCLUDE)')
        if not blank_flag.empty:
            plt.errorbar([blank_x]*len(blank_flag), blank_flag['median_delta'], yerr=blank_flag['iqr_delta']/2.0,
                         fmt='x', color='gray', markersize=8, capsize=4, label='Blank (FLAGGED)')
                         
        plt.axvline(x=blank_x + 0.75, color='gray', linestyle=':', alpha=0.5)
        
    plt.title(f"Dose Response Curve ({assay_type}) - Strict QC Applied\nAggregated by Median & IQR", fontsize=14)
    plt.xlabel("Log10 Concentration (M)", fontsize=12)
    plt.ylabel("Δ Intensity (Median of Trimmed Means)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", type=str, choices=['SAM', 'DNA'], required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    analyze_and_plot(args.assay, args.dir, args.out)
