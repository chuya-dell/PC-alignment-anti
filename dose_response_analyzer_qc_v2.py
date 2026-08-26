# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
import cv2
import matplotlib.pyplot as plt
from qc_filter_v2 import get_sample_info, get_l3_mask

def analyze_and_plot_v2(assay_type, date_str, exp_dir, out_png, roi_dir=None):
    th, bl, pct, tr = 15, 7, 99.9, 0.2
    exp_dir = os.path.abspath(exp_dir)
    
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "**", "*-0.tif"), recursive=True))
    
    image_stats_raw = []
    image_stats_qc = []
    
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        
        parts = base_prefix.split('-')
        if len(parts) < 2: continue
            
        sample_id = int(parts[0])
        pos_id = int(parts[1])
        
        post_path = os.path.join(os.path.dirname(pre_path), f"{base_prefix}-1.tif")
        if not os.path.exists(post_path):
            print(f"Skipping {base_prefix}: No post-image found. DNA assay must wait for post-images.")
            continue
            
        # 1. Image Processing
        pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        h, w = pre_img.shape
        
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
        dx, dy = shift
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
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
        
        if num_peaks <= 1: continue
            
        xi = np.clip(np.round(centroids[1:, 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(centroids[1:, 1]).astype(np.int32), 0, h - 1)
        
        pre_blur = cv2.blur(pre_img, (3, 3))
        post_blur = cv2.blur(post_warped, (3, 3))
        
        pre_ints_full = pre_blur[yi, xi]
        post_ints_full = post_blur[yi, xi]
        valid_full = post_ints_full > 0
        
        if np.sum(valid_full) > 0:
            delta_full = post_ints_full[valid_full] - pre_ints_full[valid_full]
            trimmed_delta_raw = scipy.stats.trim_mean(delta_full, tr)
        else:
            trimmed_delta_raw = np.nan
            
        # Get Sample Info for both lists so we know the concentration
        is_valid_l1, true_conc, run_id, reason = get_sample_info(date_str, assay_type, sample_id)
        
        if not np.isnan(trimmed_delta_raw):
            image_stats_raw.append({
                'series_id': base_prefix, 'sample_id': sample_id, 'pos_id': pos_id,
                'true_conc': true_conc, 'run_id': run_id,
                'trimmed_delta': trimmed_delta_raw, 'num_valid_pillars': np.sum(valid_full)
            })
            
        # --- QC Logics ---
        if not is_valid_l1:
            print(f"QC: Skipping Sample {sample_id} (Pos {pos_id}): {reason}")
            continue
            
        # Design knife scratch band exclusion (dummy: e.g. center 100px band)
        # We will assume no scratch mask unless specified.
        scratch_mask = np.zeros((h, w), dtype=bool)
        
        valid_mask, masked_ratio = get_l3_mask(date_str, sample_id, pos_id, xi, yi, w, h, scratch_mask, roi_dir)
        
        if masked_ratio > 0.15:
            print(f"QC: Escalating to L2 Exclusion for {base_prefix}: Masked ratio {masked_ratio:.2%} > 15%")
            continue
            
        if np.sum(valid_mask) == 0:
            continue
            
        xi_qc = xi[valid_mask]
        yi_qc = yi[valid_mask]
        
        pre_ints_qc = pre_blur[yi_qc, xi_qc]
        post_ints_qc = post_blur[yi_qc, xi_qc]
        valid_qc = post_ints_qc > 0
        
        if np.sum(valid_qc) > 0:
            delta_qc = post_ints_qc[valid_qc] - pre_ints_qc[valid_qc]
            trimmed_delta_qc = scipy.stats.trim_mean(delta_qc, tr)
            
            image_stats_qc.append({
                'series_id': base_prefix, 'sample_id': sample_id, 'pos_id': pos_id,
                'true_conc': true_conc, 'run_id': run_id,
                'trimmed_delta': trimmed_delta_qc, 'num_valid_pillars': np.sum(valid_qc),
                'masked_ratio': masked_ratio
            })
            
    df_raw = pd.DataFrame(image_stats_raw)
    df_qc = pd.DataFrame(image_stats_qc)
    
    if len(df_raw) == 0:
        print("No valid data points found.")
        return
        
    def aggregate_and_plot(df, title_suffix, color_main, color_blank, out_suffix):
        if len(df) == 0: return
        df_agg = df.groupby(['true_conc', 'run_id', 'sample_id']).agg(
            median_delta=('trimmed_delta', 'median'),
            iqr_delta=('trimmed_delta', lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        ).reset_index()
        
        plt.figure(figsize=(10, 7))
        df_inc = df_agg[df_agg['true_conc'] > 0]
        if len(df_inc) > 0:
            log_c = np.log10(df_inc['true_conc'].values)
            y_true = df_inc['median_delta'].values
            y_err = df_inc['iqr_delta'].values / 2.0
            slope, intercept, r_value, _, _ = scipy.stats.linregress(log_c, y_true)
            plt.errorbar(log_c, y_true, yerr=y_err, fmt='o', color=color_main, markersize=8, capsize=4, label='Data')
            x_line = np.linspace(min(log_c), max(log_c), 100)
            plt.plot(x_line, slope * x_line + intercept, color='red', linestyle='--', label=f'Fit (R² = {r_value**2:.3f})')
            
        df_blank = df_agg[df_agg['true_conc'] == 0]
        if not df_blank.empty:
            blank_x = np.log10(df_inc['true_conc'].min()) - 1.5 if not df_inc.empty else -15
            plt.errorbar([blank_x]*len(df_blank), df_blank['median_delta'], yerr=df_blank['iqr_delta']/2.0,
                         fmt='s', color=color_blank, markersize=8, capsize=4, label='Blank')
            plt.axvline(x=blank_x + 0.75, color='gray', linestyle=':', alpha=0.5)
            
        plt.title(f"Dose Response Curve ({assay_type}) - {title_suffix}", fontsize=14)
        plt.xlabel("Log10 Concentration (M)", fontsize=12)
        plt.ylabel("Δ Intensity (Median of Trimmed Means)", fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(out_png.replace('.png', out_suffix), dpi=300, bbox_inches='tight')
        plt.close()
        
    aggregate_and_plot(df_raw, "WITHOUT EXCLUSION (Raw)", 'gray', 'lightgray', '_raw.png')
    aggregate_and_plot(df_qc, "WITH EXCLUSION (L1/L2/L3 QC)", 'blue', 'orange', '_qc.png')
    print(f"Saved plots to {out_png.replace('.png', '_raw.png')} and _qc.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--roi", type=str, default=None)
    args = parser.parse_args()
    analyze_and_plot_v2(args.assay, args.date, args.dir, args.out, args.roi)
