import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt

def get_sample_info(date_str, assay_type, sample_id, base_dir):
    ledger_path = os.path.join(base_dir, "experiment_ledger.csv")
    df = pd.read_csv(ledger_path)
    df = df[(df['Date'] == int(date_str)) & (df['ExperimentType'] == assay_type) & (df['SampleID'] == sample_id)]
    if len(df) == 0:
        return False, None, None, "Not found", None
    row = df.iloc[0]
    if row['NumberStatus'] != 'valid' or row['WellStatus'] != 'normal':
        return False, None, None, f"Invalid status: {row['NumberStatus']}/{row['WellStatus']}", None
    return True, row['Concentration_M'], f"{assay_type}_{date_str}_{sample_id}", "", row['SubstrateID']

def get_l3_mask_values(date_str, sub_id, set_num, seq_num, xi, yi, exp_dir):
    # Expecting mask name like: 260706_1-1_mask.npy
    mask_path = os.path.join(exp_dir, "auto_masks", f"{date_str}_{sub_id}-{set_num}_mask.npy")
    if not os.path.exists(mask_path):
        return np.ones(len(xi), dtype=bool), 0.0
        
    mask = np.load(mask_path)
    h, w = mask.shape
    # clamp coordinates just in case
    xic = np.clip(xi, 0, w-1)
    yic = np.clip(yi, 0, h-1)
    valid = mask[yic, xic]
    return valid, 1.0 - (np.sum(valid) / len(valid))

def extract_pillars_otsu(pre_img):
    # Otsu extraction identical to original analyzer.py "blob" method
    tophat = cv2.morphologyEx(pre_img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1: return np.array([]), np.array([])
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_indices = np.where((areas >= 3) & (areas <= 150))[0] + 1
    
    if len(valid_indices) == 0: return np.array([]), np.array([])
    
    centroids_valid = centroids[valid_indices]
    xi = np.clip(np.round(centroids_valid[:, 0]).astype(np.int32), 0, pre_img.shape[1] - 1)
    yi = np.clip(np.round(centroids_valid[:, 1]).astype(np.int32), 0, pre_img.shape[0] - 1)
    return xi, yi

def get_l3_mask_values(date_str, sample_id, pos_id, xi, yi, exp_dir):
    mask_path = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\auto_masks", f"{date_str}_{sample_id}-{pos_id}_mask.npy")
    if not os.path.exists(mask_path):
        return np.ones(len(xi), dtype=bool), 0.0
    mask = np.load(mask_path)
    h, w = mask.shape
    in_mask = mask[np.clip(yi, 0, h-1), np.clip(xi, 0, w-1)] # True means defective
    valid = ~in_mask
    return valid, 1.0 - (np.sum(valid) / len(valid))

def get_knife_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[h//2 - 50 : h//2 + 50, :] = True
    return mask

def analyze_and_plot_v4(assay_type, date_str, exp_dir, out_png, no_mask=False):
    exp_dir = os.path.abspath(exp_dir)
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "**", "*-0.tif"), recursive=True))
    if not pre_files:
        print(f"No tif files found in {exp_dir}")
        return
        
    image_stats = []
    base_dir = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer"
    
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        parts = base_prefix.split('-')
        if len(parts) < 2: continue
            
        sample_id = int(parts[0])
        pos_id = int(parts[1])
        
        post_path = os.path.join(os.path.dirname(pre_path), f"{base_prefix}-1.tif")
        if not os.path.exists(post_path): continue
        
        is_valid_l1, true_conc, run_id, reason, substrate_id = get_sample_info(date_str, assay_type, sample_id, base_dir)
        if not is_valid_l1: continue
            
        pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        h, w = pre_img.shape
        
        xi, yi = extract_pillars_otsu(pre_img)
        if len(xi) == 0: continue
            
        knife_mask = get_knife_mask(h, w)
        valid_raw = ~knife_mask[yi, xi]
        
        # Align images using Phase Correlate
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        pre_blur = cv2.blur(pre_img, (3, 3))
        post_blur = cv2.blur(post_warped, (3, 3))
        
        pre_ints = pre_blur[yi, xi]
        post_ints = post_blur[yi, xi]
        
        valid_p = post_ints > 0
        valid_raw = valid_raw & valid_p
        
        # QC mask
        if no_mask:
            valid_l3 = np.ones(len(xi), dtype=bool)
            masked_ratio = 0.0
        else:
            valid_l3, masked_ratio = get_l3_mask_values(date_str, sample_id, pos_id, xi, yi, exp_dir)
            
        is_valid_l2 = masked_ratio <= 0.15
        
        valid_qc = valid_raw & valid_l3
        
        delta_qc_arr = post_ints[valid_qc] - pre_ints[valid_qc] if np.sum(valid_qc) > 0 else np.array([])
        
        image_stats.append({
            'sample_id': sample_id, 'pos_id': pos_id, 'true_conc': true_conc, 'run_id': run_id,
            'substrate_id': substrate_id,
            'delta_qc_arr': delta_qc_arr,
            'total_matched': np.sum(valid_raw),
            'valid_pillars': np.sum(valid_qc),
            'is_valid_l1': is_valid_l1, 'is_valid_l2': is_valid_l2, 'masked_ratio': masked_ratio
        })
        
    df = pd.DataFrame(image_stats)
    if len(df) == 0: return
    
    print(f"\n--- Sanity Checks for {date_str} (Mask={'OFF' if no_mask else 'ON'}) ---")
    for _, row in df.iterrows():
        pass_rate = (row['valid_pillars'] / row['total_matched']) * 100.0 if row['total_matched'] > 0 else 0
        print(f"FOV {row['sample_id']}-{row['pos_id']}: Matched={row['total_matched']}, Valid={row['valid_pillars']} (Pass: {pass_rate:.1f}%)")
        if row['total_matched'] < 10000:
            print(f"  -> WARNING: Matched pillars {row['total_matched']} is abnormally low!")
            
    df_blanks_qc = df[(df['true_conc'] == 0) & (df['delta_qc_arr'].apply(lambda x: len(x) > 0))]
    substrate_thresh_qc = {}
    for sub_id, group in df_blanks_qc.groupby('substrate_id'):
        all_qc = np.concatenate(group['delta_qc_arr'].values)
        substrate_thresh_qc[sub_id] = np.mean(all_qc) - 3 * np.std(all_qc)
        
    global_qc_thresh = np.mean(list(substrate_thresh_qc.values())) if substrate_thresh_qc else 0
    
    results_qc = []
    for _, row in df.iterrows():
        sub_id = row['substrate_id']
        qc_t = substrate_thresh_qc.get(sub_id, global_qc_thresh)
        if row['is_valid_l2'] and len(row['delta_qc_arr']) > 0:
            arr = row['delta_qc_arr']
            grid_pct = np.sum(arr < qc_t) / len(arr) * 100.0
            results_qc.append({
                'true_conc': row['true_conc'], 'run_id': row['run_id'], 'sample_id': row['sample_id'],
                'grid_pct': grid_pct,
                'mean_delta': np.mean(arr) * 100.0,
                'arr': arr,
                'total_matched': row['total_matched'],
                'valid_pillars': row['valid_pillars']
            })
            
    if len(results_qc) > 0:
        df_qc = pd.DataFrame(results_qc)
        df_blank = df_qc[df_qc['true_conc'] == 0]
        blank_mean = df_blank['mean_delta'].mean() if len(df_blank) > 0 else 0
        blank_std = df_blank['mean_delta'].std() if len(df_blank) > 1 else 0
        
        summary_rows = []
        for conc, group in df_qc.groupby('true_conc'):
            n_fovs = len(group)
            total_matched = group['total_matched'].sum()
            total_valid = group['valid_pillars'].sum()
            cond_mean = group['mean_delta'].mean()
            cond_grid = group['grid_pct'].mean()
            z_score = (cond_mean - blank_mean) / blank_std if blank_std > 0 else np.nan
            
            all_pillars = np.concatenate(group['arr'].values) if len(group) > 0 else np.array([])
            skewness = stats.skew(all_pillars) if len(all_pillars) > 2 else np.nan
            pass_rate = (total_valid / total_matched) * 100.0 if total_matched > 0 else 0
            
            flags = []
            if total_valid < 10000: flags.append(f"LowValidPillars({total_valid})")
            if blank_std < 0.05 or blank_std > 0.20: flags.append("AbnormalBlankStd")
            if n_fovs > 1:
                grids = group['grid_pct'].values
                if np.max(grids) > 20 and np.max(grids) > 5 * np.median(grids) and np.median(grids) < 10:
                    flags.append("SingleFOVOutlier")
                    
            summary_rows.append({
                'Concentration (M)': conc,
                'FOV Count (n)': n_fovs,
                'Total Matched': total_matched,
                'Valid Pillars': total_valid,
                'Mask Pass Rate (%)': pass_rate,
                'Mean Delta (%)': cond_mean,
                'Neg Grid (%)': cond_grid,
                'Z-Score (vs Blank)': z_score,
                'Skewness': skewness,
                'Blank Mean (%)': blank_mean,
                'Blank Std (%)': blank_std,
                'Flags': " | ".join(flags)
            })
            
        df_summary = pd.DataFrame(summary_rows).sort_values('Concentration (M)', ascending=False)
        out_csv = out_png.replace('.png', '_summary.csv')
        df_summary.to_csv(out_csv, index=False)
        print(f"Saved V4 report to {out_csv}")
        print(df_summary.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC Dose Response Analyzer V4")
    parser.add_argument("--assay", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--no-mask", action="store_true")
    args = parser.parse_args()
    
    analyze_and_plot_v4(args.assay, args.date, args.dir, args.out, args.no_mask)
