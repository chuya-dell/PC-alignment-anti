import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
import cv2
import matplotlib.pyplot as plt
from qc_filter_v2 import get_sample_info, get_l3_mask
from verification_tests_v3 import get_knife_mask, extract_pillars, extract_delta

def analyze_and_plot_v3(assay_type, date_str, exp_dir, out_png, no_mask=False, roi_dir=None):
    exp_dir = os.path.abspath(exp_dir)
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "**", "*-0.tif"), recursive=True))
    
    image_stats = []
    
    # 1. First Pass: Collect valid pillars and deltas for ALL positions
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        parts = base_prefix.split('-')
        if len(parts) < 2: continue
            
        sample_id = int(parts[0])
        pos_id = int(parts[1])
        
        post_path = os.path.join(os.path.dirname(pre_path), f"{base_prefix}-1.tif")
        if not os.path.exists(post_path): continue
            
        pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        h, w = pre_img.shape
        
        xi, yi = extract_pillars(pre_img)
        if len(xi) == 0: continue
            
        is_valid_l1, true_conc, run_id, reason, substrate_id = get_sample_info(date_str, assay_type, sample_id)
        
        # Calculate Base Delta (without QC) for the RAW plot
        knife_mask = get_knife_mask(h, w)
        valid_raw = ~knife_mask[yi, xi]
        delta_raw_arr = None
        
        if np.sum(valid_raw) > 0:
            # We need the full array of deltas to calculate % grid over threshold
            window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            pre_blur = cv2.blur(pre_img, (3, 3))
            post_blur = cv2.blur(post_warped, (3, 3))
            
            p_ints_raw = pre_blur[yi[valid_raw], xi[valid_raw]]
            pt_ints_raw = post_blur[yi[valid_raw], xi[valid_raw]]
            valid_p_raw = pt_ints_raw > 0
            if np.sum(valid_p_raw) > 0:
                delta_raw_arr = pt_ints_raw[valid_p_raw] - p_ints_raw[valid_p_raw]
                
        # Calculate QC Delta
        if no_mask:
            valid_l3 = np.ones_like(valid_raw, dtype=bool)
            masked_ratio = 0.0
        else:
            valid_l3, masked_ratio = get_l3_mask(date_str, sample_id, pos_id, xi, yi, w, h, knife_mask, roi_dir)
        is_valid_l2 = masked_ratio <= 0.15
        
        valid_qc = valid_raw & valid_l3
        delta_qc_arr = None
        
        if is_valid_l1 and is_valid_l2 and np.sum(valid_qc) > 0:
            p_ints_qc = pre_blur[yi[valid_qc], xi[valid_qc]]
            pt_ints_qc = post_blur[yi[valid_qc], xi[valid_qc]]
            valid_p_qc = pt_ints_qc > 0
            if np.sum(valid_p_qc) > 0:
                delta_qc_arr = pt_ints_qc[valid_p_qc] - p_ints_qc[valid_p_qc]
                
        image_stats.append({
            'sample_id': sample_id, 'pos_id': pos_id, 'true_conc': true_conc, 'run_id': run_id,
            'substrate_id': substrate_id,
            'delta_raw_arr': delta_raw_arr, 'delta_qc_arr': delta_qc_arr,
            'is_valid_l1': is_valid_l1, 'is_valid_l2': is_valid_l2
        })
        
    df = pd.DataFrame(image_stats)
    if len(df) == 0: return
    
    # 2. Second Pass: Calculate Substrate-Specific Blank Thresholds
    df_blanks_raw = df[(df['true_conc'] == 0) & (df['delta_raw_arr'].notnull())]
    df_blanks_qc = df[(df['true_conc'] == 0) & (df['delta_qc_arr'].notnull())]
    
    substrate_thresh_raw = {}
    substrate_thresh_qc = {}
    
    for sub_id, group in df_blanks_raw.groupby('substrate_id'):
        all_raw = np.concatenate(group['delta_raw_arr'].values)
        substrate_thresh_raw[sub_id] = np.mean(all_raw) - 3 * np.std(all_raw)
        
    for sub_id, group in df_blanks_qc.groupby('substrate_id'):
        all_qc = np.concatenate(group['delta_qc_arr'].values)
        substrate_thresh_qc[sub_id] = np.mean(all_qc) - 3 * np.std(all_qc)
        
    # Global fallback just in case
    global_raw_thresh = np.mean(list(substrate_thresh_raw.values())) if substrate_thresh_raw else 0
    global_qc_thresh = np.mean(list(substrate_thresh_qc.values())) if substrate_thresh_qc else 0
        
    # 3. Third Pass: Calculate Grid % using Thresholds
    results_raw = []
    results_qc = []
    
    for _, row in df.iterrows():
        sub_id = row['substrate_id']
        raw_t = substrate_thresh_raw.get(sub_id, global_raw_thresh)
        qc_t = substrate_thresh_qc.get(sub_id, global_qc_thresh)
        
        # Raw %
        if row['delta_raw_arr'] is not None:
            arr = row['delta_raw_arr']
            grid_pct = np.sum(arr < raw_t) / len(arr) * 100.0
            results_raw.append({
                'true_conc': row['true_conc'], 'run_id': row['run_id'], 'sample_id': row['sample_id'],
                'grid_pct': grid_pct
            })
            
        # QC %
        if row['delta_qc_arr'] is not None and row['is_valid_l1'] and row['is_valid_l2']:
            arr = row['delta_qc_arr']
            grid_pct = np.sum(arr < qc_t) / len(arr) * 100.0
            results_qc.append({
                'true_conc': row['true_conc'], 'run_id': row['run_id'], 'sample_id': row['sample_id'],
                'grid_pct': grid_pct,
                'mean_delta': np.mean(arr) if len(arr) > 0 else 0,
                'arr': arr
            })
            
    df_res_raw = pd.DataFrame(results_raw)
    df_res_qc = pd.DataFrame(results_qc)
    
    # 4. Export to Excel (Raw Pillar Deltas for Prism)
    excel_out = out_png.replace('.png', '.xlsx')
    
    with pd.ExcelWriter(excel_out) as writer:
        concs = sorted(df['true_conc'].unique(), reverse=True)
        if 0 in concs:
            concs.remove(0)
            concs.append(0)
            
        for c in concs:
            sheet_name = f"{c:g} M" if c > 0 else "Blank"
            # Get all valid qc arrays for this concentration
            group = df[(df['true_conc'] == c) & (df['is_valid_l1'] == True) & (df['is_valid_l2'] == True)]
            
            col_data = {}
            max_len = 0
            for _, row in group.iterrows():
                arr = row['delta_qc_arr']
                if arr is not None:
                    col_name = f"Sub{row['substrate_id']}_S{row['sample_id']}_P{row['pos_id']}"
                    col_data[col_name] = arr
                    max_len = max(max_len, len(arr))
                    
            if col_data:
                # Pad with NaNs so we can make a DataFrame
                for k in col_data:
                    arr = col_data[k]
                    if len(arr) < max_len:
                        col_data[k] = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                        
                df_sheet = pd.DataFrame(col_data)
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                
    def aggregate_and_plot(df_res, title_suffix, color_main, color_blank, out_suffix):
        if len(df_res) == 0: return
        df_agg = df_res.groupby(['true_conc', 'run_id', 'sample_id']).agg(
            mean_grid=('grid_pct', 'mean'),
            std_grid=('grid_pct', 'std')
        ).reset_index()
        plt.figure(figsize=(10, 7))
        df_inc = df_agg[df_agg['true_conc'] > 0]
        if len(df_inc) > 0:
            log_c = np.log10(df_inc['true_conc'].values)
            y_true = df_inc['mean_grid'].values
            y_err = df_inc['std_grid'].values
            y_err = np.nan_to_num(y_err, nan=0.0)
            
            slope, intercept, r_value, _, _ = scipy.stats.linregress(log_c, y_true)
            plt.errorbar(log_c, y_true, yerr=y_err, fmt='o', color=color_main, markersize=8, capsize=4, label='Data (Mean ± Std)')
            x_line = np.linspace(min(log_c), max(log_c), 100)
            plt.plot(x_line, slope * x_line + intercept, color='red', linestyle='--', label=f'Fit (R² = {r_value**2:.3f})')
            
        df_blank = df_agg[df_agg['true_conc'] == 0]
        if not df_blank.empty:
            blank_x = np.log10(df_inc['true_conc'].min()) - 1.5 if not df_inc.empty else -15
            y_err = np.nan_to_num(df_blank['std_grid'].values, nan=0.0)
            plt.errorbar([blank_x]*len(df_blank), df_blank['mean_grid'], yerr=y_err,
                         fmt='s', color=color_blank, markersize=8, capsize=4, label='Blank (N.C.)')
            plt.axvline(x=blank_x + 0.75, color='gray', linestyle=':', alpha=0.5)
            
        plt.title(f"Dose Response Curve ({assay_type}) - {title_suffix}", fontsize=14)
        plt.xlabel("Log10 Concentration (M)", fontsize=12)
        plt.ylabel("Grid > N.C. ave + 3σ (%)", fontsize=12)
        plt.ylim(-5, 105)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(out_png.replace('.png', out_suffix), dpi=300, bbox_inches='tight')
        plt.close()
        
    out_raw = out_png.replace('.png', '_raw_grid.png')
    out_qc = out_png.replace('.png', '_qc_grid.png')
    aggregate_and_plot(df_res_raw, "WITHOUT EXCLUSION (Raw - Grid %)", 'gray', 'lightgray', '_raw_grid.png')
    aggregate_and_plot(df_res_qc, "WITH EXCLUSION (L1/L2/L3 QC - Grid %)", 'blue', 'orange', '_qc_grid.png')
    print(f"Saved plots to {out_raw} and {out_qc}")
    
    # Generate automatic summary CSV with advanced flags and skewness
    if len(results_qc) > 0:
        import scipy.stats as stats
        df_qc = pd.DataFrame(results_qc)
        df_blank = df_qc[df_qc['true_conc'] == 0]
        if len(df_blank) > 0:
            blank_mean = df_blank['mean_delta'].mean()
            blank_std = df_blank['mean_delta'].std() if len(df_blank) > 1 else 0
        else:
            blank_mean, blank_std = 0, 0
            
        summary_rows = []
        for conc, group in df_qc.groupby('true_conc'):
            n_fovs = len(group)
            total_pillars = sum(len(x) for x in group['arr'])
            cond_mean = group['mean_delta'].mean()
            cond_grid = group['grid_pct'].mean()
            z_score = (cond_mean - blank_mean) / blank_std if blank_std > 0 else np.nan
            
            # Skewness calculation (using all pillars in condition)
            all_pillars = np.concatenate(group['arr'].values) if len(group) > 0 else np.array([])
            skewness = stats.skew(all_pillars) if len(all_pillars) > 2 else np.nan
            
            # Flags calculation
            flags = []
            if n_fovs == 1:
                flags.append("n=1")
            if total_pillars < 50:
                flags.append(f"LowPillarCount({total_pillars})")
            if blank_std < 0.05 or blank_std > 0.20:
                flags.append("AbnormalBlankStd")
                
            # Check for extreme single-FOV contamination
            if n_fovs > 1:
                grids = group['grid_pct'].values
                if np.max(grids) > 20 and np.max(grids) > 5 * np.median(grids) and np.median(grids) < 10:
                    flags.append("SingleFOVOutlier")
                    
            summary_rows.append({
                'Concentration (M)': conc,
                'FOV Count (n)': n_fovs,
                'Valid Pillars': total_pillars,
                'Mean Delta (%)': cond_mean,
                'Neg Grid (%)': cond_grid,
                'Z-Score (vs Blank)': z_score,
                'Skewness': skewness,
                'Blank Mean (%)': blank_mean,
                'Blank Std (%)': blank_std,
                'Flags': " | ".join(flags)
            })
            
        df_summary = pd.DataFrame(summary_rows).sort_values('Concentration (M)', ascending=False)
        out_csv = args.out.replace('.png', '_summary.csv')
        df_summary.to_csv(out_csv, index=False)
        print(f"Saved auto-summary report to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QC Dose Response Analyzer V3")
    parser.add_argument("--assay", type=str, required=True, help="SAM or DNA")
    parser.add_argument("--date", type=str, required=True, help="Experiment date (e.g., 260828)")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the npy masks")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    parser.add_argument("--no-mask", action="store_true", help="Disable L3 masking")
    parser.add_argument("--roi", type=str, default=None)
    args = parser.parse_args()
    analyze_and_plot_v3(args.assay, args.date, args.dir, args.out, args.no_mask)
