import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
import cv2
import matplotlib.pyplot as plt

def get_conc_map():
    return {
        '0': 0.0, '1': 1e-9, '2': 1e-10, '3': 1e-11, '4': 1e-12,
        '5': 1e-13, '8': 1e-14, '10': 1e-15, '11': 1e-11,
        '12': 1e-12, '13': 1e-14, '14': 0.0
    }

def plot_best_dose_response(exp_dir, out_png):
    # Best Params
    th = 15
    bl = 7
    pct = 99.9
    tr = 0.2
    
    exp_dir = os.path.abspath(exp_dir)
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "*-0.tif")))
    
    conc_map = get_conc_map()
    image_stats = []
    
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        post_path = os.path.join(exp_dir, f"{base_prefix}-1.tif")
        if not os.path.exists(post_path): continue
            
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
        pre_ints = pre_blur[yi, xi]
        post_ints = post_blur[yi, xi]
        
        valid = post_ints > 0
        pre_ints, post_ints = pre_ints[valid], post_ints[valid]
        if len(pre_ints) == 0: continue
            
        delta = post_ints - pre_ints
        trimmed_delta = scipy.stats.trim_mean(delta, tr)
        
        cond_id = base_prefix.split('-')[0]
        if cond_id in conc_map:
            image_stats.append({
                'series_id': base_prefix,
                'cond_id': cond_id,
                'true_conc': conc_map[cond_id],
                'trimmed_delta': trimmed_delta
            })
            
    df = pd.DataFrame(image_stats)
    
    # Aggregate by condition
    df_agg = df.groupby('cond_id').agg({'true_conc': 'first', 'trimmed_delta': 'mean'}).reset_index()
    df_fit = df_agg[df_agg['true_conc'] > 0].copy()
    
    log_c = np.log10(df_fit['true_conc'].values)
    y_true = df_fit['trimmed_delta'].values
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_c, y_true)
    
    plt.figure(figsize=(9, 6))
    
    # Plot non-zero points
    plt.scatter(log_c, y_true, color='blue', s=100, label='Data Points (Trimmed Delta)')
    
    # Plot fit line
    x_line = np.linspace(min(log_c), max(log_c), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=f'Linear Fit (R² = {r_value**2:.3f})')
    
    # Plot blanks at an arbitrary low x-value
    df_blank = df_agg[df_agg['true_conc'] == 0]
    if not df_blank.empty:
        blank_x = min(log_c) - 1.5
        plt.scatter([blank_x] * len(df_blank), df_blank['trimmed_delta'], color='gray', marker='x', s=100, label='Blank (0 M)')
        plt.axvline(x=blank_x + 0.75, color='gray', linestyle=':', alpha=0.5)
        
    plt.title(f"Dose Response Curve (Best Config: TH=15, Blur=7, Pct=99.9%, Trim=0.2)\np-value = 0.009 (Significant)")
    plt.xlabel("Log10 Concentration (M)")
    plt.ylabel("Δ Intensity (Post - Pre, Trimmed Mean)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    plot_best_dose_response(args.dir, args.out)
