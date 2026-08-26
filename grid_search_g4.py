import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import cv2
from itertools import product
import copy

def get_conc_map():
    # True mapping for 2060824
    return {
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

def run_g4_permutation_test(df_stats, conc_map, iterations=1000):
    df = df_stats.copy()
    df['cond_id'] = df['series_id'].apply(lambda x: str(x).split('-')[0])
    
    data = []
    for cond_id, group in df.groupby('cond_id'):
        if cond_id not in conc_map:
            continue
        true_conc = conc_map[cond_id]
        agg_trimmed = group['trimmed_delta'].mean()
        data.append({
            'cond_id': cond_id,
            'true_conc': true_conc,
            'trimmed_delta': agg_trimmed
        })
        
    df_agg = pd.DataFrame(data)
    df_fit = df_agg[df_agg['true_conc'] > 0].copy()
    if len(df_fit) < 3:
        return 0, 0, 1.0
        
    log_c = np.log10(df_fit['true_conc'].values)
    y_true = df_fit['trimmed_delta'].values
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_c, y_true)
    true_r2 = r_value**2
    
    null_r2s = []
    np.random.seed(42)
    for _ in range(iterations):
        y_shuffled = np.random.permutation(y_true)
        _, _, r, _, _ = scipy.stats.linregress(log_c, y_shuffled)
        null_r2s.append(r**2)
        
    null_r2s = np.array(null_r2s)
    p_95 = np.percentile(null_r2s, 95)
    p_val = np.sum(null_r2s >= true_r2) / iterations
    
    return true_r2, p_95, p_val

def grid_search_pipeline(exp_dir, output_csv):
    t_start = time.time()
    exp_dir = os.path.abspath(exp_dir)
    print(f"Starting Grid Search in: {exp_dir}")
    
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "*-0.tif")))
    
    # Pre-load and pre-align to save time
    print("Pre-loading and aligning images...")
    cache = {}
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        post_path = os.path.join(exp_dir, f"{base_prefix}-1.tif")
        if not os.path.exists(post_path):
            continue
            
        pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        
        h, w = pre_img.shape
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
        dx, dy = shift
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        cache[base_prefix] = {
            'pre': pre_img,
            'post_warped': post_warped,
            'h': h, 'w': w
        }
        
    # Parameter grid
    top_hat_sizes = [15, 31, 51]
    blurs = [3, 5, 7]
    percentiles = [99.0, 99.5, 99.9]
    trim_ratios = [0.0, 0.1, 0.2]
    
    combinations = list(product(top_hat_sizes, blurs, percentiles, trim_ratios))
    print(f"Total parameter combinations to test: {len(combinations)}")
    
    conc_map = get_conc_map()
    results = []
    
    for idx, (th, bl, pct, tr) in enumerate(combinations):
        if idx % 10 == 0:
            print(f"Testing combination {idx+1}/{len(combinations)}...")
            
        image_stats = []
        
        for base_prefix, data in cache.items():
            pre_img = data['pre']
            post_warped = data['post_warped']
            h, w = data['h'], data['w']
            
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
                
            xi = np.round(centroids[1:, 0]).astype(np.int32)
            yi = np.round(centroids[1:, 1]).astype(np.int32)
            xi = np.clip(xi, 0, w - 1)
            yi = np.clip(yi, 0, h - 1)
            
            pre_blur = cv2.blur(pre_img, (3, 3))
            post_blur = cv2.blur(post_warped, (3, 3))
            pre_ints = pre_blur[yi, xi]
            post_ints = post_blur[yi, xi]
            
            valid = post_ints > 0
            pre_ints = pre_ints[valid]
            post_ints = post_ints[valid]
            
            if len(pre_ints) == 0:
                continue
                
            delta = post_ints - pre_ints
            trimmed_delta = scipy.stats.trim_mean(delta, tr)
            
            image_stats.append({
                'series_id': base_prefix,
                'trimmed_delta': trimmed_delta
            })
            
        df_stats = pd.DataFrame(image_stats)
        if df_stats.empty:
            continue
            
        true_r2, null_95, p_val = run_g4_permutation_test(df_stats, conc_map, iterations=1000)
        
        results.append({
            'top_hat': th,
            'blur': bl,
            'percentile': pct,
            'trim_ratio': tr,
            'true_r2': true_r2,
            'null_95': null_95,
            'p_value': p_val,
            'significant': true_r2 > null_95
        })
        
    df_results = pd.DataFrame(results)
    df_results.sort_values(by='true_r2', ascending=False, inplace=True)
    df_results.to_csv(output_csv, index=False)
    
    print(f"\nGrid Search completed in {time.time() - t_start:.2f} seconds.")
    print(f"Results saved to: {output_csv}")
    
    print("\n--- Top 5 Configurations ---")
    print(df_results.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    grid_search_pipeline(args.exp_dir, args.out)
