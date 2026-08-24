import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import cv2

def process_all_pairs(exp_dir, output_summary_csv):
    """
    Phase 3: Image-level Statistical Unit
    Processes all Pre/Post pairs in the experiment directory,
    warps Post to Pre, samples intensities, and aggregates robust image-level statistics.
    """
    t_start = time.time()
    exp_dir = os.path.abspath(exp_dir)
    print(f"Processing experiment directory: {exp_dir}")
    
    # Find all Pre images (ending in -0.tif)
    pre_files = sorted(glob.glob(os.path.join(exp_dir, "*-0.tif")))
    
    image_stats = []
    
    for pre_path in pre_files:
        basename = os.path.basename(pre_path)
        base_prefix = basename.rsplit('-0.tif', 1)[0]
        post_path = os.path.join(exp_dir, f"{base_prefix}-1.tif")
        
        if not os.path.exists(post_path):
            print(f"Skipping {basename}: No matching Post image found.")
            continue
            
        print(f"\n--- Processing Pair: {base_prefix} ---")
        
        try:
            # 1. Load 16-bit float32 images
            pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            
            # 2. Phase Correlation Alignment
            h, w = pre_img.shape
            window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
            dx, dy = shift
            
            # 3. Warp Post to Pre
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # 4. Detect Pillars in Pre
            from analyzer import process_tile
            # We process the whole image as one tile for simplicity in this extraction
            # In a real pipeline, we would tile this for memory efficiency on huge images
            top_hat = cv2.morphologyEx(pre_img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            blurred = cv2.GaussianBlur(top_hat, (5, 5), 0)
            
            bg_local = cv2.boxFilter(blurred, -1, (21, 21))
            local_contrast = blurred - bg_local
            
            pos_contrast = local_contrast[local_contrast > 0]
            thresh_val = np.percentile(pos_contrast, 99.5) if len(pos_contrast) > 0 else 0.01
            
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)) # min_dist=4 -> 9x9
            dilated = cv2.dilate(blurred, kernel_dil)
            
            peaks_mask = (blurred == dilated) & (local_contrast >= thresh_val)
            num_peaks, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
            
            if num_peaks <= 1:
                print(f"Warning: No pillars detected in {basename}")
                continue
                
            xi = np.round(centroids[1:, 0]).astype(np.int32)
            yi = np.round(centroids[1:, 1]).astype(np.int32)
            
            # Clip
            xi = np.clip(xi, 0, w - 1)
            yi = np.clip(yi, 0, h - 1)
            
            # 5. Sample paired intensities
            pre_blur = cv2.blur(pre_img, (3, 3))
            post_blur = cv2.blur(post_warped, (3, 3))
            
            pre_ints = pre_blur[yi, xi]
            post_ints = post_blur[yi, xi]
            
            valid = post_ints > 0 # Remove border artifacts
            pre_ints = pre_ints[valid]
            post_ints = post_ints[valid]
            
            if len(pre_ints) == 0:
                continue
                
            delta = post_ints - pre_ints
            
            # 6. Aggregate Robust Image-Level Statistics
            median_delta = np.median(delta)
            trimmed_delta = scipy.stats.trim_mean(delta, 0.1)
            
            image_stats.append({
                'series_id': base_prefix,
                'n_pillars': len(delta),
                'dx': dx,
                'dy': dy,
                'median_delta': median_delta,
                'trimmed_delta': trimmed_delta
            })
            
            print(f"  -> N={len(delta)}, Median Delta={median_delta:.6f}, Trimmed Delta={trimmed_delta:.6f}")
            
        except Exception as e:
            print(f"Error processing {base_prefix}: {e}")
            
    # Save Image-Level Summary
    df_stats = pd.DataFrame(image_stats)
    df_stats.to_csv(output_summary_csv, index=False)
    print(f"\n=======================================================")
    print(f" Saved Image-Level Statistical Summary to: {output_summary_csv}")
    print(f" Total time: {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory containing TIFs")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()
    
    process_all_pairs(args.exp_dir, args.out)
