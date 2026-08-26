import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_pillars_v3(img):
    th, bl, pct = 15, 7, 99.9
    top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th, th)))
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
        return np.array([]), np.array([])
        
    h, w = img.shape
    xi = np.clip(np.round(centroids[1:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(centroids[1:, 1]).astype(np.int32), 0, h - 1)
    return xi, yi

def calibrate_defect_radius_v3(img_path, cx, cy, visual_radius, max_iterations=5, max_radius=800):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
        
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    pre_blur = cv2.blur(img, (3, 3))
    
    xi, yi = extract_pillars_v3(img)
    if len(xi) == 0:
        print("No pillars found.")
        return
        
    pillar_ints = pre_blur[yi, xi]
    dist_from_center = np.sqrt((xi - cx)**2 + (yi - cy)**2)
    
    # Initial exclusion radius based on visual estimation
    current_exclusion_r = visual_radius * 3
    
    bin_width = 10 # 10px bins
    bins = np.arange(0, max_radius, bin_width)
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # 1. Calculate Baseline and Sigma using points OUTSIDE current_exclusion_r
        clean_mask = dist_from_center > current_exclusion_r
        if np.sum(clean_mask) < 50:
            print("Too few clean pillars. Stopping.")
            break
            
        baseline_mean = np.mean(pillar_ints[clean_mask])
        baseline_std = np.std(pillar_ints[clean_mask]) # Pillar-level sigma
        print(f"Baseline Mean: {baseline_mean:.4f}, Sigma (Pillar): {baseline_std:.4f}")
        
        # 2. Binning
        mean_intensities = []
        n_pillars = []
        for r in bins:
            mask = (dist_from_center >= r) & (dist_from_center < r + bin_width)
            count = np.sum(mask)
            n_pillars.append(count)
            if count > 0:
                mean_intensities.append(np.mean(pillar_ints[mask]))
            else:
                mean_intensities.append(np.nan)
                
        mean_intensities = np.array(mean_intensities)
        n_pillars = np.array(n_pillars)
        
        # 3. Find Convergence
        thresh = 0.5 * baseline_std
        diff = np.abs(mean_intensities - baseline_mean)
        
        new_convergence_r = None
        for i in range(len(diff) - 2):
            # Require 3 consecutive bins within 0.5 sigma, AND sufficient N (>5 per bin ideally)
            if not np.isnan(diff[i:i+3]).any() and np.all(diff[i:i+3] < thresh) and np.all(n_pillars[i:i+3] >= 3):
                new_convergence_r = bins[i]
                break
                
        if new_convergence_r is None:
            print("Did not converge within max_radius.")
            break
            
        print(f"Calculated Convergence Radius (r1): {new_convergence_r} px")
        
        # Check stability
        if abs(new_convergence_r - current_exclusion_r) < bin_width:
            print("Convergence stable!")
            current_exclusion_r = new_convergence_r
            break
            
        # Update for next iteration
        current_exclusion_r = new_convergence_r
        
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Top Plot: Intensity Profile
    ax1.plot(bins, mean_intensities, 'o-', label="Pillar Mean Intensity", color='blue', markersize=4)
    ax1.axhline(baseline_mean, color='red', linestyle='-', label=f"Baseline (μ = {baseline_mean:.4f})")
    ax1.axhline(baseline_mean + 0.5 * baseline_std, color='red', linestyle=':', alpha=0.5, label=f"μ ± 0.5σ (σ={baseline_std:.4f})")
    ax1.axhline(baseline_mean - 0.5 * baseline_std, color='red', linestyle=':', alpha=0.5)
    
    if current_exclusion_r:
        ax1.axvline(current_exclusion_r, color='orange', linestyle='--', label=f"Converged r = {current_exclusion_r} px")
        # Final mask radius (r * 1.2 safety margin)
        final_mask_r = int(current_exclusion_r * 1.2)
        ax1.axvline(final_mask_r, color='purple', linestyle='-', label=f"Mask Radius (r*1.2) = {final_mask_r} px")
        
    ax1.set_title(f"Stain Radial Profile (Iterative v3)\nImage: {os.path.basename(img_path)} | Center: ({cx}, {cy})")
    ax1.set_ylabel("Pillar Intensity (0-1)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom Plot: N count per bin
    ax2.bar(bins, n_pillars, width=bin_width, align='edge', color='gray', alpha=0.5)
    ax2.set_xlabel(f"Distance from Center (pixels) - Bin Width {bin_width}px")
    ax2.set_ylabel("N (Pillars)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_png = os.path.join(os.path.dirname(img_path), f"radial_v3_{os.path.basename(img_path)}_{cx}_{cy}.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", f"radial_v3_{os.path.basename(img_path)}.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    
    print(f"Saved v3 radial profile to {brain_out}")
    return final_mask_r if current_exclusion_r else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--vr", type=int, required=True, help="Visual radius estimate")
    args = parser.parse_args()
    calibrate_defect_radius_v3(args.img, args.cx, args.cy, args.vr)
