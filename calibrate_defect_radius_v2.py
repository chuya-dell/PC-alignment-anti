import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calibrate_defect_radius(img_path, cx, cy, max_radius=300):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
        
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = img.shape
    
    # Pillar extraction (TopHat 99.9%)
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
        print("No pillars found.")
        return
        
    xi = np.clip(np.round(centroids[1:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(centroids[1:, 1]).astype(np.int32), 0, h - 1)
    
    pre_blur = cv2.blur(img, (3, 3))
    pillar_ints = pre_blur[yi, xi]
    
    dist_from_center = np.sqrt((xi - cx)**2 + (yi - cy)**2)
    
    # Calculate global background using pillars far from the defect
    global_bg_mask = dist_from_center > max_radius * 1.5
    if np.sum(global_bg_mask) < 10:
        global_bg_mask = dist_from_center > max_radius
        
    global_bg_mean = np.mean(pillar_ints[global_bg_mask])
    global_bg_std = np.std(pillar_ints[global_bg_mask])
    
    bin_width = 10 # 10px bins
    bins = np.arange(0, max_radius, bin_width)
    mean_intensities = []
    
    for r in bins:
        mask = (dist_from_center >= r) & (dist_from_center < r + bin_width)
        if np.sum(mask) > 0:
            mean_intensities.append(np.mean(pillar_ints[mask]))
        else:
            mean_intensities.append(np.nan)
            
    mean_intensities = np.array(mean_intensities)
    
    # Convergence calculation
    def find_convergence(threshold_sigma):
        thresh = threshold_sigma * global_bg_std
        diff = np.abs(mean_intensities - global_bg_mean)
        # Find first index where 3 consecutive bins are < thresh
        for i in range(len(diff) - 2):
            if not np.isnan(diff[i:i+3]).any() and np.all(diff[i:i+3] < thresh):
                return bins[i]
        return None
        
    conv_03 = find_convergence(0.3)
    conv_05 = find_convergence(0.5)
    conv_10 = find_convergence(1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bins, mean_intensities, 'o-', label="Pillar Mean Intensity", color='blue', linewidth=2)
    plt.axhline(global_bg_mean, color='red', linestyle='-', label=f"Baseline (μ = {global_bg_mean:.4f})")
    plt.axhline(global_bg_mean + global_bg_std, color='red', linestyle=':', alpha=0.5, label=f"μ ± 1.0σ (σ = {global_bg_std:.4f})")
    plt.axhline(global_bg_mean - global_bg_std, color='red', linestyle=':', alpha=0.5)
    
    colors = {0.3: 'green', 0.5: 'orange', 1.0: 'purple'}
    if conv_03: plt.axvline(conv_03, color=colors[0.3], linestyle='--', label=f"Conv (0.3σ) = {conv_03} px")
    if conv_05: plt.axvline(conv_05, color=colors[0.5], linestyle='--', label=f"Conv (0.5σ) = {conv_05} px")
    if conv_10: plt.axvline(conv_10, color=colors[1.0], linestyle='--', label=f"Conv (1.0σ) = {conv_10} px")
    
    plt.title(f"Stain Radial Intensity Profile (Pillar-Centric)\nImage: {os.path.basename(img_path)}\nCenter: ({cx}, {cy})")
    plt.xlabel(f"Distance from Center (pixels) - Bin Width {bin_width}px")
    plt.ylabel("Pillar Intensity (0-1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_png = os.path.join(os.path.dirname(img_path), f"radial_profile_v2_{os.path.basename(img_path)}_{cx}_{cy}.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", f"radial_profile_v2.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    
    print(f"Saved v2 radial profile to {brain_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--r", type=int, default=300)
    args = parser.parse_args()
    calibrate_defect_radius(args.img, args.cx, args.cy, args.r)
