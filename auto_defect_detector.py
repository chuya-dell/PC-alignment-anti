import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def auto_detect_defects(pre_path, out_png=None):
    img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = img.shape
    
    # 1. Detect Huge Dust (Very Bright Spots)
    # Blur to remove standard small pillars
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Find background using a large median or box filter
    bg = cv2.boxFilter(img, -1, (101, 101))
    
    # Deviation from background
    diff = blurred - bg
    
    # Pillars are periodic and small. Dust is large and bright.
    # We can threshold the difference. 
    dust_mask = diff > np.percentile(diff, 99.9) # top 0.1% brightness might be dust if it's large
    
    # Let's try morphological closing to group dust halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dust_mask = cv2.morphologyEx(dust_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # 2. Detect Stains (Loss of contrast or abnormal background)
    # Contrast is the variance of the image locally
    mean_sq = cv2.boxFilter(img**2, -1, (51, 51))
    sq_mean = cv2.boxFilter(img, -1, (51, 51))**2
    variance = mean_sq - sq_mean
    variance[variance < 0] = 0
    std_dev = np.sqrt(variance)
    
    # Stains often destroy the pillar contrast (very low std_dev) or add huge noise (very high std_dev)
    median_std = np.median(std_dev)
    # Stains are where std_dev is abnormally high (edges of stains) or abnormally low (inside thick stains)
    stain_mask = (std_dev < median_std * 0.3) | (std_dev > median_std * 3.0)
    
    stain_mask = cv2.morphologyEx(stain_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, kernel)
    
    # Combine
    combined_mask = np.maximum(dust_mask, stain_mask)
    
    if out_png:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        axes[0].set_title("Original Pre-Image")
        
        axes[1].imshow(std_dev, cmap='viridis')
        axes[1].set_title("Local Standard Deviation")
        
        # Overlay mask on image
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[..., 0] = img # R
        overlay[..., 1] = img # G
        overlay[..., 2] = img # B
        overlay[combined_mask > 0] = [1.0, 0.0, 0.0] # Red mask
        
        axes[2].imshow(overlay)
        axes[2].set_title("Detected Defects (Red)")
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        
    return combined_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    auto_detect_defects(args.img, args.out)
