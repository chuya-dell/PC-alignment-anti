import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob

def auto_detect_defects_v2(pre_path, out_png=None):
    img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = img.shape
    
    # 1. Background and Contrast estimation
    # Use a large median/box filter to find the baseline of the pillars
    bg_mean = cv2.boxFilter(img, -1, (101, 101))
    
    mean_sq = cv2.boxFilter(img**2, -1, (51, 51))
    sq_mean = cv2.boxFilter(img, -1, (51, 51))**2
    variance = mean_sq - sq_mean
    variance[variance < 0] = 0
    std_dev = np.sqrt(variance)
    median_std = np.median(std_dev)
    
    # --- DUST (White spots) ---
    # Dust is significantly brighter than the local background.
    # We blur slightly to merge the bright pixels of a single dust particle.
    blurred_for_dust = cv2.GaussianBlur(img, (11, 11), 0)
    # White dust mask: much brighter than background OR extremely high absolute intensity
    dust_mask = (blurred_for_dust > bg_mean + 0.1) | (img > 0.8)
    
    kernel_dust = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dust_mask = cv2.morphologyEx(dust_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_dust)
    dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_DILATE, kernel_dust) # Expand slightly for safety halo
    
    # --- STAIN (Black spots / Contrast loss) ---
    # Stains appear dark (lower than normal background) OR lose their pillar contrast (low variance)
    blurred_for_stain = cv2.GaussianBlur(img, (31, 31), 0)
    global_median_brightness = np.median(bg_mean)
    
    stain_mask_dark = blurred_for_stain < global_median_brightness * 0.5 # 50% darker than typical image
    stain_mask_flat = std_dev < median_std * 0.4 # Lost 60% of its pillar contrast (smooth black area)
    
    stain_mask = stain_mask_dark | stain_mask_flat
    
    kernel_stain = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    stain_mask = cv2.morphologyEx(stain_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_stain)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel_stain)
    
    # Combine
    combined_mask = np.maximum(dust_mask, stain_mask)
    
    if out_png:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        axes[0].set_title(f"Original: {os.path.basename(pre_path)}")
        
        # Overlay mask on image (Dust = Blue, Stain = Red)
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        # Normalize background image for display
        img_disp = np.clip((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5), 0, 1)
        
        overlay[..., 0] = img_disp
        overlay[..., 1] = img_disp
        overlay[..., 2] = img_disp
        
        # Apply Red for Stain
        overlay[stain_mask > 0] = [1.0, 0.0, 0.0]
        # Apply Blue for Dust (overwrites stain if overlap)
        overlay[dust_mask > 0] = [0.0, 0.5, 1.0]
        
        axes[1].imshow(overlay)
        axes[1].set_title("Red = Stain (Black/Flat), Blue = Dust (White)")
        
        axes[2].imshow(combined_mask, cmap='gray')
        axes[2].set_title("Final Boolean Mask")
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        
    return combined_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    auto_detect_defects_v2(args.img, args.out)
