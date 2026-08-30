import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
from skimage import restoration

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_image(sample, prepost):
    path = glob.glob(os.path.join(p50_dir, "**", f"{sample}-1-{prepost}.tif"), recursive=True)[0]
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

pre_img_blank = get_image(8, 0)
post_img_blank = get_image(8, 1)

pre_img_1nm = get_image(1, 0)
post_img_1nm = get_image(1, 1)

# 1. Re-calculate SNR using Valleys for p50
def recalc_snr(img):
    # Find local maxima
    local_max = ndi.maximum_filter(img, size=5) == img
    # Find local minima
    local_min = ndi.minimum_filter(img, size=5) == img
    
    # Filter out boundaries
    margin = 10
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    
    peaks = img[local_max & mask]
    valleys = img[local_min & mask]
    
    peak_mean = np.mean(peaks)
    valleys_mean = np.mean(valleys)
    valleys_std = np.std(valleys)
    
    snr = (peak_mean - valleys_mean) / valleys_std if valleys_std > 0 else 0
    return peak_mean, valleys_mean, valleys_std, snr, len(peaks)

pm, vm, vstd, snr, n_peaks = recalc_snr(pre_img_blank)
print(f"--- Improved SNR (Valleys as BG) ---")
print(f"Peak: {pm:.4f}, Valley (BG): {vm:.4f}, BG Std: {vstd:.4f}, SNR: {snr:.2f}, Count: {n_peaks}")


# 2. FFT-based grid extraction
def extract_fft_grid(img, pitch_px=6.29):
    # Bandpass filter around the expected frequency to find phase
    f = np.fft.fft2(img - np.mean(img))
    fshift = np.fft.fftshift(f)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask for frequencies corresponding to the pitch
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    
    # Center frequency radius (normalized to 1.0 at Nyquist)
    r_target = rows * freq # assuming square image
    
    mask = (r >= r_target - 15) & (r <= r_target + 15)
    fshift_filtered = fshift * mask
    
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    
    # The filtered image contains the fundamental 2D sine waves of the grid
    # Local maxima of the filtered image represent the theoretical grid points
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 15
    valid_mask = np.zeros_like(img, dtype=bool)
    valid_mask[margin:-margin, margin:-margin] = True
    
    grid_y, grid_x = np.where(local_max & valid_mask)
    return grid_x, grid_y

def evaluate_extraction(name, xi, yi, pre, post):
    pre_ints = pre[yi, xi]
    post_ints = post[yi, xi]
    delta = (post_ints - pre_ints) * 100.0
    
    print(f"--- {name} ---")
    print(f"Extracted Pillars: {len(delta)}")
    print(f"Pre Mean: {np.mean(pre_ints):.4f}")
    print(f"Delta Mean: {np.mean(delta):.4f}%")
    print(f"Delta Std (σ): {np.std(delta):.4f}%")
    return np.mean(delta)

gx, gy = extract_fft_grid(pre_img_blank)
blank_mean = evaluate_extraction("FFT Grid (Blank)", gx, gy, pre_img_blank, post_img_blank)

gx_1nm, gy_1nm = extract_fft_grid(pre_img_1nm)
_1nm_mean = evaluate_extraction("FFT Grid (1nM)", gx_1nm, gy_1nm, pre_img_1nm, post_img_1nm)

print(f"\nBlank Mean: {blank_mean:.4f}%")
print(f"1nM Mean: {_1nm_mean:.4f}%")
print(f"Signal (1nM - Blank): {_1nm_mean - blank_mean:.4f}%")
