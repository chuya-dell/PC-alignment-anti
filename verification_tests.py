import os
import cv2
import numpy as np
import scipy.stats
from qc_filter_v2 import get_l3_mask

def extract_delta(pre_img, post_img, xi, yi):
    # Dummy delta extraction for testing
    h, w = pre_img.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(pre_img * window, post_img * window)
    dx, dy = shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    pre_blur = cv2.blur(pre_img, (3, 3))
    post_blur = cv2.blur(post_warped, (3, 3))
    
    pre_ints = pre_blur[yi, xi]
    post_ints = post_blur[yi, xi]
    valid = post_ints > 0
    
    if np.sum(valid) == 0: return 0.0
    
    delta = post_ints[valid] - pre_ints[valid]
    return scipy.stats.trim_mean(delta, 0.2)

def extract_pillars(pre_img):
    th, bl, pct = 15, 7, 99.9
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
    
    if num_peaks <= 1: return np.array([]), np.array([])
    h, w = pre_img.shape
    xi = np.clip(np.round(centroids[1:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(centroids[1:, 1]).astype(np.int32), 0, h - 1)
    return xi, yi

def test_noop(pre_path, post_path):
    pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    xi, yi = extract_pillars(pre_img)
    
    # 1. Base Delta
    delta_base = extract_delta(pre_img, post_img, xi, yi)
    
    # 2. Mask with nothing
    valid_mask, _ = get_l3_mask("999999", 999, 999, xi, yi, pre_img.shape[1], pre_img.shape[0])
    xi_masked = xi[valid_mask]
    yi_masked = yi[valid_mask]
    
    delta_masked = extract_delta(pre_img, post_img, xi_masked, yi_masked)
    
    assert np.isclose(delta_base, delta_masked), f"No-op failed: base={delta_base}, masked={delta_masked}"
    print(f"[OK] No-op validation passed. Delta={delta_base:.6f}")
    return delta_base, xi, yi

def test_recovery(pre_path, post_path, xi, yi, original_delta):
    pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    # Inject fake dust/stain at (2048, 2048) radius 1500 (massive shift to break 20% trim_mean)
    cx, cy, r_fake = 2048, 2048, 1500
    Y, X = np.ogrid[:pre_img.shape[0], :pre_img.shape[1]]
    dist = (X - cx)**2 + (Y - cy)**2
    mask = dist <= r_fake**2
    post_img[mask] = np.clip(post_img[mask] + 0.1, 0, 1.0)
    
    # Recalculate pillars with fake defect
    xi_fake, yi_fake = extract_pillars(pre_img)
    delta_infected = extract_delta(pre_img, post_img, xi_fake, yi_fake)
    
    # Apply manual mask specifically for the fake defect
    dist_sq = (xi_fake - cx)**2 + (yi_fake - cy)**2
    valid_mask = dist_sq > (r_fake + 5)**2 # slight margin
    
    xi_recovered = xi_fake[valid_mask]
    yi_recovered = yi_fake[valid_mask]
    
    delta_recovered = extract_delta(pre_img, post_img, xi_recovered, yi_recovered)
    
    print(f"Original Delta: {original_delta:.6f}")
    print(f"Infected Delta: {delta_infected:.6f} (Should be corrupted)")
    print(f"Recovered Delta: {delta_recovered:.6f} (Should match Original closely)")
    
    # Since extracting pillars on corrupted image might slightly shift background filter, 
    # the recovered delta might not be *bitwise* identical but should be extremely close.
    assert np.isclose(original_delta, delta_recovered, atol=1e-3), "Recovery validation failed!"
    print("[OK] Recovery validation passed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, required=True)
    parser.add_argument("--post", type=str, required=True)
    args = parser.parse_args()
    
    print("Running No-op Validation...")
    base_delta, xi, yi = test_noop(args.pre, args.post)
    
    print("Running Recovery Validation...")
    test_recovery(args.pre, args.post, xi, yi, base_delta)
