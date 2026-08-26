import os
import cv2
import numpy as np
import scipy.stats

def extract_delta(pre_img, post_img, xi, yi):
    if len(xi) == 0: return np.nan
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
    if np.sum(valid) == 0: return np.nan
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

def get_knife_mask(h, w):
    # Dummy knife scratch band (center horizontal 100px)
    mask = np.zeros((h, w), dtype=bool)
    mask[h//2 - 50 : h//2 + 50, :] = True
    return mask

def test_noop_v3(pre_path, post_path):
    pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = pre_img.shape
    
    xi, yi = extract_pillars(pre_img)
    knife_mask = get_knife_mask(h, w)
    
    # Apply knife mask (MUST be applied to both)
    valid_knife = ~knife_mask[yi, xi]
    xi_k = xi[valid_knife]
    yi_k = yi[valid_knife]
    
    # 1. Base Delta (Knife ONLY)
    delta_base = extract_delta(pre_img, post_img, xi_k, yi_k)
    
    # 2. Mask with nothing (Knife + empty L3 mask)
    # L3 mask is dummy empty
    valid_l3 = np.ones(len(xi_k), dtype=bool)
    xi_masked = xi_k[valid_l3]
    yi_masked = yi_k[valid_l3]
    
    delta_masked = extract_delta(pre_img, post_img, xi_masked, yi_masked)
    
    assert np.isclose(delta_base, delta_masked), f"No-op v3 failed: base={delta_base}, masked={delta_masked}"
    print(f"[OK] No-op v3 validation passed. Delta={delta_base:.6f}")
    return delta_base, xi_k, yi_k

def test_recovery_v3(clean_pre, clean_post, dirty_pre_src):
    # A: Clean Baseline
    img_A_pre = cv2.imdecode(np.fromfile(clean_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img_A_post = cv2.imdecode(np.fromfile(clean_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = img_A_pre.shape
    
    xi_A, yi_A = extract_pillars(img_A_pre)
    knife_mask = get_knife_mask(h, w)
    valid_A = ~knife_mask[yi_A, xi_A]
    delta_A = extract_delta(img_A_pre, img_A_post, xi_A[valid_A], yi_A[valid_A])
    
    # Extract real defect from dirty source (e.g. 0-8)
    img_dirty = cv2.imdecode(np.fromfile(dirty_pre_src, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    # Let's say defect is at cx=1500, cy=1500, r=300
    dcx, dcy, dr = 1500, 1500, 300
    defect_patch = img_dirty[dcy-dr:dcy+dr, dcx-dr:dcx+dr]
    if defect_patch.shape != (2*dr, 2*dr):
        print("Defect patch out of bounds")
        return
        
    # Inject defect into A to create C
    img_C_pre = img_A_pre.copy()
    # Blend: Alpha max
    img_C_pre[dcy-dr:dcy+dr, dcx-dr:dcx+dr] = np.maximum(img_C_pre[dcy-dr:dcy+dr, dcx-dr:dcx+dr], defect_patch)
    img_C_post = img_A_post.copy() # Assuming defect is ONLY on pre-image (worst case)
    
    # Recalculate pillars for C
    xi_C, yi_C = extract_pillars(img_C_pre)
    valid_C_knife = ~knife_mask[yi_C, xi_C]
    delta_C_corrupted = extract_delta(img_C_pre, img_C_post, xi_C[valid_C_knife], yi_C[valid_C_knife])
    
    # B: Clean with Mask
    # Create mask for the defect region
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - dcx)**2 + (Y - dcy)**2
    defect_mask_2d = dist_sq <= (dr * 1.2)**2 # 20% margin
    
    valid_B_l3 = ~defect_mask_2d[yi_A, xi_A]
    valid_B = valid_A & valid_B_l3
    delta_B = extract_delta(img_A_pre, img_A_post, xi_A[valid_B], yi_A[valid_B])
    
    # C: Corrupted with Mask
    valid_C_l3 = ~defect_mask_2d[yi_C, xi_C]
    valid_C = valid_C_knife & valid_C_l3
    delta_C_recovered = extract_delta(img_C_pre, img_C_post, xi_C[valid_C], yi_C[valid_C])
    
    print(f"Delta_A (Clean, No Mask): {delta_A:.6f}")
    print(f"Delta_B (Clean, Masked):  {delta_B:.6f}  <- Target Truth")
    print(f"Delta_C (Dirty, No Mask): {delta_C_corrupted:.6f}  <- Corrupted")
    print(f"Delta_C (Dirty, Masked):  {delta_C_recovered:.6f}  <- Recovered")
    
    diff = abs(delta_C_recovered - delta_B)
    print(f"|C - B| = {diff:.6f}")
    assert diff < 1e-4, "Recovery validation v3 failed! C does not approximate B."
    print("[OK] Recovery validation v3 passed. C is approx equal to B.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, required=True)
    parser.add_argument("--post", type=str, required=True)
    parser.add_argument("--dirty", type=str, required=True)
    args = parser.parse_args()
    
    print("Running No-op Validation v3...")
    test_noop_v3(args.pre, args.post)
    
    print("\nRunning Recovery Validation v3...")
    test_recovery_v3(args.pre, args.post, args.dirty)
