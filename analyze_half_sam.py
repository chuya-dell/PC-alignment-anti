import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import sys

def get_coarse_shift_phase_correlate(img0, img1):
    # Use phase correlation for robust translation shift
    # Pad images to avoid edge effects
    img0_pad = np.pad(img0, 100, mode='reflect')
    img1_pad = np.pad(img1, 100, mode='reflect')
    
    # Hann window to reduce edge artifacts
    h, w = img0_pad.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    
    shift, response = cv2.phaseCorrelate(img1_pad.astype(np.float32), img0_pad.astype(np.float32), win)
    dx, dy = shift
    return dx, dy

def extract_pillars(img, freq_expected=1.0/6.29):
    rows, cols = img.shape
    f = np.fft.fft2(img - np.nanmean(img))
    fshift = np.fft.fftshift(f)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq_expected - 15) & (r <= rows*freq_expected + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    
    margin = 30
    valid = np.zeros_like(img, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max & valid)
    return gy, gx

def analyze_half_sam(p_pre, p_post, out_name):
    img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    
    img0 = img0_raw.astype(np.float32) / 65535.0
    img1 = img1_raw.astype(np.float32) / 65535.0
    
    # 1. Align
    dx, dy = get_coarse_shift_phase_correlate(img0, img1)
    print(f"Alignment Shift: dx={dx:.2f}, dy={dy:.2f}")
    
    # Shift img1 to match img0 using Cubic
    rows, cols = img0.shape
    warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    # 2. Extract Pillars
    gy, gx = extract_pillars(img0)
    
    # 3. Extract Intensity
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_aligned, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    
    int0 = np.zeros(len(gy), dtype=np.float32)
    int1 = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            int0 += pad0[gy_p + dy_, gx_p + dx_]
            int1 += pad1[gy_p + dy_, gx_p + dx_]
            
    # 4. Filter Defects
    valid = ~np.isnan(int1) & ~np.isnan(int0)
    
    # 5. Split Top/Bottom (Top: SAM, Bottom: No SAM)
    mid_y = rows // 2
    top_mask = valid & (gy < mid_y)    # Top half
    bot_mask = valid & (gy >= mid_y)   # Bottom half
    
    # Top Correlation & Delta
    if np.sum(top_mask) > 100:
        c_top = np.corrcoef(int0[top_mask], int1[top_mask])[0, 1]
        delta_top = np.mean(int1[top_mask]) - np.mean(int0[top_mask])
    else:
        c_top, delta_top = 0, 0
        
    # Bottom Correlation & Delta
    if np.sum(bot_mask) > 100:
        c_bot = np.corrcoef(int0[bot_mask], int1[bot_mask])[0, 1]
        delta_bot = np.mean(int1[bot_mask]) - np.mean(int0[bot_mask])
    else:
        c_bot, delta_bot = 0, 0
        
    print(f"Top Half (SAM)    - Correlation: {c_top:.4f}, Mean Delta: {delta_top:.4f}, Pillars: {np.sum(top_mask)}")
    print(f"Bottom Half (No SAM) - Correlation: {c_bot:.4f}, Mean Delta: {delta_bot:.4f}, Pillars: {np.sum(bot_mask)}")
    
    # Scatter Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(int0[top_mask], int1[top_mask], alpha=0.1, s=1, color='red')
    plt.title(f"Top (SAM) r={c_top:.3f}")
    plt.xlabel("Pre Intensity")
    plt.ylabel("Post Intensity")
    plt.plot([0, np.max(int0[top_mask])], [0, np.max(int0[top_mask])], 'k--')
    
    plt.subplot(1, 2, 2)
    plt.scatter(int0[bot_mask], int1[bot_mask], alpha=0.1, s=1, color='blue')
    plt.title(f"Bottom (No SAM) r={c_bot:.3f}")
    plt.xlabel("Pre Intensity")
    plt.plot([0, np.max(int0[bot_mask])], [0, np.max(int0[bot_mask])], 'k--')
    
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Saved scatter plot to {out_name}")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        analyze_half_sam(sys.argv[1], sys.argv[2], sys.argv[3])
