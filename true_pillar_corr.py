import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

def get_pillar_arrays(p_pre, p_post):
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if img0 is None or img1 is None: return None
    
    img0 = img0.astype(np.float32)
    img1 = img1.astype(np.float32)
    
    shift, _ = cv2.phaseCorrelate(img0, img1)
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    
    # Do not crop, just shift img1 back to match img0's coordinates
    img1_shifted = ndi.shift(img1, (-dy, -dx), order=1)
    
    img0 /= 65535.0
    img1_shifted /= 65535.0
    
    # FFT Grid on img0
    f = np.fft.fft2(img0 - np.mean(img0))
    fshift = np.fft.fftshift(f)
    rows, cols = img0.shape
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / 6.29
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 20
    valid = np.zeros_like(img0, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max & valid)
    
    # Get 3x3 sum for each pillar
    pad0 = np.pad(img0, 1, mode='reflect')
    pad1 = np.pad(img1_shifted, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    
    sum0 = np.zeros(len(gy), dtype=np.float32)
    sum1 = np.zeros(len(gy), dtype=np.float32)
    
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum0 += pad0[gy_p + dy_, gx_p + dx_]
            sum1 += pad1[gy_p + dy_, gx_p + dx_]
            
    # Calculate background (valley)
    local_min = ndi.minimum_filter(img0, size=5) == img0
    bg0 = np.mean(img0[local_min & valid])
    bg1 = np.mean(img1_shifted[local_min & valid])
    
    return sum0/bg0, sum1/bg1, dx, dy

print("Processing 260704 (SAM, Optical)...")
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif"
p0, p1, dx, dy = get_pillar_arrays(p_pre, p_post)
corr_704 = np.corrcoef(p0, p1)[0, 1]
print(f"260704 Single FOV Pillar-by-Pillar Corr: {corr_704:.4f}, Shift: dx={dx}, dy={dy}")

print("Processing 260602 (Bare, Optical)...")
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260602_位置合わせ\p50_bare\1.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260602_位置合わせ\p50_bare\2.tif"
p0, p1, dx, dy = get_pillar_arrays(p_pre, p_post)
corr_602 = np.corrcoef(p0, p1)[0, 1]
print(f"260602 Single FOV Pillar-by-Pillar Corr: {corr_602:.4f}, Shift: dx={dx}, dy={dy}")

    pass
print(f"p50 Full Exp Single FOV Pillar-by-Pillar Corr: {corr_828:.4f}, Shift: dx={dx}, dy={dy}")
