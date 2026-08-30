import cv2
import numpy as np
import scipy.ndimage as ndi
import os
import pandas as pd

data_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\同一視野"

def extract_and_compare(p_img):
    img_raw = cv2.imdecode(np.fromfile(p_img, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if img_raw is None: return None
    img = img_raw.astype(np.float32) / 65535.0

    rows, cols = img.shape
    f = np.fft.fft2(img - np.nanmean(img))
    fshift = np.fft.fftshift(f)
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / 6.29
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    
    margin = 30
    valid = np.zeros_like(img, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max & valid)

    pad_img = np.pad(img, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    
    ints = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            ints += pad_img[gy_p + dy_, gx_p + dx_]

    # Use heavily blurred image for local background subtraction
    blurred = cv2.GaussianBlur(np.nan_to_num(img, nan=np.nanmean(img)), (51, 51), 0)
    
    bg_ints = np.zeros(len(gy), dtype=np.float32)
    pad_bg = np.pad(blurred, 1, mode='constant', constant_values=np.nan)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            bg_ints += pad_bg[gy_p + dy_, gx_p + dx_]
            
    # Background subtracted contrast
    contrast = ints - bg_ints
    
    mid_y = rows // 2
    top_mask = gy < mid_y
    bot_mask = gy >= mid_y

    top_contrast = np.mean(contrast[top_mask]) * 65535.0
    bot_contrast = np.mean(contrast[bot_mask]) * 65535.0
    
    top_raw = np.mean(ints[top_mask]) * 65535.0
    bot_raw = np.mean(ints[bot_mask]) * 65535.0
    
    return top_contrast, bot_contrast, top_raw, bot_raw

results = []
for i in range(1, 7):
    p = os.path.join(data_dir, f"{i}.tif")
    if not os.path.exists(p): continue
    res = extract_and_compare(p)
    if res:
        tc, bc, tr, br = res
        results.append({
            'Image': f"{i}.tif",
            'SAM_Contrast': tc,
            'NoSAM_Contrast': bc,
            'SAM_Raw': tr,
            'NoSAM_Raw': br,
            'Contrast_Drop_%': (bc - tc) / bc * 100 if bc != 0 else 0
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
