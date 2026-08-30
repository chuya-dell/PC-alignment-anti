import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

dir_260602 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260602_位置合わせ\p50_bare"

def get_valley_bg(img):
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 15
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    valleys = img[local_min & mask]
    return np.mean(valleys) if len(valleys) > 0 else np.nan

def extract_fft_grid(img, pitch_px=6.29):
    f = np.fft.fft2(img - np.mean(img))
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 15
    valid = np.zeros_like(img, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    return np.where(local_max & valid)

def get_pillar_sum(img, gy, gx):
    padded = np.pad(img, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    total_sum = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            total_sum += np.sum(padded[gy_p + dy, gx_p + dx])
    return total_sum

def process_aligned_pair(p_pre, p_post):
    img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    
    if img0_raw is None or img1_raw is None: return None
    
    img0 = img0_raw.astype(np.float32)
    img1 = img1_raw.astype(np.float32)
    
    # Calculate shift
    shift, response = cv2.phaseCorrelate(img0, img1)
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    
    # Align images
    h, w = img0.shape
    x_start_0 = max(0, dx)
    y_start_0 = max(0, dy)
    x_end_0 = min(w, w + dx)
    y_end_0 = min(h, h + dy)
    
    x_start_1 = max(0, -dx)
    y_start_1 = max(0, -dy)
    x_end_1 = min(w, w - dx)
    y_end_1 = min(h, h - dy)
    
    img0_crop = img0[y_start_0:y_end_0, x_start_0:x_end_0] / 65535.0
    img1_crop = img1[y_start_1:y_end_1, x_start_1:x_end_1] / 65535.0
    
    bg0 = get_valley_bg(img0_crop)
    bg1 = get_valley_bg(img1_crop)
    
    gy, gx = extract_fft_grid(img0_crop)
    
    sum0 = get_pillar_sum(img0_crop, gy, gx)
    sum1 = get_pillar_sum(img1_crop, gy, gx)
    
    norm_sum0 = sum0 / bg0
    norm_sum1 = sum1 / bg1
    delta = (norm_sum1 - norm_sum0) / norm_sum0 * 100.0
    
    bg_delta = (bg1 - bg0) / bg0 * 100.0
    corr = np.corrcoef(norm_sum0, norm_sum1)[0, 1]
    
    return {
        "Dataset": "260602_aligned",
        "dx": dx,
        "dy": dy,
        "Norm_Pre_mean": np.mean(norm_sum0),
        "Corr": corr,
        "Delta": np.mean(delta),
        "BG_Delta": bg_delta
    }

results = []
for i in range(1, 25):
    p_pre = os.path.join(dir_260602, f"{i}.tif")
    p_post = os.path.join(dir_260602, f"{i+1}.tif")
    if os.path.exists(p_pre) and os.path.exists(p_post):
        res = process_aligned_pair(p_pre, p_post)
        if res:
            results.append(res)
            
df = pd.DataFrame(results)
print("=== 260602 Aligned Results ===")
print("Mean shift (dx, dy):", df["dx"].mean(), df["dy"].mean())
print("Max shift (dx, dy):", df["dx"].abs().max(), df["dy"].abs().max())
print(f"Corr(Pre, Post): {df['Corr'].mean():.4f}")
print(f"Pillar Delta (%): {df['Delta'].mean():.4f}%")
print(f"Background Delta (%): {df['BG_Delta'].mean():.4f}%")

# Visualize the first pair overlay to confirm
p1 = os.path.join(dir_260602, "1.tif")
p2 = os.path.join(dir_260602, "2.tif")
img1 = cv2.imdecode(np.fromfile(p1, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
img2 = cv2.imdecode(np.fromfile(p2, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)

shift, _ = cv2.phaseCorrelate(img1, img2)
dx, dy = int(round(shift[0])), int(round(shift[1]))

img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())

# Make a color overlay (red for pre, green for post)
overlay = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.float32)
overlay[:,:,0] = img1_norm

img2_shifted = ndi.shift(img2_norm, (-dy, -dx))
overlay[:,:,1] = img2_shifted

plt.figure(figsize=(10, 10))
plt.imshow(overlay[500:800, 500:800])
plt.title(f"Overlay Pre(Red) / Post(Green) Shift: dx={dx}, dy={dy}")
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\overlay_260602.png")
