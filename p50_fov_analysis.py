import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_valley_snr(img):
    local_max = ndi.maximum_filter(img, size=5) == img
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 10
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    
    peaks = img[local_max & mask]
    valleys = img[local_min & mask]
    
    if len(peaks) < 100 or len(valleys) < 100:
        return 0, 0, 0, 0
        
    pm = np.mean(peaks)
    vm = np.mean(valleys)
    vstd = np.std(valleys)
    snr = (pm - vm) / vstd if vstd > 0 else 0
    return pm, vm, pm - vm, snr

def extract_fft_grid(img, pitch_px=6.29):
    f = np.fft.fft2(img - np.mean(img))
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    r_target = rows * freq
    mask = (r >= r_target - 15) & (r <= r_target + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 15
    valid_mask = np.zeros_like(img, dtype=bool)
    valid_mask[margin:-margin, margin:-margin] = True
    return np.where(local_max & valid_mask)

results = []
pre_files = glob.glob(os.path.join(p50_dir, "**", "*-0.tif"), recursive=True)

for pre_path in pre_files:
    basename = os.path.basename(pre_path)
    sample_id, pos_id, _ = basename.split('-')
    
    img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    pm, vm, contrast, snr = get_valley_snr(img)
    
    results.append({
        "Sample": int(sample_id),
        "FOV": int(pos_id),
        "Path": pre_path,
        "Peak": pm,
        "BG": vm,
        "Contrast": contrast,
        "SNR": snr
    })

df = pd.DataFrame(results)
print("--- SNR Distribution for p50 (260828) ---")
print(df["SNR"].describe())

# Save high vs low SNR visualization
df_sorted = df.sort_values(by="SNR")
low_fov = df_sorted.iloc[0]
high_fov = df_sorted.iloc[-1]

img_low = cv2.imdecode(np.fromfile(low_fov["Path"], dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
img_high = cv2.imdecode(np.fromfile(high_fov["Path"], dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_low[:500, :500], cmap='gray', vmin=0.4, vmax=0.6)
plt.title(f'Low SNR: {low_fov["SNR"]:.2f} (Sample {low_fov["Sample"]}-{low_fov["FOV"]})')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_high[:500, :500], cmap='gray', vmin=0.4, vmax=0.6)
plt.title(f'High SNR: {high_fov["SNR"]:.2f} (Sample {high_fov["Sample"]}-{high_fov["FOV"]})')
plt.axis('off')
plt.tight_layout()
out_vis = r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\p50_snr_comparison.png'
plt.savefig(out_vis, dpi=300)
print(f"Saved visualization to {out_vis}")

# Filter high SNR (e.g., top 50%)
median_snr = df["SNR"].median()
df_high = df[df["SNR"] >= median_snr]

print("\n--- High SNR FOVs Delta Analysis (FFT Grid) ---")
# Compute Blank vs 1nM delta using ONLY high SNR FOVs
delta_results = []
for _, row in df_high.iterrows():
    if row["Sample"] not in [1, 8]: # 1nM and Blank only
        continue
        
    pre_path = row["Path"]
    post_path = pre_path.replace("-0.tif", "-1.tif")
    if not os.path.exists(post_path):
        continue
        
    pre_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    post_img = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    gy, gx = extract_fft_grid(pre_img)
    delta = (post_img[gy, gx] - pre_img[gy, gx]) * 100.0
    
    delta_results.append({
        "Sample": row["Sample"],
        "FOV": row["FOV"],
        "Delta Mean (%)": np.mean(delta)
    })

df_d = pd.DataFrame(delta_results)
if not df_d.empty:
    blank_fovs = df_d[df_d["Sample"] == 8]["Delta Mean (%)"]
    _1nm_fovs = df_d[df_d["Sample"] == 1]["Delta Mean (%)"]
    
    if len(blank_fovs) > 0 and len(_1nm_fovs) > 0:
        blank_mean = blank_fovs.mean()
        blank_std = blank_fovs.std() if len(blank_fovs) > 1 else 0
        _1nm_mean = _1nm_fovs.mean()
        
        print(f"Blank (High SNR, {len(blank_fovs)} FOVs): Mean = {blank_mean:.4f}%, Std = {blank_std:.4f}%")
        print(f"1nM (High SNR, {len(_1nm_fovs)} FOVs): Mean = {_1nm_mean:.4f}%")
        
        if blank_std > 0:
            z_score = (_1nm_mean - blank_mean) / blank_std
            print(f"Z-Score (1nM vs Blank) = {z_score:.4f}")
        else:
            print("Not enough Blank FOVs for Z-score")
else:
    print("No valid Delta results.")
