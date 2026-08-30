import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
p200_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"

# Task 2: Histogram of Blank with/without Mask
# Find a blank FOV for p200, say 11-1 (which is 0 concentration in intensity_summary_3s_5s.xlsx?)
# Wait, from my logs, 0.0 concentration has FOVs: 4-X or 11-X. Let's find one.
df = pd.read_csv(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv")
blank_sample = df[(df['Date'] == 260706) & (df['Concentration_M'] == 0)]['SampleID'].iloc[0]
print(f"Blank sample for 260706: {blank_sample}")

# Load Pre and Post images for a blank FOV, e.g. {blank_sample}-1
pre_img_path = glob.glob(os.path.join(p200_dir, f"{blank_sample}-1-0.tif"))[0]
post_img_path = glob.glob(os.path.join(p200_dir, f"{blank_sample}-1-1.tif"))[0]
mask_path = os.path.join(out_dir, "auto_masks", f"260706_{blank_sample}-1_mask.npy")

pre_img = cv2.imdecode(np.fromfile(pre_img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
post_img = cv2.imdecode(np.fromfile(post_img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

# V4 Otsu extraction
tophat = cv2.morphologyEx(pre_img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
_, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

valid_indices = (stats[:, cv2.CC_STAT_AREA] >= 4) & (stats[:, cv2.CC_STAT_AREA] <= 100)
centroids_valid = centroids[valid_indices]
xi = np.clip(np.round(centroids_valid[:, 0]).astype(np.int32), 0, pre_img.shape[1] - 1)
yi = np.clip(np.round(centroids_valid[:, 1]).astype(np.int32), 0, pre_img.shape[0] - 1)

pre_ints = pre_img[yi, xi]
post_ints = post_img[yi, xi]
delta_all = (post_ints - pre_ints) * 100.0

mask = np.load(mask_path)
in_mask = mask[yi, xi] # True means defective
valid_l3 = ~in_mask

delta_clean = delta_all[valid_l3]
delta_defects = delta_all[in_mask]

plt.figure(figsize=(8, 6))
plt.hist(delta_all, bins=100, range=(-5, 5), alpha=0.5, label='Mask OFF (All Pillars)', color='gray')
plt.hist(delta_clean, bins=100, range=(-5, 5), alpha=0.7, label='Mask ON (Clean Pillars)', color='blue')
plt.title(f'p200 Blank FOV ({blank_sample}-1) Delta Histogram', fontname='MS Gothic')
plt.xlabel('Delta (%)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'task2_histogram.png'), dpi=300)

plt.figure(figsize=(6, 6))
defect_xi = xi[in_mask]
defect_yi = yi[in_mask]
plt.scatter(xi, yi, s=1, c='gray', alpha=0.1, label='All Pillars')
plt.scatter(defect_xi, defect_yi, s=5, c='red', label='Removed by L3')
plt.title(f'L3 Removed Pillars Spatial Distribution', fontname='MS Gothic')
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'task2_spatial.png'), dpi=300)


# Task 3: Top 0.5% vs All
plt.figure(figsize=(8, 6))
plt.hist(pre_ints, bins=100, range=(0, 0.1), alpha=0.5, label='V4 (All Pillars)', color='gray')
thresh_995 = np.percentile(pre_img, 99.5)
# Original pipeline extracted pixels > thresh_995
top_05_idx = pre_ints > thresh_995
plt.hist(pre_ints[top_05_idx], bins=100, range=(0, 0.1), alpha=0.7, label='Top 0.5% (Orig Pipeline)', color='red')
plt.title('p200 Pre-Intensity Distribution (All vs Top 0.5%)', fontname='MS Gothic')
plt.xlabel('Pre Intensity [0, 1]')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'task3_intensity.png'), dpi=300)

plt.figure(figsize=(8, 6))
plt.hist(delta_all, bins=100, range=(-5, 5), alpha=0.5, label='V4 Delta (All)', density=True, color='gray')
plt.hist(delta_all[top_05_idx], bins=100, range=(-5, 5), alpha=0.7, label='Top 0.5% Delta (Orig Pipeline)', density=True, color='red')
plt.title('p200 Delta Distribution (All vs Top 0.5%) - Normalized', fontname='MS Gothic')
plt.xlabel('Delta (%)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'task3_delta.png'), dpi=300)

print(f"Blank Std (All): {np.std(delta_all):.4f}%")
print(f"Blank Std (Top 0.5%): {np.std(delta_all[top_05_idx]):.4f}%")

# Task 4: FFT for p200 vs p50
p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"
p50_img_path = glob.glob(os.path.join(p50_dir, "**", "3-3-0.tif"), recursive=True)[0]
p50_img = cv2.imdecode(np.fromfile(p50_img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

def plot_fft(img, title, out_path):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    plt.figure(figsize=(6, 6))
    # Zoom in to the center 200x200 pixels of FFT
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    plt.imshow(magnitude_spectrum[cy-100:cy+100, cx-100:cx+100], cmap='gray')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)

plot_fft(pre_img, "FFT: p200 (260706)", os.path.join(out_dir, "task4_fft_p200.png"))
plot_fft(p50_img, "FFT: p50 (260824)", os.path.join(out_dir, "task4_fft_p50.png"))

