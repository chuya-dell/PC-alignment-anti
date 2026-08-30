import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load images
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"

img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)

# 2. Extract pillars from img0
import scipy.ndimage as ndi
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
margin = 30
valid = np.zeros_like(img0, dtype=bool)
valid[margin:-margin, margin:-margin] = True
gy, gx = np.where(local_max & valid)

# 3. Calculate intensity for each pillar
def get_pillar_intensities(img, gy, gx):
    pad = np.pad(img, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    sum_int = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum_int += pad[gy_p + dy_, gx_p + dx_]
    return sum_int

# We will use the macroscopic alignment (148, -4)
true_dx = 148
true_dy = -4

# Mask valid overlapping pillars
valid_overlap = (gy + true_dy >= margin) & (gy + true_dy < rows - margin) & (gx + true_dx >= margin) & (gx + true_dx < cols - margin)

gy_v = gy[valid_overlap]
gx_v = gx[valid_overlap]

int0_v = get_pillar_intensities(img0, gy_v, gx_v)
int1_v = get_pillar_intensities(img1, gy_v + true_dy, gx_v + true_dx)

# Calculate Delta (Post - Pre) / Pre
delta = (int1_v - int0_v) / int0_v * 100.0 # Percentage

# Create heatmap
heatmap = np.zeros((rows, cols), dtype=np.float32)
heatmap[gy_v, gx_v] = delta

# Mask out areas with no pillars for better visualization
vis = np.full((rows, cols), np.nan, dtype=np.float32)
vis[gy_v, gx_v] = delta

plt.figure(figsize=(12, 10))
plt.imshow(vis, cmap='coolwarm', vmin=-10, vmax=10)
plt.colorbar(label='Delta (%)')
plt.title('Spatial Heatmap of Pillar Delta (Post - Pre)')
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\delta_heatmap.png")
print("Saved heatmap to delta_heatmap.png")

# Calculate L3 defect masks
def get_defects(img):
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    diff = img - blurred
    std = np.std(diff)
    defects = (np.abs(diff) > 3 * std)
    return defects

defects0 = get_defects(img0)
# Shift img1 back to align with img0 to compare defects in same coordinate space
M = np.float32([[1, 0, -true_dx], [0, 1, -true_dy]])
img1_aligned = cv2.warpAffine(img1, M, (cols, rows))
defects1 = get_defects(img1_aligned)

new_defects = defects1 & ~defects0
removed_defects = defects0 & ~defects1
common_defects = defects0 & defects1

print(f"Defect Area (Pre): {np.sum(defects0)} px")
print(f"Defect Area (Post): {np.sum(defects1)} px")
print(f"New Defects (Post only): {np.sum(new_defects)} px")
print(f"Removed Defects (Pre only): {np.sum(removed_defects)} px")
print(f"Common Defects: {np.sum(common_defects)} px")

# Check correlation outside defects
defect_mask_pillars = defects0[gy_v, gx_v] | defects1[gy_v + true_dy, gx_v + true_dx]
clean_mask = ~defect_mask_pillars

clean_int0 = int0_v[clean_mask]
clean_int1 = int1_v[clean_mask]

clean_corr = np.corrcoef(clean_int0, clean_int1)[0, 1]
print(f"\nCorrelation of ALL valid pillars: {np.corrcoef(int0_v, int1_v)[0, 1]:.4f}")
print(f"Correlation of CLEAN pillars (excluding defects): {clean_corr:.4f}")
