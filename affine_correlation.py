import os
import cv2
import numpy as np
import scipy.ndimage as ndi

p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif"

img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
rows, cols = img0.shape

# Run ECC directly initialized at true peak
warp_matrix = np.float32([[1, 0, 5.5], [0, 1, -20.5]])
# Skip ECC to avoid SSD local minimum on background gradient
print("Skipping ECC, using exact translation dx=5.5, dy=-20.5")
# 2. Extract pillars from img0
f = np.fft.fft2(img0 - np.nanmean(img0))
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
valid_border = np.zeros_like(img0, dtype=bool)
valid_border[margin:-margin, margin:-margin] = True
gy, gx = np.where(local_max & valid_border)

# 3. Calculate intensity for each pillar
def get_pillar_intensities(img, gy, gx):
    pad = np.pad(img, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    sum_int = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum_int += pad[gy_p + dy_, gx_p + dx_]
    return sum_int

# Instead of warpAffine, we map coordinates directly and round to nearest integer to avoid interpolation blur
gy_flat = gy.astype(np.float32)
gx_flat = gx.astype(np.float32)
ones = np.ones_like(gy_flat)
pts0 = np.vstack([gx_flat, gy_flat, ones]) # Shape: (3, N)

# pts1 = warp_matrix * pts0
pts1 = warp_matrix @ pts0 # Shape: (2, N)
gx1 = np.round(pts1[0, :]).astype(int)
gy1 = np.round(pts1[1, :]).astype(int)

# Filter valid overlapping
valid = (gy1 >= margin) & (gy1 < rows - margin) & (gx1 >= margin) & (gx1 < cols - margin)

gy0_v = gy[valid]
gx0_v = gx[valid]
gy1_v = gy1[valid]
gx1_v = gx1[valid]

int0_v = get_pillar_intensities(img0, gy0_v, gx0_v)
int1_v = get_pillar_intensities(img1, gy1_v, gx1_v)

print(f"Total valid overlapping pillars after precise Affine: {np.sum(valid)}")
print(f"Correlation of all valid pillars (Affine coordinate mapping): {np.corrcoef(int0_v, int1_v)[0, 1]:.4f}")

# 4. Filter out defects
def get_defects(img):
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    diff = img - blurred
    std = np.std(diff)
    return (np.abs(diff) > 3 * std)

defects0 = get_defects(img0)
defects1 = get_defects(img1)

defect_mask_pillars = defects0[gy0_v, gx0_v] | defects1[gy1_v, gx1_v]

clean_int0 = int0_v[~defect_mask_pillars]
clean_int1 = int1_v[~defect_mask_pillars]

print(f"Correlation of CLEAN pillars (Affine coordinate mapping): {np.corrcoef(clean_int0, clean_int1)[0, 1]:.4f}")
