import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
from scipy.optimize import curve_fit

def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y, C):
    x, y = xy
    return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))) + C

def measure_fwhm(img, gy, gx, box_size=7):
    half_box = box_size // 2
    fwhms = []
    intensities = []
    
    rows, cols = img.shape
    x = np.arange(box_size)
    y = np.arange(box_size)
    x, y = np.meshgrid(x, y)
    xy = (x.ravel(), y.ravel())
    
    for r, c in zip(gy, gx):
        if r - half_box < 0 or r + half_box + 1 > rows or c - half_box < 0 or c + half_box + 1 > cols:
            continue
            
        patch = img[r - half_box:r + half_box + 1, c - half_box:c + half_box + 1]
        
        # Initial guess
        A_guess = np.max(patch) - np.min(patch)
        x0_guess, y0_guess = half_box, half_box
        sigma_guess = 1.5
        C_guess = np.min(patch)
        p0 = [A_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, C_guess]
        
        try:
            popt, _ = curve_fit(gaussian_2d, xy, patch.ravel(), p0=p0, maxfev=400)
            sigma_x, sigma_y = popt[3], popt[4]
            # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
            fwhm_x = 2.3548 * abs(sigma_x)
            fwhm_y = 2.3548 * abs(sigma_y)
            fwhm = (fwhm_x + fwhm_y) / 2.0
            
            if 1.0 < fwhm < 10.0:
                fwhms.append(fwhm)
                intensities.append(np.sum(patch))
        except:
            pass
            
    if len(fwhms) == 0:
        return np.nan, np.nan
        
    return np.median(fwhms), np.mean(intensities)

dir_260602 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260602_位置合わせ\p50_bare"

results = []
# Need to use the same gy, gx for all images so we track the same pillars
img1 = cv2.imdecode(np.fromfile(os.path.join(dir_260602, "1.tif"), dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

# Extract grid
f = np.fft.fft2(img1 - np.mean(img1))
fshift = np.fft.fftshift(f)
freq = 1.0 / 6.29
rows, cols = img1.shape
y, x = np.ogrid[-rows//2:rows-rows//2, -cols//2:cols-cols//2]
r = np.sqrt(x**2 + y**2)
mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
fshift_filtered = fshift * mask
img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
valid = np.zeros_like(img1, dtype=bool)
valid[20:-20, 20:-20] = True
gy, gx = np.where(local_max & valid)

# Limit to 1000 pillars for speed
gy = gy[:1000]
gx = gx[:1000]

base_intensity = None

for i in range(1, 26):
    p = os.path.join(dir_260602, f"{i}.tif")
    if not os.path.exists(p): continue
    
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    fwhm, intensity = measure_fwhm(img, gy, gx)
    
    if i == 1:
        base_intensity = intensity
        
    delta = (intensity - base_intensity) / base_intensity * 100.0 if base_intensity else 0.0
    
    results.append({
        "Image": i,
        "FWHM": fwhm,
        "Intensity": intensity,
        "Delta_vs_1": delta
    })
    
df = pd.DataFrame(results)
print("=== 260602 (p50 bare) Z-Stack Analysis ===")
print(df.to_string(index=False))
