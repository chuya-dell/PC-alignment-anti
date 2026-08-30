import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_2d_gaussian(patch):
    h, w = patch.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    x, y = np.meshgrid(x, y)
    
    initial_guess = (np.max(patch)-np.min(patch), w/2, h/2, 1.0, 1.0, 0, np.min(patch))
    try:
        popt, _ = opt.curve_fit(gaussian_2d, (x, y), patch.ravel(), p0=initial_guess, maxfev=1000)
        sigma_x, sigma_y = popt[3], popt[4]
        # FWHM = 2.355 * sigma
        fwhm_x = 2.355 * abs(sigma_x)
        fwhm_y = 2.355 * abs(sigma_y)
        return (fwhm_x + fwhm_y) / 2.0
    except:
        return np.nan

def analyze_fwhm(label, img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    valid_indices = (stats[:, cv2.CC_STAT_AREA] >= 4) & (stats[:, cv2.CC_STAT_AREA] <= 100)
    centroids_valid = centroids[valid_indices]
    
    xi = np.round(centroids_valid[:, 0]).astype(np.int32)
    yi = np.round(centroids_valid[:, 1]).astype(np.int32)
    
    # Filter out centroids too close to the edge
    margin = 10
    valid_mask = (xi >= margin) & (xi < img.shape[1] - margin) & (yi >= margin) & (yi < img.shape[0] - margin)
    xi = xi[valid_mask]
    yi = yi[valid_mask]
    
    # Pick a random subset to avoid fitting 20k pillars
    np.random.seed(42)
    if len(xi) > 500:
        idx = np.random.choice(len(xi), 500, replace=False)
        xi = xi[idx]
        yi = yi[idx]
        
    fwhms = []
    avg_profile = np.zeros(2 * margin + 1)
    
    for x, y in zip(xi, yi):
        patch = img[y-margin:y+margin+1, x-margin:x+margin+1]
        fwhm = fit_2d_gaussian(patch)
        if not np.isnan(fwhm) and fwhm < 20: # Sanity check
            fwhms.append(fwhm)
            # Add to 1D profile across the center row of the patch
            avg_profile += patch[margin, :] - np.min(patch)
            
    avg_profile /= len(fwhms)
    # Normalize peak to 1 for overlay plotting later
    if np.max(avg_profile) > 0:
        avg_profile /= np.max(avg_profile)
        
    return {
        "Condition": label,
        "Valid Fits": len(fwhms),
        "Mean FWHM (px)": np.mean(fwhms),
        "Std FWHM (px)": np.std(fwhms),
        "Median FWHM (px)": np.median(fwhms),
        "Profile": avg_profile
    }

dirs = [
    ("p200", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"),
    ("p50", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM")
]

results = []
plt.figure(figsize=(8, 6))
x_axis = np.arange(-10, 11)

for label, d in dirs:
    img_path = glob.glob(os.path.join(d, "**", "*-0.tif"), recursive=True)[0]
    res = analyze_fwhm(label, img_path)
    results.append(res)
    
    plt.plot(x_axis, res["Profile"], label=f'{label} (FWHM={res["Mean FWHM (px)"]:.2f}px)', marker='o')

plt.title('Normalized 1D Cross-section of Averaged Pillars')
plt.xlabel('Distance from centroid (pixels)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\fwhm_profiles.png', dpi=300)

df = pd.DataFrame(results).drop(columns=["Profile"])
print(df.to_string(index=False))
