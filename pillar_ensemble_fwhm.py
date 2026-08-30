import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import concurrent.futures

dirs = {
    "p200": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200",
    "p50": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"
}

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

def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + c

def get_median_fwhm(img, gy, gx, num_samples=1000):
    n_pts = len(gy)
    if n_pts == 0: return np.nan
    idx = np.random.choice(n_pts, min(num_samples, n_pts), replace=False)
    fwhms = []
    
    padded = np.pad(img, 3, mode='reflect')
    for i in idx:
        cy, cx = gy[i] + 3, gx[i] + 3
        patch = padded[cy-3:cy+4, cx-3:cx+4]
        # x-axis projection
        proj = np.sum(patch, axis=0)
        x = np.arange(7)
        try:
            # simple FWHM without fitting
            bg = np.min(proj)
            pk = np.max(proj)
            half = bg + (pk - bg)/2
            # find roots
            crossings = np.where(np.diff(proj > half))[0]
            if len(crossings) >= 2:
                fwhms.append(crossings[-1] - crossings[0])
            else:
                popt, _ = curve_fit(gaussian, x, proj, p0=[pk-bg, 3, 1, bg], maxfev=400)
                sigma = abs(popt[2])
                if sigma < 5:
                    fwhms.append(2.355 * sigma)
        except:
            pass
    return np.median(fwhms) if len(fwhms) > 0 else np.nan

def process_file_fwhm(args):
    dataset_name, p = args
    base = os.path.basename(p)
    a, b, c = map(int, base.replace('.tif', '').split('-'))
    post_p = p.replace('-0.tif', '-1.tif')
    if not os.path.exists(post_p): return None
    
    img0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(post_p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    bg0 = get_valley_bg(img0)
    bg1 = get_valley_bg(img1)
    
    gy, gx = extract_fft_grid(img0)
    
    sum0 = get_pillar_sum(img0, gy, gx)
    sum1 = get_pillar_sum(img1, gy, gx)
    
    norm_sum0 = sum0 / bg0
    norm_sum1 = sum1 / bg1
    delta = (norm_sum1 - norm_sum0) / norm_sum0 * 100.0
    
    fwhm0 = get_median_fwhm(img0, gy, gx)
    fwhm1 = get_median_fwhm(img1, gy, gx)
    
    return {
        "Dataset": dataset_name,
        "Sample": a,
        "Position": b,
        "Delta_Masked": delta,
        "FWHM_Pre": fwhm0,
        "FWHM_Post": fwhm1
    }

if __name__ == '__main__':
    tasks = []
    for dname, dpath in dirs.items():
        pre_files = glob.glob(os.path.join(dpath, "**", "*-0.tif"), recursive=True)
        for p in pre_files:
            tasks.append((dname, p))
            
    print(f"Processing {len(tasks)} tasks (Masked Ensemble & FWHM)...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file_fwhm, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
    
    for dname, group in df.groupby("Dataset"):
        plt.figure(figsize=(15, 5))
        
        # 1. Masked Ensemble Delta
        plt.subplot(1, 3, 1)
        samples = sorted(group["Sample"].unique())
        means = [group[group["Sample"] == s]["Delta_Masked"].mean() for s in samples]
        stds = [group[group["Sample"] == s]["Delta_Masked"].std() for s in samples]
        plt.errorbar(samples, means, yerr=stds, fmt='-o', capsize=5, color='blue')
        plt.title(f"{dname}: Masked Pillar Integral Delta")
        plt.xlabel("Sample (1=1nM, 8=Blank)")
        plt.ylabel("Delta (%)")
        plt.grid(True)
        
        # 2. FWHM Pre vs Sample
        plt.subplot(1, 3, 2)
        fwhm0_means = [group[group["Sample"] == s]["FWHM_Pre"].mean() for s in samples]
        plt.plot(samples, fwhm0_means, '-o', color='green')
        plt.title(f"{dname}: Pre FWHM vs Time")
        plt.xlabel("Sample (Time)")
        plt.ylabel("FWHM (pixels)")
        plt.grid(True)
        
        # 3. FWHM Post vs Sample
        plt.subplot(1, 3, 3)
        fwhm1_means = [group[group["Sample"] == s]["FWHM_Post"].mean() for s in samples]
        plt.plot(samples, fwhm1_means, '-o', color='red')
        plt.title(f"{dname}: Post FWHM vs Time")
        plt.xlabel("Sample (Time)")
        plt.ylabel("FWHM (pixels)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fwhm_masked_ensemble_{dname}.png"))
        plt.close()
        
    print("Done. Masked Ensemble Delta:")
    print(df.groupby(["Dataset", "Sample"])["Delta_Masked"].mean().to_string())
    print("Pre FWHM:")
    print(df.groupby(["Dataset", "Sample"])["FWHM_Pre"].mean().to_string())
