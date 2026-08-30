import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing

dirs = {
    "p200": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200",
    "p100": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260718_p100_sam",
    "p50_260828": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"
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

def get_3x3_pillar_intensities(img, gy, gx):
    padded = np.pad(img, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    res = np.zeros_like(gy_p, dtype=np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            res += padded[gy_p + dy, gx_p + dx]
    return res / 9.0

def process_file(args):
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
    p0 = get_3x3_pillar_intensities(img0, gy, gx)
    p1 = get_3x3_pillar_intensities(img1, gy, gx)
    
    p0_norm = p0 / bg0
    p1_norm = p1 / bg1
    delta_corr = (p1_norm - p0_norm) / p0_norm * 100.0
    
    N = len(delta_corr)
    mean_corr = np.mean(delta_corr)
    integral_corr = mean_corr * N
    
    # Original Neg Grid logic (vs sample 8) done later
    
    return {
        "Dataset": dataset_name,
        "Sample": a,
        "Position": b,
        "N_pillars": N,
        "Mean_Delta": mean_corr,
        "Integral": integral_corr,
        "Delta_Arr": delta_corr
    }

if __name__ == '__main__':
    tasks = []
    for dname, dpath in dirs.items():
        pre_files = glob.glob(os.path.join(dpath, "**", "*-0.tif"), recursive=True)
        for p in pre_files:
            tasks.append((dname, p))
            
    print(f"Processing {len(tasks)} files...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
    
    summary = []
    for dname, group in df.groupby("Dataset"):
        blank_group = group[group["Sample"] == 8]
        if len(blank_group) == 0: continue
        
        all_blank_deltas = np.concatenate(blank_group['Delta_Arr'].values)
        pillar_blank_mean = np.mean(all_blank_deltas)
        pillar_blank_std = np.std(all_blank_deltas)
        threshold_neg_grid = pillar_blank_mean - 3 * pillar_blank_std
        
        # Integral stats for Blank
        blank_integral_mean = blank_group["Integral"].mean()
        blank_integral_std = blank_group["Integral"].std()
        
        plt.figure(figsize=(10, 5))
        plot_data_integral = []
        plot_data_neggrid = []
        samples = sorted(group["Sample"].unique())
        
        for sample in samples:
            s_group = group[group["Sample"] == sample]
            
            # 1. Integral Method
            s_integrals = s_group["Integral"].values
            s_int_mean = np.mean(s_integrals)
            z_score_int = (s_int_mean - blank_integral_mean) / blank_integral_std if blank_integral_std > 0 else 0
            plot_data_integral.append(s_integrals)
            
            # 2. Neg Grid Method
            s_neg_grids = []
            for arr in s_group['Delta_Arr'].values:
                s_neg_grids.append(np.sum(arr < threshold_neg_grid) / len(arr) * 100.0)
            s_neg_grid_mean = np.mean(s_neg_grids)
            plot_data_neggrid.append(s_neg_grids)
            
            summary.append({
                "Dataset": dname,
                "Sample": sample,
                "Integral Mean": s_int_mean,
                "Integral Z-Score": z_score_int,
                "Neg Grid %": s_neg_grid_mean
            })
            
        plt.subplot(1, 2, 1)
        plt.boxplot(plot_data_integral, labels=samples)
        plt.title(f"{dname}: Integral Response")
        plt.xlabel("Sample (1=1nM, 8=Blank)")
        plt.ylabel("Sum of Deltas")
        
        plt.subplot(1, 2, 2)
        plt.boxplot(plot_data_neggrid, labels=samples)
        plt.title(f"{dname}: Neg Grid %")
        plt.xlabel("Sample (1=1nM, 8=Blank)")
        plt.ylabel("Neg Grid %")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dose_response_{dname}.png"))
        plt.close()
        
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string())
