import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
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

def process_file_corr(args):
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
    
    bg_delta = (bg1 - bg0) / bg0 * 100.0
    
    return {
        "Dataset": dataset_name,
        "Sample": a,
        "Norm_Pre": norm_sum0,
        "Norm_Post": norm_sum1,
        "Delta": delta,
        "BG_Delta": bg_delta
    }

if __name__ == '__main__':
    tasks = []
    for dname, dpath in dirs.items():
        pre_files = glob.glob(os.path.join(dpath, "**", "*-0.tif"), recursive=True)
        for p in pre_files:
            tasks.append((dname, p))
            
    print(f"Processing {len(tasks)} tasks...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file_corr, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    print("=== Correlation Analysis ===")
    for dname, group in df.groupby("Dataset"):
        corr_pre_post = group["Norm_Pre"].corr(group["Norm_Post"])
        var_pre = group["Norm_Pre"].std() / group["Norm_Pre"].mean() * 100.0 # CV %
        var_delta = group["Delta"].std()
        
        print(f"[{dname}]")
        print(f"  Corr(Pre, Post): {corr_pre_post:.4f}")
        print(f"  CV of Pre (%):   {var_pre:.4f}%")
        print(f"  Std of Delta (%): {var_delta:.4f}%")
        
    print("\n=== Universal Offset Analysis ===")
    print("Overall offset (All conditions):")
    print(df.groupby("Dataset")["Delta"].agg(['mean', 'std', 'min', 'max']).to_string())
    
    print("\nBackground Delta (to check washing impact on flat surfaces):")
    print(df.groupby("Dataset")["BG_Delta"].agg(['mean', 'std']).to_string())
