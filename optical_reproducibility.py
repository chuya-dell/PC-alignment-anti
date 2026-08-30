import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import concurrent.futures

dir_260704 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df"
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

def process_pair(args):
    dataset_name, p_pre, p_post = args
    if not os.path.exists(p_pre) or not os.path.exists(p_post): return None
    
    img0_dec = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1_dec = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    
    if img0_dec is None or img1_dec is None: return None
    
    img0 = img0_dec.astype(np.float32) / 65535.0
    img1 = img1_dec.astype(np.float32) / 65535.0
    
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
        "Norm_Pre": norm_sum0,
        "Norm_Post": norm_sum1,
        "Delta": delta,
        "BG_Delta": bg_delta
    }

if __name__ == '__main__':
    tasks = []
    
    # 260704
    pre_files_704 = glob.glob(os.path.join(dir_260704, "*-0.tif"))
    for p in pre_files_704:
        tasks.append(("260704_sam_optical", p, p.replace('-0.tif', '-1.tif')))
        
    # 260602
    for i in range(1, 25):
        p_pre = os.path.join(dir_260602, f"{i}.tif")
        p_post = os.path.join(dir_260602, f"{i+1}.tif")
        tasks.append(("260602_p50bare_optical", p_pre, p_post))
        
    print(f"Processing {len(tasks)} optical control pairs...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_pair, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    print("=== Optical Reproducibility Results ===")
    for dname, group in df.groupby("Dataset"):
        corr_pre_post = group["Norm_Pre"].corr(group["Norm_Post"])
        mean_delta = group["Delta"].mean()
        mean_bg = group["BG_Delta"].mean()
        
        print(f"\n[{dname}]")
        print(f"  Corr(Pre, Post): {corr_pre_post:.4f}")
        print(f"  Pillar Delta (%): {mean_delta:.4f}%")
        print(f"  Background Delta (%): {mean_bg:.4f}%")
