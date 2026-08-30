import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import concurrent.futures

dirs = {
    "p200": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200",
    "p50_260828": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"
}

def get_valley_bg(img):
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 15
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    valleys = img[local_min & mask]
    return np.mean(valleys) if len(valleys) > 0 else np.nan

def process_file_ensemble(args):
    dataset_name, p, block_size = args
    base = os.path.basename(p)
    a, b, c = map(int, base.replace('.tif', '').split('-'))
    post_p = p.replace('-0.tif', '-1.tif')
    if not os.path.exists(post_p): return None
    
    img0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(post_p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    bg0 = get_valley_bg(img0)
    bg1 = get_valley_bg(img1)
    
    norm0 = img0 / bg0
    norm1 = img1 / bg1
    
    h, w = img0.shape
    deltas = []
    
    if block_size == 'full':
        sum0 = np.sum(norm0)
        sum1 = np.sum(norm1)
        deltas.append((sum1 - sum0) / sum0 * 100.0)
    else:
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                chunk0 = norm0[y:y+block_size, x:x+block_size]
                chunk1 = norm1[y:y+block_size, x:x+block_size]
                sum0 = np.sum(chunk0)
                sum1 = np.sum(chunk1)
                if sum0 > 0:
                    deltas.append((sum1 - sum0) / sum0 * 100.0)
    
    return {
        "Dataset": dataset_name,
        "Sample": a,
        "Position": b,
        "BlockSize": block_size,
        "Delta_Mean": np.mean(deltas),
        "Delta_Std": np.std(deltas) if len(deltas) > 1 else 0,
        "N_blocks": len(deltas)
    }

if __name__ == '__main__':
    block_sizes = [63, 315, 'full'] # ~10x10 pillars, ~50x50 pillars, Full FOV
    tasks = []
    
    for dname, dpath in dirs.items():
        pre_files = glob.glob(os.path.join(dpath, "**", "*-0.tif"), recursive=True)
        for p in pre_files:
            for bs in block_sizes:
                tasks.append((dname, p, bs))
                
    results = []
    print(f"Processing {len(tasks)} tasks (ensemble mode)...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file_ensemble, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
    
    for dname, group in df.groupby("Dataset"):
        plt.figure(figsize=(15, 5))
        for i, bs in enumerate(block_sizes):
            plt.subplot(1, 3, i+1)
            bs_group = group[group["BlockSize"] == bs]
            
            samples = sorted(bs_group["Sample"].unique())
            means = [bs_group[bs_group["Sample"] == s]["Delta_Mean"].mean() for s in samples]
            stds = [bs_group[bs_group["Sample"] == s]["Delta_Mean"].std() for s in samples]
            
            plt.errorbar(samples, means, yerr=stds, fmt='-o', capsize=5)
            plt.title(f"Block: {bs} px")
            plt.xlabel("Sample (1=High, 8=Blank)")
            plt.ylabel("Ensemble Delta (%)")
            plt.grid(True)
            
        plt.suptitle(f"{dname}: Ensemble Delta vs Concentration")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ensemble_response_{dname}.png"))
        plt.close()
        
    print(df.groupby(["Dataset", "BlockSize", "Sample"])["Delta_Mean"].mean().to_string())
