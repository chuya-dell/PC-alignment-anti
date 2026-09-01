import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import concurrent.futures

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

def get_pitch(img):
    rows, cols = img.shape
    f = np.fft.fft2(img - np.nanmean(img))
    power_masked = np.abs(np.fft.fftshift(f))**2
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    power_masked[x**2 + y**2 <= 100**2] = 0
    max_idx = np.unravel_index(np.argmax(power_masked), power_masked.shape)
    r = np.sqrt((max_idx[0]-crow)**2 + (max_idx[1]-ccol)**2)
    return rows / r

def process_fov_heatmap(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre): return None
    
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape
    
    pitch = get_pitch(img0)
    
    patch = img0[rows//2-500:rows//2+500, cols//2-500:cols//2+500]
    res = cv2.matchTemplate(img1[max(0,rows//2-515):min(rows,rows//2+515), max(0,cols//2-515):min(cols,cols//2+515)], patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    warp = np.float32([[1, 0, max_loc[0]-15], [0, 1, max_loc[1]-15]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255,0,255).astype(np.uint8), np.clip(img1*255,0,255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    y, x = np.ogrid[-rows//2:rows-rows//2, -cols//2:cols-cols//2]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*(1/pitch) - 15) & (r <= rows*(1/pitch) + 15)
    img_filt = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * mask)).real
    local_max = ndi.maximum_filter(img_filt, size=5) == img_filt
    
    gy, gx = np.where(local_max)
    # Simple margin filter
    valid = (gy > 30) & (gy < rows-30) & (gx > 30) & (gx < cols-30) & ~np.isnan(img1_al[gy, gx])
    gy, gx = gy[valid], gx[valid]
    
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_al, 1, mode='constant', constant_values=np.nan)
    int0, int1 = np.zeros(len(gy), dtype=np.float32), np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            int0 += pad0[gy + 1 + dy_, gx + 1 + dx_]
            int1 += pad1[gy + 1 + dy_, gx + 1 + dx_]
            
    corr = np.corrcoef(int0, int1)[0, 1] if len(gy) > 100 else 0
    if corr < 0.60: return None
    
    delta = (int0 - int1) / int0 * 100
    
    # 2D Spatial Binning (32x32 bins)
    bins = 32
    heatmap = np.zeros((bins, bins), dtype=np.float32)
    counts = np.zeros((bins, bins), dtype=np.float32)
    
    y_bins = np.digitize(gy, np.linspace(0, rows, bins+1)) - 1
    x_bins = np.digitize(gx, np.linspace(0, cols, bins+1)) - 1
    
    np.add.at(heatmap, (y_bins, x_bins), delta)
    np.add.at(counts, (y_bins, x_bins), 1)
    
    with np.errstate(invalid='ignore'):
        heatmap = heatmap / counts
    return heatmap

if __name__ == '__main__':
    args = [(s, f) for s in range(1, 9) for f in range(1, 9)]
    heatmaps = {s: [] for s in range(1, 9)}
    
    print("Generating Heatmaps (Parallel)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_fov_heatmap, [a[0] for a in args], [a[1] for a in args])
        
    for arg, res in zip(args, results):
        if res is not None:
            heatmaps[arg[0]].append(res)
            
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for s in range(1, 9):
        ax = axes[s-1]
        if len(heatmaps[s]) > 0:
            mean_hm = np.nanmean(np.stack(heatmaps[s]), axis=0)
            im = ax.imshow(mean_hm, cmap='coolwarm', vmin=-1, vmax=4)
            ax.set_title(f'Sample {s} (n={len(heatmaps[s])})')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f'Sample {s} (No Valid FOVs)')
            ax.axis('off')
            
    plt.tight_layout()
    hm_path = os.path.join(out_dir, 'p50_spatial_heatmap.png')
    plt.savefig(hm_path)
    print(f"\nSaved Heatmap to {hm_path}")
