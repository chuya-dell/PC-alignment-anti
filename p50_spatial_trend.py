import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_valley_snr(img):
    local_max = ndi.maximum_filter(img, size=5) == img
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 10
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    
    peaks = img[local_max & mask]
    valleys = img[local_min & mask]
    
    if len(peaks) < 100 or len(valleys) < 100:
        return 0, 0, 0, 0
        
    pm = np.mean(peaks)
    vm = np.mean(valleys)
    vstd = np.std(valleys)
    snr = (pm - vm) / vstd if vstd > 0 else 0
    return pm, vm, pm - vm, snr

pre_files = glob.glob(os.path.join(p50_dir, "**", "*-0.tif"), recursive=True)
results = []
for pre_path in pre_files:
    basename = os.path.basename(pre_path)
    sample_id, pos_id, _ = basename.split('-')
    
    img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    pm, vm, contrast, snr = get_valley_snr(img)
    results.append({
        "Sample": int(sample_id),
        "FOV": int(pos_id),
        "SNR": snr
    })

df = pd.DataFrame(results)

# Print specific 8-5 FOV
fov85 = df[(df["Sample"] == 8) & (df["FOV"] == 5)]
print("--- FOV 8-5 ---")
print(fov85)

# Create a pivot table to see spatial trends (Sample vs FOV)
pivot = df.pivot(index="Sample", columns="FOV", values="SNR")
print("\n--- SNR Heatmap (Sample vs FOV) ---")
print(pivot.round(2).to_string())

# Trend by Sample (which corresponds to different doses and imaging order)
print("\n--- Mean SNR by Sample (Imaging Order 1 -> 8) ---")
print(df.groupby("Sample")["SNR"].mean().round(2))
