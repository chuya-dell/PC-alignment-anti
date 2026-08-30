import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

p50_dirs = [
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260824_p50_SHC6OH\Raw_Images_データのみ",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260825_p50_SHC6OH\Raw_Images_データのみ",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260826_p50_SHC6OH\Raw_Images_データのみ",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260827_p50_SHC6OH\Raw_Images_データのみ",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260829-p50-sam",
    r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260829_p50_DNA"
]

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

results = []
for d in p50_dirs:
    pre_files = glob.glob(os.path.join(d, "**", "*-0.tif"), recursive=True)
    for pre_path in pre_files:
        try:
            img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            pm, vm, contrast, snr = get_valley_snr(img)
            results.append({
                "Dir": os.path.basename(d),
                "Path": pre_path,
                "SNR": snr
            })
        except:
            pass

df = pd.DataFrame(results)
print(f"--- SNR Distribution for ALL p50 ({len(df)} FOVs) ---")
print(df["SNR"].describe())
