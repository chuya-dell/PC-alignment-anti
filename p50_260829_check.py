import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

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

fovs = [
    {
        "Label": "260828 SAM (3-1)",
        "Path": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\3-1-0.tif"
    },
    {
        "Label": "260829 SAM (4-1)",
        "Path": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260829-p50-sam\4-1-0.tif"
    },
    {
        "Label": "260829 DNA (1-1)",
        "Path": r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260829_p50_DNA\1-1-0.tif"
    }
]

results = []
imgs = []

for item in fovs:
    img = cv2.imdecode(np.fromfile(item["Path"], dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    pm, vm, contrast, snr = get_valley_snr(img)
    results.append({
        "Label": item["Label"],
        "Peak": pm,
        "BG": vm,
        "Contrast": contrast,
        "SNR": snr
    })
    imgs.append(img)

df = pd.DataFrame(results)
print(df.to_string(index=False))

plt.figure(figsize=(15, 5))
for i, img in enumerate(imgs):
    plt.subplot(1, 3, i+1)
    # Use the same display range as the low SNR one (e.g. 0.4 to 0.6)
    plt.imshow(img[:500, :500], cmap='gray', vmin=0.45, vmax=0.6)
    plt.title(f'{fovs[i]["Label"]}\nSNR: {results[i]["SNR"]:.2f}')
    plt.axis('off')
plt.tight_layout()
plt.savefig(r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\p50_260829_comparison.png', dpi=300)
