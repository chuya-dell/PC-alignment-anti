import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def analyze_snr(label, img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    # Simple blob extraction to find peaks
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilate thresh to get pillar footprint, then invert for background
    kernel = np.ones((3,3), np.uint8)
    pillar_mask = cv2.dilate(thresh, kernel, iterations=1)
    bg_mask = cv2.bitwise_not(pillar_mask)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    valid_indices = (stats[:, cv2.CC_STAT_AREA] >= 4) & (stats[:, cv2.CC_STAT_AREA] <= 100)
    centroids_valid = centroids[valid_indices]
    xi = np.clip(np.round(centroids_valid[:, 0]).astype(np.int32), 0, img.shape[1] - 1)
    yi = np.clip(np.round(centroids_valid[:, 1]).astype(np.int32), 0, img.shape[0] - 1)
    
    peaks = img[yi, xi]
    bg_pixels = img[bg_mask > 0]
    
    peak_mean = np.mean(peaks)
    bg_mean = np.mean(bg_pixels)
    bg_std = np.std(bg_pixels)
    snr = (peak_mean - bg_mean) / bg_std if bg_std > 0 else 0
    
    # Saturation check
    saturated_pct = np.sum(img > 0.99) / img.size * 100.0
    
    if label == "p100":
        # Save a zoomed image with centroids
        plt.figure(figsize=(6, 6))
        img_8u = (np.clip(img * 15, 0, 1) * 255).astype(np.uint8)
        plt.imshow(img_8u[:500, :500], cmap='gray')
        
        # Plot centroids within the zoomed area
        mask_in_view = (xi < 500) & (yi < 500)
        plt.scatter(xi[mask_in_view], yi[mask_in_view], s=2, c='red')
        plt.title('p100 Zoomed Raw Image + Extracted Centroids')
        plt.tight_layout()
        plt.savefig(r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\p100_extraction_check.png', dpi=300)
        plt.close()
        
    return {
        "Condition": label,
        "Pillar Count": len(peaks),
        "Peak Mean": peak_mean,
        "BG Mean": bg_mean,
        "BG Std": bg_std,
        "SNR": snr,
        "Saturated%": saturated_pct
    }

dirs = [
    ("p200", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"),
    ("p100", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1"),
    ("p50", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM")
]

results = []
for label, d in dirs:
    img_path = glob.glob(os.path.join(d, "**", "*-0.tif"), recursive=True)[0]
    res = analyze_snr(label, img_path)
    results.append(res)
    
df = pd.DataFrame(results)
print(df.to_string(index=False))
