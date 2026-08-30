import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_valley_snr(img):
    local_max = ndi.maximum_filter(img, size=5) == img
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 10
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    peaks = img[local_max & mask]
    valleys = img[local_min & mask]
    if len(peaks) < 100 or len(valleys) < 100: return 0
    return (np.mean(peaks) - np.mean(valleys)) / np.std(valleys) if np.std(valleys) > 0 else 0

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

def process_file(p):
    base = os.path.basename(p)
    a, b, c = map(int, base.replace('.tif', '').split('-'))
    post_p = p.replace('-0.tif', '-1.tif')
    if not os.path.exists(post_p): return None
    
    img0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(post_p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    snr = get_valley_snr(img0)
    gy, gx = extract_fft_grid(img0)
    delta = np.mean((img1[gy, gx] - img0[gy, gx]) / img0[gy, gx] * 100)
    
    group = "Cross 1 (1-4)" if b <= 4 else "Cross 2 (5-8)"
    return {"Sample": a, "Position": b, "Group": group, "SNR": snr, "Delta": delta}

if __name__ == '__main__':
    pre_files = glob.glob(os.path.join(p50_dir, "**", "*-0.tif"), recursive=True)
    results = []
    
    print(f"Processing {len(pre_files)} files with {multiprocessing.cpu_count()} CPUs...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file, pre_files):
            if res:
                results.append(res)

    df = pd.DataFrame(results)

    # Plot Sample vs Delta using matplotlib boxplot
    plt.figure(figsize=(10, 6))
    groups = df.groupby('Sample')
    
    # Extract data for boxplot
    data_to_plot = [group['Delta'].values for name, group in groups]
    positions = [name for name, group in groups]
    
    plt.boxplot(data_to_plot, positions=positions, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
                
    # Overlay scatter points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(positions[i], 0.04, size=len(data))
        plt.scatter(x, data, color='black', alpha=0.6, s=20)

    plt.title('Sample Number (Imaging Order) vs Delta (%)')
    plt.xlabel('Sample Number (1=1nM, 8=Blank)')
    plt.ylabel('Mean Delta (%) [FFT Grid]')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    out_path = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\drift_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved drift plot to {out_path}")
