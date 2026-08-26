# -*- coding: utf-8 -*-
import cv2, numpy as np, os, glob

base_dir = u'G:\\マイドライブ\\1.実験データ_gdrive\\4.生データ\\4.生データ D'

dirs_to_check = {
    'p50 (DNA a)': os.path.join(base_dir, u'260822_p50_dna', u'a'),
    'p100 (SAM)': os.path.join(base_dir, u'260707_sam_p100_1'),
    'p200 (SAM)': os.path.join(base_dir, u'260706_sam_p200')
}

for name, d in dirs_to_check.items():
    tifs = glob.glob(os.path.join(d, '*.tif'))
    if not tifs:
        print(f'{name}: No TIFs found in {d}')
        continue
    
    # Pick the first image
    path = tifs[0]
    # For robust unicode path reading in opencv on windows
    with open(path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.
    
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    bl = cv2.GaussianBlur(th, (7,7), 0)
    dil = cv2.dilate(bl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    
    bg_mask = bl < np.percentile(bl, 90)
    bg_mean = np.mean(img[bg_mask])
    bg_std = np.std(img[bg_mask])
    
    tv = np.percentile(bl[bl>0], 99.9) if len(bl[bl>0])>0 else 0
    peaks = (bl == dil) & (bl >= tv)
    npk, _, _, cent = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    num_pillars = npk - 1
    
    if num_pillars > 0:
        xi, yi = np.clip(np.round(cent[1:,0]).astype(int), 0, img.shape[1]-1), np.clip(np.round(cent[1:,1]).astype(int), 0, img.shape[0]-1)
        peak_vals = img[yi, xi]
        peak_mean = np.mean(peak_vals)
        snr = (peak_mean - bg_mean) / bg_std if bg_std > 0 else 0
    else:
        peak_mean = 0
        snr = 0
        
    print(f'[{name}] file: {os.path.basename(path)}')
    print(f'  - Detected 99.9% Pillars: {num_pillars}')
    print(f'  - Avg Pillar Brightness: {peak_mean:.4f}')
    print(f'  - Background Noise (Std): {bg_std:.4f}')
    print(f'  - SNR (Signal/Noise): {snr:.1f}')
