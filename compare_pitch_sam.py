# -*- coding: utf-8 -*-
import cv2, numpy as np, os, glob
import matplotlib.pyplot as plt
import pandas as pd

base_dir = u'G:\\マイドライブ\\1.実験データ_gdrive\\4.生データ\\4.生データ D'
out_png = u'C:\\Users\\chuya\\.gemini\\antigravity\\brain\\60f5a68e-8281-4136-8936-9e4e95854572\\pitch_snr_comparison.png'

datasets = {
    'p50': os.path.join(base_dir, u'260706_sam_p50'),
    'p100': os.path.join(base_dir, u'260707_sam_p100_1'),
    'p200': os.path.join(base_dir, u'260706_sam_p200')
}

results = []

for pitch, d in datasets.items():
    if not os.path.exists(d):
        continue
    tifs = glob.glob(os.path.join(d, '*.tif'))
    for path in tifs:
        with open(path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
        try:
            img = cv2.imdecode(img_array, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.
        except:
            continue
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
        bl = cv2.GaussianBlur(th, (7,7), 0)
        dil = cv2.dilate(bl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
        bg_mask = bl < np.percentile(bl, 90)
        bg_std = np.std(img[bg_mask])
        bg_mean = np.mean(img[bg_mask])
        bl_pos = bl[bl>0]
        if len(bl_pos) == 0: continue
        tv = np.percentile(bl_pos, 99.9)
        peaks = (bl == dil) & (bl >= tv)
        npk, _, _, cent = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
        num_pillars = npk - 1
        if num_pillars > 0:
            xi = np.clip(np.round(cent[1:,0]).astype(int), 0, img.shape[1]-1)
            yi = np.clip(np.round(cent[1:,1]).astype(int), 0, img.shape[0]-1)
            peak_mean = np.mean(img[yi, xi])
            snr = (peak_mean - bg_mean) / bg_std if bg_std > 0 else 0
        else:
            snr = 0
        results.append({'Pitch': pitch, 'SNR': snr, 'NumPillars': num_pillars})

df = pd.DataFrame(results)
if len(df) == 0:
    print('No data found!')
    exit(1)

for pitch in ['p50', 'p100', 'p200']:
    sub = df[df['Pitch'] == pitch]
    if len(sub) == 0: continue
    print(f'\n{pitch} Statistics (n={len(sub)}):')
    print(f'  SNR Mean: {sub["SNR"].mean():.2f}')
    print(f'  SNR Std : {sub["SNR"].std():.2f}')
    print(f'  Avg Pillars Detected: {sub["NumPillars"].mean():.1f}')

plt.figure(figsize=(8, 6))
data_to_plot = [df[df['Pitch'] == p]['SNR'].values for p in ['p50', 'p100', 'p200'] if len(df[df['Pitch'] == p]) > 0]
labels = [p for p in ['p50', 'p100', 'p200'] if len(df[df['Pitch'] == p]) > 0]
plt.boxplot(data_to_plot, labels=labels)
for i, d in enumerate(data_to_plot):
    x = np.random.normal(i + 1, 0.04, size=len(d))
    plt.plot(x, d, 'r.', alpha=0.3)

plt.title('SNR Comparison Across Pitches (SAM Assay)')
plt.ylabel('Signal-to-Noise Ratio (SNR)')
plt.xlabel('Pillar Pitch')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(out_png, dpi=150)
print('Plot saved.')
