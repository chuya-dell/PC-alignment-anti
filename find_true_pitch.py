import os
import cv2
import numpy as np
import scipy.ndimage as ndi

def get_pitch(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    rows, cols = img.shape
    f = np.fft.fft2(img - np.nanmean(img))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift)**2
    # block out center DC (radius 100)
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x**2 + y**2 > 100**2
    power_masked = power * mask
    # find global max
    max_idx = np.unravel_index(np.argmax(power_masked), power_masked.shape)
    dy, dx = max_idx[0] - crow, max_idx[1] - ccol
    r = np.sqrt(dx**2 + dy**2)
    pitch_px = rows / r
    return pitch_px

data_dir_p50 = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
data_dir_half = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\同一視野'

pitches = []
for s in [1, 4, 8]:
    for f in [1, 5]:
        p = os.path.join(data_dir_p50, f'{s}-{f}-0.tif')
        if os.path.exists(p): pitches.append(('p50', f'S{s}F{f}', get_pitch(p)))

for f in [1, 3, 6]:
    p = os.path.join(data_dir_half, f'{f}.tif')
    if os.path.exists(p): pitches.append(('HalfSAM', f'Img{f}', get_pitch(p)))

print('--- True Pitch Measurement from FFT ---')
for ds, name, p in pitches:
    fov_um = 2048 / p * 0.46
    print(f'{ds} {name}: Pitch = {p:.4f} px, FOV = {fov_um:.2f} um')

mean_pitch = np.mean([p for _, _, p in pitches])
mean_fov = 2048 / mean_pitch * 0.46
print(f'\nMean Pitch: {mean_pitch:.4f} px')
print(f'Calculated FOV: {mean_fov:.2f} um')

