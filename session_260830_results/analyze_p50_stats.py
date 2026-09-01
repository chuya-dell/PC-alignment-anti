import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

def get_coarse_shift(img0, img1):
    rows, cols = img0.shape
    patch_size = 1000
    cy, cx = rows // 2, cols // 2
    patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
    # Restrict search range to +/- 15 pixels as requested by user!
    # Crop img1 around the expected center +/- 15px (+ patch_size/2)
    search_margin = 15
    y1 = cy - patch_size//2 - search_margin
    y2 = cy + patch_size//2 + search_margin
    x1 = cx - patch_size//2 - search_margin
    x2 = cx + patch_size//2 + search_margin
    
    # Boundary checks
    y1, y2 = max(0, y1), min(rows, y2)
    x1, x2 = max(0, x1), min(cols, x2)
    
    img1_search = img1[y1:y2, x1:x2]
    res = cv2.matchTemplate(img1_search, patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    match_x, match_y = max_loc
    # offset in the cropped region
    dx = match_x - search_margin
    dy = match_y - search_margin
    return dx, dy

def extract_defects(img):
    img_filled = np.nan_to_num(img, nan=np.nanmean(img))
    blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
    diff = img_filled - blurred
    std = np.std(diff[~np.isnan(img)])
    # user specified: defect mask 3px dilate
    mask_bin = (np.abs(diff) > 3 * std).astype(np.uint8)
    mask_dilated = cv2.dilate(mask_bin, np.ones((7, 7), np.uint8))
    return mask_dilated.astype(bool)

def get_ints(img, gy, gx):
    pad = np.pad(img, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    sum_int = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum_int += pad[gy_p + dy_, gx_p + dx_]
    return sum_int

results = []
all_deltas = {}
fov_diagnostics = []

print("Processing p50 dataset...")
for s in range(1, 9):
    for f in range(1, 9):
        p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
        p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
        if not os.path.exists(p_pre): continue
        
        img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        img0 = img0_raw.astype(np.float32) / 65535.0
        img1 = img1_raw.astype(np.float32) / 65535.0
        rows, cols = img0.shape
        
        # 1. Alignment (Standard pipeline: coarse +/-15px -> ECC -> Cubic)
        dx_c, dy_c = get_coarse_shift(img0, img1)
        warp_matrix = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
        img0_8u = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
        img1_8u = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
        
        try:
            _, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
        except cv2.error:
            pass
            
        img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        
        # 2. Pillar Extraction (Grid Point Method via FFT)
        f_fft = np.fft.fft2(img0 - np.nanmean(img0))
        fshift = np.fft.fftshift(f_fft)
        crow, ccol = rows // 2, cols // 2
        freq = 1.0 / 6.29
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        r = np.sqrt(x**2 + y**2)
        fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
        fshift_filtered = fshift * fft_mask
        img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
        
        # Grid generation directly from the 2D standing wave
        local_max = ndi.maximum_filter(img_filtered, size=5) == img_filtered
        margin = 30
        valid_border = np.zeros_like(img0, dtype=bool)
        valid_border[margin:-margin, margin:-margin] = True
        
        # Stage 1: Extracted Grid points
        gy, gx = np.where(local_max & valid_border)
        stage1_count = len(gy)
        
        int0 = get_ints(img0, gy, gx)
        int1 = get_ints(img1_aligned, gy, gx)
        
        # Stage 2: Defect Exclusion
        defects0 = extract_defects(img0)
        defects1 = extract_defects(img1_aligned)
        defect_mask_pillars = defects0[gy, gx] | defects1[gy, gx]
        
        valid_pixel_mask = ~np.isnan(int1) & ~np.isnan(int0)
        final_valid_mask = valid_pixel_mask & ~defect_mask_pillars
        stage2_count = np.sum(final_valid_mask)
        
        # Intensity Bias check for dropped pillars
        dropped_mask = valid_pixel_mask & defect_mask_pillars
        mean_int_kept = np.mean(int0[final_valid_mask]) if stage2_count > 0 else 0
        mean_int_dropped = np.mean(int0[dropped_mask]) if np.sum(dropped_mask) > 0 else 0
        
        # Correlation
        int0_clean = int0[final_valid_mask]
        int1_clean = int1[final_valid_mask]
        if stage2_count > 100:
            corr = np.corrcoef(int0_clean, int1_clean)[0, 1]
        else:
            corr = 0
            
        # Delta = (pre - post) / pre * 100
        delta_pct = (int0_clean - int1_clean) / int0_clean * 100
        
        all_deltas[(s, f)] = delta_pct
        
        fov_diagnostics.append({
            'Sample': s, 'FOV': f, 
            'Theoretical_Max': (rows-2*margin)*(cols-2*margin) / (6.29**2),
            'Stage1_Grid_Points': stage1_count,
            'Stage2_Valid_Pillars': stage2_count,
            'Mean_Int_Kept': mean_int_kept,
            'Mean_Int_Dropped': mean_int_dropped,
            'Correlation': corr
        })

df_diag = pd.DataFrame(fov_diagnostics)
print("\n--- Pillar Extraction Statistics ---")
print(df_diag.groupby('Sample').agg({
    'Stage1_Grid_Points': 'mean', 
    'Stage2_Valid_Pillars': 'mean',
    'Correlation': 'mean'
}))

# Check bias
kept_mean = df_diag['Mean_Int_Kept'].mean()
dropped_mean = df_diag['Mean_Int_Dropped'].mean()
print(f"\nIntensity Bias Check: Kept average ADU = {kept_mean*65535:.0f}, Dropped average ADU = {dropped_mean*65535:.0f}")

# Filter outliers based on correlation relative to the dataset
all_corrs = df_diag['Correlation'].values
median_corr = np.median(all_corrs)
std_corr = np.std(all_corrs)
thresh_corr = 0.6 # Arbitrary safety threshold, typically standard pipeline yields >0.65
filtered_keys = []
for idx, row in df_diag.iterrows():
    if row['Correlation'] < thresh_corr:
        print(f"DROPPED FOV: S{int(row['Sample'])} F{int(row['FOV'])} (Corr: {row['Correlation']:.4f})")
    else:
        filtered_keys.append((int(row['Sample']), int(row['FOV'])))

# Blank Stats (Sample 8)
blank_deltas = np.concatenate([all_deltas[k] for k in filtered_keys if k[0] == 8])
blank_mean = np.mean(blank_deltas)
blank_std = np.std(blank_deltas)
thresh_val = blank_mean + 3 * blank_std

print("\n--- Blank (S8) Statistics ---")
print(f"Blank Mean: {blank_mean:.4f}%")
print(f"Blank Std (σ): {blank_std:.4f}% (Expected ~0.229%)")
print(f"Threshold (Mean + 3σ): {thresh_val:.4f}%")

# Exceedance Rates
rates = []
for k in filtered_keys:
    s, f = k
    deltas = all_deltas[k]
    if len(deltas) == 0: continue
    rate = np.sum(deltas > thresh_val) / len(deltas) * 100
    rates.append({'Sample': s, 'FOV': f, 'ExceedanceRate_%': rate})

df_rates = pd.DataFrame(rates)
conc_map = {1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None}
df_rates['log_conc'] = df_rates['Sample'].map(conc_map)

summary = df_rates.groupby('Sample').agg({'ExceedanceRate_%': ['mean', 'sem']})
summary.columns = ['mean', 'sem']
print("\n--- Exceedance Rate Summary ---")
print(summary)

# Stats for trend
valid_df = df_rates.dropna(subset=['log_conc'])
slope, intercept, r_val, p_val, std_err = stats.linregress(valid_df['log_conc'], valid_df['ExceedanceRate_%'])
print(f"\nTrend Analysis (S1-S7): Slope = {slope:.4f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print("-> Statistically significant trend detected.")
else:
    print("-> No statistically significant trend detected.")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sam_summary = summary.loc[1:7]
ax.errorbar(sam_summary.index, sam_summary['mean'], yerr=sam_summary['sem'], fmt='o-', color='blue', label='S1-S7 (SAM)', capsize=5)

# Scatter individual FOVs
ax.scatter(valid_df['Sample'], valid_df['ExceedanceRate_%'], color='lightblue', alpha=0.6, zorder=2)

# Blank line
blank_mean_rate = summary.loc[8, 'mean']
blank_sem = summary.loc[8, 'sem']
ax.axhline(blank_mean_rate, color='red', linestyle='--', label=f'Blank (S8) Mean: {blank_mean_rate:.2f}%')
ax.axhspan(blank_mean_rate - blank_sem, blank_mean_rate + blank_sem, color='red', alpha=0.2)

ax.set_xticks(range(1, 9))
ax.set_xticklabels(['1nM(S1)', '100pM(S2)', '10pM(S3)', '1pM(S4)', '100fM(S5)', '10fM(S6)', '1fM(S7)', 'Blank(S8)'])
ax.set_ylabel('Threshold Exceedance Rate (%)')
ax.set_xlabel('Concentration')
ax.set_title('Concentration Dependence (p50 dataset)\nDelta = (Pre - Post) / Pre * 100')
ax.legend()
plt.grid(True)
plot_path = os.path.join(out_dir, 'p50_dose_response.png')
plt.savefig(plot_path)
print(f"\nSaved plot to {plot_path}")
