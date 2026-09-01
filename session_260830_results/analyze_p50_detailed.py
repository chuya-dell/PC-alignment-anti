import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import concurrent.futures

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

def get_coarse_shift(img0, img1):
    rows, cols = img0.shape
    patch_size = 1000
    cy, cx = rows // 2, cols // 2
    patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
    search_margin = 15
    y1 = max(0, cy - patch_size//2 - search_margin)
    y2 = min(rows, cy + patch_size//2 + search_margin)
    x1 = max(0, cx - patch_size//2 - search_margin)
    x2 = min(cols, cx + patch_size//2 + search_margin)
    img1_search = img1[y1:y2, x1:x2]
    res = cv2.matchTemplate(img1_search, patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc[0] - search_margin, max_loc[1] - search_margin

def extract_defects(img):
    img_filled = np.nan_to_num(img, nan=np.nanmean(img))
    blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
    diff = img_filled - blurred
    std = np.std(diff[~np.isnan(img)])
    mask_bin = (np.abs(diff) > 3 * std).astype(np.uint8)
    return cv2.dilate(mask_bin, np.ones((7, 7), np.uint8)).astype(bool)

def find_grooves(img):
    # Grooves are dark lines. 
    # Vertical groove (1)
    col_means = np.nanmean(img, axis=0)
    row_means = np.nanmean(img, axis=1)
    
    mask = np.zeros(img.shape, dtype=bool)
    
    # Smooth to find wide valleys
    col_smooth = ndi.gaussian_filter1d(col_means, 10)
    row_smooth = ndi.gaussian_filter1d(row_means, 10)
    
    # Threshold for grooves (e.g., bottom 5%)
    c_thresh = np.percentile(col_smooth, 5)
    r_thresh = np.percentile(row_smooth, 5)
    
    mask[:, col_smooth < c_thresh] = True
    mask[row_smooth < r_thresh, :] = True
    
    # Dilate slightly to ensure full coverage
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return mask

def process_fov(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre): return None
    
    img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img0 = img0_raw.astype(np.float32) / 65535.0
    img1 = img1_raw.astype(np.float32) / 65535.0
    rows, cols = img0.shape
    total_area_px = rows * cols
    
    # 1. Alignment
    dx_c, dy_c = get_coarse_shift(img0, img1)
    warp_matrix = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    img0_8u = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
    img1_8u = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
    except: pass
    img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    # 2. Area calculations
    valid_overlap = ~np.isnan(img1_aligned)
    overlap_area_frac = np.sum(valid_overlap) / total_area_px
    
    groove_mask = find_grooves(img0)
    groove_area_frac = np.sum(groove_mask & valid_overlap) / np.sum(valid_overlap)
    
    defects0 = extract_defects(img0)
    defects1 = extract_defects(img1_aligned)
    defect_mask = defects0 | defects1
    defect_area_frac = np.sum(defect_mask & valid_overlap & ~groove_mask) / np.sum(valid_overlap)
    
    # Expected Theoretical Count (Hexagonal array)
    theory_base = 85274
    expected_count = theory_base * overlap_area_frac * (1 - groove_area_frac) * (1 - defect_area_frac)
    
    # 3. FFT Grid Point Extraction
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    fshift = np.fft.fftshift(f_fft)
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / 6.29
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * fft_mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
    
    local_max = ndi.maximum_filter(img_filtered, size=5) == img_filtered
    
    # Margin
    margin = 30
    L3_mask = np.zeros_like(img0, dtype=bool)
    L3_mask[margin:-margin, margin:-margin] = True
    
    gy, gx = np.where(local_max)
    st1_count = len(gy) # After Peak Detection
    
    m_L3 = L3_mask[gy, gx] & valid_overlap[gy, gx]
    gy_L3, gx_L3 = gy[m_L3], gx[m_L3]
    st2_count = len(gy_L3) # After overlap & margin
    
    m_groove = ~groove_mask[gy_L3, gx_L3]
    gy_g, gx_g = gy_L3[m_groove], gx_L3[m_groove]
    st3_count = len(gy_g) # After groove mask
    
    m_def = ~defect_mask[gy_g, gx_g]
    gy_fin, gx_fin = gy_g[m_def], gx_g[m_def]
    st4_count = len(gy_fin) # Final (After Defect Mask)
    
    def get_ints(img, py, px):
        pad = np.pad(img, 1, mode='constant', constant_values=np.nan)
        y_p, x_p = py + 1, px + 1
        sum_int = np.zeros(len(py), dtype=np.float32)
        for dy_ in [-1, 0, 1]:
            for dx_ in [-1, 0, 1]:
                sum_int += pad[y_p + dy_, x_p + dx_]
        return sum_int
        
    int0 = get_ints(img0, gy_fin, gx_fin)
    int1 = get_ints(img1_aligned, gy_fin, gx_fin)
    
    # Intensity Bias check
    # Let's compare dropped vs kept
    dropped_mask_1 = ~L3_mask[gy, gx] | ~valid_overlap[gy, gx]
    gy_drop1, gx_drop1 = gy[dropped_mask_1], gx[dropped_mask_1]
    int_drop1 = get_ints(img0, gy_drop1, gx_drop1)
    
    corr = np.corrcoef(int0, int1)[0, 1] if st4_count > 100 else 0
    delta_pct = (int0 - int1) / int0 * 100
    
    return {
        's': s, 'f': f,
        'Expected_Count': expected_count,
        'St1_Peak': st1_count,
        'St2_Overlap_L3': st2_count,
        'St3_Groove': st3_count,
        'Final_Extracted': st4_count,
        'Extraction_Rate_%': st4_count / expected_count * 100 if expected_count > 0 else 0,
        'Kept_Mean_ADU': np.mean(int0) * 65535.0,
        'Dropped_L3_ADU': np.mean(int_drop1) * 65535.0 if len(int_drop1)>0 else 0,
        'Corr': corr,
        'Deltas': delta_pct
    }

if __name__ == '__main__':
    args = [(s, f) for s in range(1, 9) for f in range(1, 9)]
    results = []
    print("Processing FOVs...")
    for s, f in args:
        res = process_fov(s, f)
        if res: results.append(res)

df = pd.DataFrame(results)
print("\n--- Pillar Extraction Stages (Averages per FOV) ---")
stats_df = df[['Expected_Count', 'St1_Peak', 'St2_Overlap_L3', 'St3_Groove', 'Final_Extracted', 'Extraction_Rate_%']].mean().to_frame('Average Count')
print(stats_df.round(1))

print("\n--- Intensity Bias Check (ADU) ---")
print(f"Kept Pillars: {df['Kept_Mean_ADU'].mean():.0f}")
print(f"Dropped (L3/Overlap): {df['Dropped_L3_ADU'].mean():.0f}")

# Filter out low correlation FOVs
thresh_corr = 0.60
filtered_results = []
print("\n--- Dropped FOVs (Corr < 0.60) ---")
for r in results:
    if r['Corr'] < thresh_corr:
        print(f"S{r['s']} F{r['f']} (Corr: {r['Corr']:.4f})")
    else:
        filtered_results.append(r)

# Blank Stats
blank_deltas = np.concatenate([r['Deltas'] for r in filtered_results if r['s'] == 8])
blank_mean = np.mean(blank_deltas)
blank_std = np.std(blank_deltas)
thresh_val = blank_mean + 3 * blank_std

print("\n--- Blank (S8) Statistics ---")
print(f"Blank Mean: {blank_mean:.4f}%")
print(f"Blank Std (σ): {blank_std:.4f}% (Expected ~0.229%)")
print(f"Threshold (Mean + 3σ): {thresh_val:.4f}%")

# Dose Response
rates = []
for r in filtered_results:
    rate = np.sum(r['Deltas'] > thresh_val) / len(r['Deltas']) * 100
    rates.append({'Sample': r['s'], 'FOV': r['f'], 'ExceedanceRate_%': rate})

df_rates = pd.DataFrame(rates)
conc_map = {1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None}
df_rates['log_conc'] = df_rates['Sample'].map(conc_map)

summary = df_rates.groupby('Sample').agg({'ExceedanceRate_%': ['mean', 'sem']})
summary.columns = ['mean', 'sem']
print("\n--- Exceedance Rate Summary ---")
print(summary.round(3))

valid_df = df_rates.dropna(subset=['log_conc'])
slope, intercept, r_val, p_val, std_err = stats.linregress(valid_df['log_conc'], valid_df['ExceedanceRate_%'])
print(f"\nTrend Analysis (S1-S7): Slope = {slope:.4f}, p-value = {p_val:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
sam_summary = summary.loc[1:7]
ax.errorbar(sam_summary.index, sam_summary['mean'], yerr=sam_summary['sem'], fmt='o-', color='blue', label='S1-S7 (SAM)', capsize=5)
ax.scatter(valid_df['Sample'], valid_df['ExceedanceRate_%'], color='lightblue', alpha=0.6, zorder=2)
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
plot_path = os.path.join(out_dir, 'p50_dose_response_detailed.png')
plt.savefig(plot_path)
print(f"\nSaved plot to {plot_path}")
