import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import concurrent.futures
import pandas as pd
import scipy.stats as stats

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200'

def process_fov(args):
    s, f = args
    p0 = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p1 = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p0) or not os.path.exists(p1): return None
    img0 = cv2.imdecode(np.fromfile(p0, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)/65535.0
    img1 = cv2.imdecode(np.fromfile(p1, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)/65535.0
    res = cv2.matchTemplate(img1, img0[300:700,300:700], cv2.TM_CCOEFF_NORMED)
    _,_,_,ml = cv2.minMaxLoc(res)
    dx, dy = ml[0]-300, ml[1]-300
    warp = np.float32([[1,0,dx],[0,1,dy]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255,0,255).astype(np.uint8), np.clip(img1*255,0,255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (img0.shape[1], img0.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    r = np.sqrt(np.ogrid[-img0.shape[0]//2:img0.shape[0]-img0.shape[0]//2, -img0.shape[1]//2:img0.shape[1]-img0.shape[1]//2][0]**2 + np.ogrid[-img0.shape[0]//2:img0.shape[0]-img0.shape[0]//2, -img0.shape[1]//2:img0.shape[1]-img0.shape[1]//2][1]**2)
    freq = 1.0/6.38
    img_filt = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * ((r >= img0.shape[0]*freq-15) & (r <= img0.shape[0]*freq+15)))).real
    
    lmax = ndi.maximum_filter(img_filt, size=5) == img_filt
    lmin = ndi.minimum_filter(img_filt, size=5) == img_filt
    
    d0 = np.nan_to_num(img0, nan=np.nanmean(img0))
    diff0 = d0 - cv2.GaussianBlur(d0, (51,51), 0)
    m0 = cv2.dilate((np.abs(diff0) > 3*np.std(diff0[~np.isnan(img0)])).astype(np.uint8), np.ones((7,7), np.uint8))
    
    def get_d(mask):
        gy, gx = np.where(mask)
        valid = (gy>30) & (gy<img0.shape[0]-30) & (gx>30) & (gx<img0.shape[1]-30) & (m0[gy,gx]==0) & ~np.isnan(img1_al[gy,gx])
        gy, gx = gy[valid], gx[valid]
        if len(gy)==0: return np.nan, 0
        i0, i1 = np.zeros(len(gy)), np.zeros(len(gy))
        p0 = np.pad(img0, 1, constant_values=np.nan)
        p1 = np.pad(img1_al, 1, constant_values=np.nan)
        for dy_ in [-1,0,1]:
            for dx_ in [-1,0,1]:
                i0 += p0[gy+1+dy_, gx+1+dx_]
                i1 += p1[gy+1+dy_, gx+1+dx_]
        corr = np.corrcoef(i0, i1)[0,1] if len(gy)>100 else 0
        return np.mean((i0-i1)/i0*100), corr
    
    p_d, corr = get_d(lmax)
    b_d, _ = get_d(lmin)
    return {'s':s, 'f':f, 'pillar_delta':p_d, 'bg_delta':b_d, 'corr':corr}

if __name__ == '__main__':
    args = [(s, f) for s in range(1, 9) for f in range(1, 9)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_fov, args):
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    df = df[df['corr'] >= 0.60]
    print('\n--- p200 Sample Averages (Corr>=0.60) ---')
    for s in range(1, 9):
        sdf = df[df['s']==s]
        if len(sdf)>0:
            print(f"S{s} | Pillar: {sdf['pillar_delta'].mean():.3f}%, BG: {sdf['bg_delta'].mean():.3f}% (n={len(sdf)})")
        else:
            print(f"S{s} | N/A")
            
    df['log_conc'] = df['s'].map({1:-9,2:-10,3:-11,4:-12,5:-13,6:-14,7:-15,8:-16})
    vc = df.dropna(subset=['log_conc'])
    sc, _, _, pc, _ = stats.linregress(vc['log_conc'], vc['pillar_delta'])
    print(f'\nConc Reg (S1-S8, S8=-16): slope={sc:.4f}, p={pc:.4f}')
