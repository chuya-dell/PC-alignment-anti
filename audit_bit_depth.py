import cv2
import numpy as np
import glob
import os
import pandas as pd

def audit_image(filepath):
    try:
        # Read raw bytes to preserve depth
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        if img is None:
            return None
            
        dtype = img.dtype
        min_v = np.min(img)
        max_v = np.max(img)
        p1, p50, p99, p999 = np.percentile(img, [1, 50, 99, 99.9])
        unique_vals = len(np.unique(img))
        
        # Assume 16-bit max is 65535, 12-bit max is 4095, 14-bit max is 16383
        if max_v > 4095:
            sat_val = 65535
        else:
            sat_val = 4095
            
        sat_rate = np.sum(img == sat_val) / img.size * 100
        
        return {
            'file': os.path.basename(filepath),
            'dtype': str(dtype),
            'min': min_v,
            'max': max_v,
            'p1': p1,
            'p50': p50,
            'p99': p99,
            'p99.9': p999,
            'unique_vals': unique_vals,
            'sat_rate_%': sat_rate
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

if __name__ == "__main__":
    test_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\2060824_p50_SHC6OH"
    files = glob.glob(os.path.join(test_dir, "*.tif"))[:5] # Audit 5 files
    
    results = []
    for f in files:
        res = audit_image(f)
        if res:
            results.append(res)
            
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
