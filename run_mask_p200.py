import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image(pre_path, out_dir):
    try:
        base_name = os.path.basename(pre_path)
        dir_name = os.path.basename(os.path.dirname(pre_path))
        if "Raw_Images" in dir_name:
            dir_name = os.path.basename(os.path.dirname(os.path.dirname(pre_path)))
            
        date_prefix = dir_name.split('_')[0][:6] if '_' in dir_name else dir_name.split('-')[0][:6]
        unique_name = f"{date_prefix}_{base_name.replace('-0.tif', '')}"
        npy_path = os.path.join(out_dir, f"{unique_name}_mask.npy")
        
        if os.path.exists(npy_path):
            return
            
        img_data = np.fromfile(pre_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
            
        img_blur = cv2.GaussianBlur(img, (21, 21), 0)
        local_var = cv2.Laplacian(img_blur, cv2.CV_64F).var()
        
        # L3 mask logic
        mean_val = np.mean(img)
        std_val = np.std(img)
        mask = np.abs(img - mean_val) < 3 * std_val
        
        np.save(npy_path, mask)
        
    except Exception as e:
        print(f"Error on {pre_path}: {e}")

d = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"
out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\auto_masks"
files = glob.glob(os.path.join(d, "**", "*-0.tif"), recursive=True)
print(f"Found {len(files)} files to mask.")
for f in files:
    process_image(f, out_dir)
print("Done.")
