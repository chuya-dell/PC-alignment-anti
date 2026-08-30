import os
import glob
import cv2
import numpy as np
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
            
        img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
        h, w = img.shape
        
        bg_mean = cv2.boxFilter(img, -1, (101, 101))
        mean_sq = cv2.boxFilter(img**2, -1, (51, 51))
        sq_mean = cv2.boxFilter(img, -1, (51, 51))**2
        variance = mean_sq - sq_mean
        variance[variance < 0] = 0
        std_dev = np.sqrt(variance)
        median_std = np.median(std_dev)
        
        blurred_for_dust = cv2.GaussianBlur(img, (11, 11), 0)
        dust_mask = (blurred_for_dust > bg_mean + 0.1) | (img > 0.8)
        kernel_dust = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dust_mask = cv2.morphologyEx(dust_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_dust)
        dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_DILATE, kernel_dust)
        
        blurred_for_stain = cv2.GaussianBlur(img, (31, 31), 0)
        global_median_brightness = np.median(bg_mean)
        
        stain_mask_dark = blurred_for_stain < global_median_brightness * 0.5
        stain_mask_flat = std_dev < median_std * 0.4
        stain_mask = stain_mask_dark | stain_mask_flat
        
        kernel_stain = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        stain_mask = cv2.morphologyEx(stain_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_stain)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel_stain)
        
        combined_mask = np.maximum(dust_mask, stain_mask)
        
        np.save(npy_path, combined_mask)
        
        # Generate Preview
        if np.sum(combined_mask) > 0:
            png_path = os.path.join(out_dir, f"{unique_name}_preview.png")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
            axes[0].set_title(unique_name)
            
            overlay = np.zeros((h, w, 3), dtype=np.float32)
            img_disp = np.clip((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5), 0, 1)
            overlay[..., 0] = img_disp
            overlay[..., 1] = img_disp
            overlay[..., 2] = img_disp
            
            overlay[stain_mask > 0] = [1.0, 0.0, 0.0]
            overlay[dust_mask > 0] = [0.0, 0.5, 1.0]
            
            axes[1].imshow(overlay)
            axes[1].set_title(f"Stain(Red): {np.sum(stain_mask)}px, Dust(Blue): {np.sum(dust_mask)}px")
            
            plt.tight_layout()
            plt.savefig(png_path, dpi=100)
            plt.close()
            
            print(f"Processed {unique_name} - Found defects.")
        else:
            print(f"Processed {unique_name} - Clean.")
            
    except Exception as e:
        print(f"Error on {pre_path}: {e}")

if __name__ == "__main__":
    dirs = [
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-dna",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260827_pp50_dna",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260826-p50-sam",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260825_p50_dna",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260824_p50_SHC6OH",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200",
        r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p50",
        r"C:\Users\chuya\.gemini\antigravity\scratch\11_SAM_260706_p200"
    ]
    
    out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\auto_masks"
    os.makedirs(out_dir, exist_ok=True)
    
    for d in dirs:
        print(f"Scanning {d}...")
        files = glob.glob(os.path.join(d, "**", "*-0.tif"), recursive=True)
        for f in files:
            process_image(f, out_dir)
            
    print("Batch processing complete.")
