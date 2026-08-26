import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_radial_profile(img_path, cx, cy, max_radius=200):
    """
    Reads a 16-bit TIFF and calculates the radial intensity profile around (cx, cy).
    """
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
        
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    h, w = img.shape
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    radii = np.arange(0, max_radius, 1)
    mean_intensities = []
    
    # Calculate global background ignoring the defect region
    global_bg_mask = dist_from_center > max_radius * 2
    global_bg_mean = np.mean(img[global_bg_mask]) if np.sum(global_bg_mask) > 0 else np.mean(img)
    
    for r in radii:
        mask = (dist_from_center >= r) & (dist_from_center < r + 1)
        if np.sum(mask) > 0:
            mean_intensities.append(np.mean(img[mask]))
        else:
            mean_intensities.append(0)
            
    plt.figure(figsize=(8, 5))
    plt.plot(radii, mean_intensities, label="Radial Profile", color='blue', linewidth=2)
    plt.axhline(global_bg_mean, color='red', linestyle='--', label=f"Global Background ({global_bg_mean:.4f})")
    
    plt.title(f"Stain Radial Intensity Profile\nImage: {os.path.basename(img_path)}\nCenter: ({cx}, {cy})")
    plt.xlabel("Distance from Center (pixels)")
    plt.ylabel("Normalized Intensity (0-1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_png = os.path.join(os.path.dirname(img_path), f"radial_profile_{os.path.basename(img_path)}_{cx}_{cy}.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", f"radial_profile.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    
    print(f"Saved radial profile to {brain_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--r", type=int, default=200)
    args = parser.parse_args()
    get_radial_profile(args.img, args.cx, args.cy, args.r)
