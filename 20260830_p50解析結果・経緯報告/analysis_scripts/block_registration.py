import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"

img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)

rows, cols = img0.shape
grid_n = 5
h, w = rows // grid_n, cols // grid_n

print(f"Dividing into {grid_n}x{grid_n} blocks. Base shift estimate: dx=143, dy=0")

vectors = []

# To avoid boundary issues, we just do template matching of a patch from img0 within a search window in img1
patch_size = min(h, w) - 100
search_radius = 20

plt.figure(figsize=(10, 10))

for r in range(grid_n):
    for c in range(grid_n):
        cy = r * h + h // 2
        cx = c * w + w // 2
        
        # Patch from img0
        y0 = cy - patch_size // 2
        y1 = cy + patch_size // 2
        x0 = cx - patch_size // 2
        x1 = cx + patch_size // 2
        
        patch = img0[y0:y1, x0:x1]
        
        # We know the global shift is around dx=143, dy=0. 
        # So the corresponding center in img1 is cx + 143, cy + 0
        img1_cy = cy + 0
        img1_cx = cx + 143
        
        # Search window in img1
        sy0 = max(0, img1_cy - patch_size//2 - search_radius)
        sy1 = min(rows, img1_cy + patch_size//2 + search_radius)
        sx0 = max(0, img1_cx - patch_size//2 - search_radius)
        sx1 = min(cols, img1_cx + patch_size//2 + search_radius)
        
        search_window = img1[sy0:sy1, sx0:sx1]
        
        if search_window.shape[0] < patch.shape[0] or search_window.shape[1] < patch.shape[1]:
            continue
            
        res = cv2.matchTemplate(search_window, patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # max_loc is (x, y) in search_window
        match_x, match_y = max_loc
        
        # Global coordinate in img1
        global_match_x = sx0 + match_x + patch_size//2
        global_match_y = sy0 + match_y + patch_size//2
        
        # Actual dx, dy
        dx = global_match_x - cx
        dy = global_match_y - cy
        
        vectors.append((cx, cy, dx, dy, max_val))
        
        # Plot vector
        plt.arrow(cx, cy, dx - 143, dy, head_width=20, head_length=20, fc='red', ec='red')
        plt.text(cx, cy, f"({dx}, {dy})\nr={max_val:.2f}", color='blue', fontsize=8)

plt.xlim(0, cols)
plt.ylim(rows, 0) # Invert Y for image coordinates
plt.title("Local Displacement Vectors (Relative to dx=143, dy=0)")
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\block_vectors.png")
print("Saved local displacement vectors to block_vectors.png")

# Check if there is rotation/scaling
# We can fit an affine transform to these points
src_pts = np.array([[cx, cy] for cx, cy, dx, dy, r in vectors], dtype=np.float32)
dst_pts = np.array([[cx + dx, cy + dy] for cx, cy, dx, dy, r in vectors], dtype=np.float32)

for cx, cy, dx, dy, r in vectors:
    print(f"Block at ({cx}, {cy}): dx={dx}, dy={dy}, r={r:.4f}")
angle = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]) * 180.0 / np.pi
print(f"Scale: {scale:.5f}")
print(f"Rotation: {angle:.5f} degrees")
