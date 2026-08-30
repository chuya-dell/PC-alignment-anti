import os
import cv2
import numpy as np

p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"

img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)

# Use a central patch of img0
h, w = img0.shape
patch_size = 1000
cy, cx = h//2, w//2
patch = img0[cy-patch_size//2:cy+patch_size//2, cx-patch_size//2:cx+patch_size//2]

res = cv2.matchTemplate(img1, patch, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# max_loc is (x, y) top-left corner of the matched region in img1
# The original top-left corner in img0 was (cx-patch_size//2, cy-patch_size//2)
orig_x = cx - patch_size//2
orig_y = cy - patch_size//2

match_x, match_y = max_loc
dx = match_x - orig_x
dy = match_y - orig_y

print(f"Template Matching Max Corr: {max_val:.4f} at dx={dx}, dy={dy}")
