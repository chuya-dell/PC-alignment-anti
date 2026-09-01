import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"

img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
rows, cols = img0.shape

# 1. ECC Registration to find exact Affine transform (Translation + Rotation)
# We know the approximate translation is dx=148, dy=-4
warp_matrix = np.float32([[1, 0, 148], [0, 1, -4]])
# Convert to 8-bit for ECC
img0_8u = np.clip(img0 / 256.0, 0, 255).astype(np.uint8)
img1_8u = np.clip(img1 / 256.0, 0, 255).astype(np.uint8)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
try:
    cc, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
    print("ECC Affine Matrix (MOTION_EUCLIDEAN):")
    print(warp_matrix)
    angle = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) * 180.0 / np.pi
    print(f"ECC Rotation: {angle:.5f} degrees")
    print(f"ECC Translation: dx={warp_matrix[0,2]:.2f}, dy={warp_matrix[1,2]:.2f}")
except cv2.error as e:
    print("ECC failed:", e)

# 2. Defect Overlay
def get_defects(img):
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    diff = img - blurred
    std = np.std(diff)
    return (np.abs(diff) > 3 * std)

defects0 = get_defects(img0)
# Use the integer shift for now to be consistent with previous analysis
M_int = np.float32([[1, 0, -148], [0, 1, 4]])
img1_aligned = cv2.warpAffine(img1, M_int, (cols, rows))
defects1 = get_defects(img1_aligned)

overlay = np.zeros((rows, cols, 3), dtype=np.uint8)
overlay[defects0] = [255, 0, 0] # Pre = Red
overlay[defects1 & ~defects0] = [0, 255, 0] # Post only = Green
overlay[defects0 & defects1] = [255, 255, 0] # Both = Yellow

cv2.imwrite(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\defect_overlay.png", overlay)
print("Saved defect_overlay.png")

# Count connected components to see if it's many small dots or a few large areas
num_labels0, labels0, stats0, centroids0 = cv2.connectedComponentsWithStats(defects0.astype(np.uint8))
num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(defects1.astype(np.uint8))
print(f"Pre Defects: {num_labels0-1} distinct components")
print(f"Post Defects: {num_labels1-1} distinct components")
