import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

p100_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1"
img_path = glob.glob(os.path.join(p100_dir, "**", "*-0.tif"), recursive=True)[0]
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

# Saturated pixels
saturated_mask = img > 0.99
y_sat, x_sat = np.where(saturated_mask)

plt.figure(figsize=(6, 6))
plt.scatter(x_sat, y_sat, s=0.1, c='red', alpha=0.1)
plt.title(f'p100 Saturated Pixels Spatial Distribution\n(Total: {len(x_sat)} px, {len(x_sat)/img.size*100:.2f}%)', fontname='MS Gothic')
plt.gca().invert_yaxis()
plt.xlim(0, img.shape[1])
plt.ylim(img.shape[0], 0)
plt.tight_layout()
out_path = r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\p100_saturation_map.png'
plt.savefig(out_path, dpi=300)
print(f"Saved saturation map to {out_path}")
