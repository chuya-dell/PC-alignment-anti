import numpy as np
import glob
import os
import re

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\auto_masks"
masks = glob.glob(os.path.join(out_dir, "*.npy"))

print("--- L3 Mask Pass Rates ---")
dates = {}
for m in masks:
    arr = np.load(m)
    total = arr.size
    passed = np.sum(arr)
    
    match = re.search(r'(\d{6})', os.path.basename(m))
    if match:
        d = match.group(1)
        if d not in dates:
            dates[d] = {"total": 0, "passed": 0}
        dates[d]["total"] += total
        dates[d]["passed"] += passed

for d, v in dates.items():
    rate = (v["passed"] / v["total"]) * 100.0 if v["total"] > 0 else 0
    print(f"Date {d}: Passed {v['passed']:10d} / {v['total']:10d} ({rate:5.2f}%) pixels")
