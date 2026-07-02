import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def search():
    # Relative path to test_outputs in the workspace
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    
    ref_csv = os.path.join(outputs_dir, "1-0_pillars.csv")
    tgt_csv = os.path.join(outputs_dir, "1-1_pillars.csv")
    
    if not os.path.exists(ref_csv) or not os.path.exists(tgt_csv):
        print("Missing raw pillar CSV files in test_outputs to run search.")
        return
        
    ref_df = pd.read_csv(ref_csv)
    tgt_df = pd.read_csv(tgt_csv)
    
    # Landmark grooves coarse shift is (82.5, 50.5)
    # Filter to ROI for accurate counting and to avoid edge noise
    roi_x_min = 143.6 + 80.0
    roi_y_min = 136.2 + 80.0
    roi_x_max = 2048.0 - 50.0
    roi_y_max = 2044.0 - 50.0
    
    ref_mask = (ref_df['x'] >= roi_x_min) & (ref_df['x'] < roi_x_max) & \
               (ref_df['y'] >= roi_y_min) & (ref_df['y'] < roi_y_max)
    ref_coords = ref_df[ref_mask][['x', 'y']].values
    ref_tree = KDTree(ref_coords)
    
    tgt_coords = tgt_df[['x', 'y']].values
    
    dx_coarse = 82.5
    dy_coarse = 50.5
    
    # Grid search parameters
    dx_range = np.arange(dx_coarse - 8, dx_coarse + 8, 1.0)
    dy_range = np.arange(dy_coarse - 8, dy_coarse + 8, 1.0)
    
    best_dx = dx_coarse
    best_dy = dy_coarse
    best_rate = 0.0
    
    print("--- Searching for optimal coarse shift grid ---")
    print(f"Reference points in ROI: {len(ref_coords)}")
    
    results = []
    
    for dy in dy_range:
        for dx in dx_range:
            # Shift target points
            shifted_tgt = tgt_coords.copy()
            shifted_tgt[:, 0] += dx
            shifted_tgt[:, 1] += dy
            
            # Filter shifted targets to ROI
            tgt_mask = (shifted_tgt[:, 0] >= roi_x_min) & (shifted_tgt[:, 0] < roi_x_max) & \
                       (shifted_tgt[:, 1] >= roi_y_min) & (shifted_tgt[:, 1] < roi_y_max)
            shifted_tgt_roi = shifted_tgt[tgt_mask]
            
            if len(shifted_tgt_roi) == 0:
                continue
                
            # Query KDTree
            distances, _ = ref_tree.query(shifted_tgt_roi, k=1)
            
            # Match rate within 1.0 pixel (tighter than 1.5 to find the sharpest peak)
            matched = np.sum(distances <= 1.0)
            rate = (matched / len(shifted_tgt_roi)) * 100
            
            results.append((dx, dy, rate, len(shifted_tgt_roi)))
            
            if rate > best_rate:
                best_rate = rate
                best_dx = dx
                best_dy = dy
                
    # Sort and print top 15 shifts
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    print("\nTop 15 shifts by match rate (within 1.0 px):")
    print("-----------------------------------------------------")
    print("  dx      |  dy      | Match Rate | Target Points in ROI")
    print("-----------------------------------------------------")
    for r in results_sorted[:15]:
        print(f"  {r[0]:6.1f} |  {r[1]:6.1f} |  {r[2]:5.2f}%    | {r[3]}")
    print("-----------------------------------------------------")
    print(f"Best coarse shift: dx={best_dx:.1f}, dy={best_dy:.1f} with Match Rate={best_rate:.2f}%")

if __name__ == "__main__":
    search()
