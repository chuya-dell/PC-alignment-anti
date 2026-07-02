import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def evaluate_roi():
    # Relative path to test_outputs in the workspace
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    
    # Load reference pillars
    ref_csv = os.path.join(outputs_dir, "1-0_pillars.csv")
    if not os.path.exists(ref_csv):
        print(f"Reference file not found: {ref_csv}")
        return
        
    ref_df = pd.read_csv(ref_csv)
    
    # Define ROI based on physical landmark grooves
    # Reference grooves are at X=143.6, Y=136.2.
    # We add 80px margin to avoid the groove region itself which has scratched/missing pillars.
    roi_x_min = 143.6 + 80.0
    roi_y_min = 136.2 + 80.0
    roi_x_max = 2048.0 - 50.0
    roi_y_max = 2044.0 - 50.0
    
    # Filter reference points to ROI
    ref_roi_mask = (ref_df['x'] >= roi_x_min) & (ref_df['x'] < roi_x_max) & \
                   (ref_df['y'] >= roi_y_min) & (ref_df['y'] < roi_y_max)
    ref_roi_df = ref_df[ref_roi_mask]
    ref_roi_coords = ref_roi_df[['x', 'y']].values
    
    print("--- ROI-restricted Alignment Quality Check ---")
    print(f"ROI boundaries: X=[{roi_x_min:.1f}, {roi_x_max:.1f}], Y=[{roi_y_min:.1f}, {roi_y_max:.1f}]")
    print(f"Reference pillars in ROI: {len(ref_roi_coords)}")
    
    targets = ["1-1", "1-2", "1-3"]
    
    for tgt_name in targets:
        aligned_csv = os.path.join(outputs_dir, f"{tgt_name}_aligned_to_1-0.csv")
        if not os.path.exists(aligned_csv):
            print(f"File not found: {aligned_csv}")
            continue
            
        tgt_df = pd.read_csv(aligned_csv)
        
        # Filter aligned target points to the same ROI
        tgt_roi_mask = (tgt_df['x'] >= roi_x_min) & (tgt_df['x'] < roi_x_max) & \
                       (tgt_df['y'] >= roi_y_min) & (tgt_df['y'] < roi_y_max)
        tgt_roi_df = tgt_df[tgt_roi_mask]
        tgt_roi_coords = tgt_roi_df[['x', 'y']].values
        
        # Query closest reference points in ROI
        ref_tree = KDTree(ref_roi_coords)
        distances, indices = ref_tree.query(tgt_roi_coords, k=1)
        
        # Calculate match rates
        matched_15 = np.sum(distances <= 1.5)
        matched_10 = np.sum(distances <= 1.0)
        matched_05 = np.sum(distances <= 0.5)
        
        total_tgt = len(tgt_roi_coords)
        
        print(f"\nTarget: {tgt_name}")
        print(f"  Target pillars in ROI: {total_tgt}")
        if total_tgt > 0:
            print(f"  Match rate (<= 1.5 px): {matched_15}/{total_tgt} ({matched_15/total_tgt*100:.2f}%)")
            print(f"  Match rate (<= 1.0 px): {matched_10}/{total_tgt} ({matched_10/total_tgt*100:.2f}%)")
            print(f"  Match rate (<= 0.5 px): {matched_05}/{total_tgt} ({matched_05/total_tgt*100:.2f}%)")
            print(f"  Mean residual of matches (<= 1.5 px): {np.mean(distances[distances <= 1.5]):.4f} px")
        else:
            print("  No target pillars in ROI.")

if __name__ == "__main__":
    evaluate_roi()
