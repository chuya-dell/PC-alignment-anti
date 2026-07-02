import os
import pandas as pd
import numpy as np

def analyze():
    # Relative path to test_outputs in the workspace
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    
    # Load 1-1 aligned results
    aligned_csv = os.path.join(outputs_dir, "1-1_aligned_to_1-0.csv")
    if not os.path.exists(aligned_csv):
        print(f"Error: {aligned_csv} not found.")
        return
        
    df = pd.read_csv(aligned_csv)
    
    # Separate matched and unmatched
    # Focus only on ROI to avoid boundary/padding effect
    roi_x_min = 143.6 + 80.0
    roi_y_min = 136.2 + 80.0
    roi_x_max = 2048.0 - 50.0
    roi_y_max = 2044.0 - 50.0
    
    roi_mask = (df['x'] >= roi_x_min) & (df['x'] < roi_x_max) & \
               (df['y'] >= roi_y_min) & (df['y'] < roi_y_max)
    df_roi = df[roi_mask]
    
    matched = df_roi[df_roi['matched_ref_id'] != -1]
    unmatched = df_roi[df_roi['matched_ref_id'] == -1]
    
    print("--- Statistical Analysis of Matched vs Unmatched Pillars in ROI ---")
    print(f"Total ROI pillars: {len(df_roi)}")
    print(f"Matched count  : {len(matched)} ({len(matched)/len(df_roi)*100:.2f}%)")
    print(f"Unmatched count: {len(unmatched)} ({len(unmatched)/len(df_roi)*100:.2f}%)")
    
    print("\nMatched Pillars Properties:")
    if not matched.empty:
        print(f"  Mean Intensity: {np.mean(matched['mean_intensity']):.2f} (std={np.std(matched['mean_intensity']):.2f})")
        print(f"  Min/Max Intensity: {np.min(matched['mean_intensity']):.1f} / {np.max(matched['mean_intensity']):.1f}")
        print(f"  Median Intensity: {np.median(matched['mean_intensity']):.2f}")
    else:
        print("  No matched points")
        
    print("\nUnmatched Pillars Properties:")
    if not unmatched.empty:
        print(f"  Mean Intensity: {np.mean(unmatched['mean_intensity']):.2f} (std={np.std(unmatched['mean_intensity']):.2f})")
        print(f"  Min/Max Intensity: {np.min(unmatched['mean_intensity']):.1f} / {np.max(unmatched['mean_intensity']):.1f}")
        print(f"  Median Intensity: {np.median(unmatched['mean_intensity']):.2f}")
    else:
        print("  No unmatched points")

    # Let's check spatial distribution of unmatched points
    # Let's print out the spatial grid (4x4) unmatched rate
    w, h = 2048, 2044
    grid_size_x = w / 4
    grid_size_y = h / 4
    
    print("\nUnmatched Rate (%) by 4x4 Grid Cells:")
    print("--------------------------------------------------")
    for r in range(4):
        row_str = ""
        for c in range(4):
            x_min = c * grid_size_x
            x_max = (c + 1) * grid_size_x
            y_min = r * grid_size_y
            y_max = (r + 1) * grid_size_y
            
            cell_mask = (df['x'] >= x_min) & (df['x'] < x_max) & (df['y'] >= y_min) & (df['y'] < y_max)
            cell_df = df[cell_mask]
            
            cell_unmatched = cell_df[cell_df['matched_ref_id'] == -1]
            unmatched_rate = len(cell_unmatched) / len(cell_df) * 100 if len(cell_df) > 0 else 0
            row_str += f"| {unmatched_rate:5.1f}% ({len(cell_df):5d}) "
        print(row_str + "|")
    print("--------------------------------------------------")

if __name__ == "__main__":
    analyze()
