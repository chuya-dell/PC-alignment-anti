import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_debug_plot():
    # Relative path to test_outputs in the workspace
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    
    ref_csv = os.path.join(outputs_dir, "1-0_pillars.csv")
    tgt_csv = os.path.join(outputs_dir, "1-1_aligned_to_1-0.csv")
    
    if not os.path.exists(ref_csv) or not os.path.exists(tgt_csv):
        print("Missing pillar CSV files in test_outputs to generate plot.")
        return
        
    # Load data
    ref_df = pd.read_csv(ref_csv)
    tgt_df = pd.read_csv(tgt_csv)
    
    # Define ROIs (left, center, right)
    rois = {
        "Left Edge (X: 250-450, Y: 900-1100)": {
            "x_range": (250, 450),
            "y_range": (900, 1100)
        },
        "Center (X: 900-1100, Y: 900-1100)": {
            "x_range": (900, 1100),
            "y_range": (900, 1100)
        },
        "Right Edge (X: 1600-1800, Y: 900-1100)": {
            "x_range": (1600, 1800),
            "y_range": (900, 1100)
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, (title, limits) in zip(axes, rois.items()):
        xr = limits["x_range"]
        yr = limits["y_range"]
        
        # Filter points
        ref_sub = ref_df[(ref_df['x'] >= xr[0]) & (ref_df['x'] < xr[1]) & \
                         (ref_df['y'] >= yr[0]) & (ref_df['y'] < yr[1])]
                         
        tgt_sub = tgt_df[(tgt_df['x'] >= xr[0]) & (tgt_df['x'] < xr[1]) & \
                         (tgt_df['y'] >= yr[0]) & (tgt_df['y'] < yr[1])]
                         
        # Plot reference as blue circles
        ax.scatter(ref_sub['x'], ref_sub['y'], color='blue', marker='o', s=30, alpha=0.6, label='Reference (1-0)')
        
        # Plot target as red crosses
        ax.scatter(tgt_sub['x'], tgt_sub['y'], color='red', marker='x', s=40, alpha=0.8, label='Aligned Target (1-1)')
        
        ax.set_xlim(xr)
        ax.set_ylim(yr)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "alignment_debug_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved alignment debug plot to: {plot_path}")

if __name__ == "__main__":
    generate_debug_plot()
