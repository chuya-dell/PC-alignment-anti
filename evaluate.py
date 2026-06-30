import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import cv2
import matplotlib.pyplot as plt
import time
import os

def evaluate(gt_path, det_path, match_threshold=3.0):
    print("Loading datasets...")
    df_gt = pd.read_csv(gt_path)
    df_det = pd.read_csv(det_path)
    
    print(f"Ground Truth Count: {len(df_gt):,}")
    print(f"Detected Count: {len(df_det):,}")
    
    if len(df_det) == 0:
        print("No pillars detected!")
        return
        
    # Build KD-Tree for fast spatial matching
    gt_coords = df_gt[['x', 'y']].values
    det_coords = df_det[['x', 'y']].values
    
    print("Building KD-Tree and querying matches...")
    t0 = time.time()
    tree = KDTree(det_coords)
    # For each ground truth pillar, find the closest detected pillar
    distances, indices = tree.query(gt_coords, distance_upper_bound=match_threshold)
    print(f"Spatial query completed in {time.time() - t0:.2f} seconds.")
    
    # Filter valid matches (distance within threshold)
    valid_mask = distances < match_threshold
    num_matches = np.sum(valid_mask)
    
    matched_gt_indices = np.where(valid_mask)[0]
    matched_det_indices = indices[valid_mask]
    
    # Calculate Precision and Recall
    precision = num_matches / len(df_det)
    recall = num_matches / len(df_gt)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Evaluation Results ---")
    print(f"True Positives (Matches): {num_matches:,}")
    print(f"Precision (適合率): {precision:.4%}")
    print(f"Recall (再現率): {recall:.4%}")
    print(f"F1-Score: {f1:.4%}")
    
    # Calculate coordinate error
    if num_matches > 0:
        errors = distances[valid_mask]
        print(f"Mean Coordinate Error (pixels): {np.mean(errors):.4f}")
        print(f"Max Coordinate Error (pixels): {np.max(errors):.4f}")
        
        # Calculate intensity error (compare true_intensity vs detected intensity metrics)
        gt_matched_intensities = df_gt.iloc[matched_gt_indices]['true_intensity'].values
        det_matched_means = df_det.iloc[matched_det_indices]['mean_intensity'].values
        det_matched_box = df_det.iloc[matched_det_indices]['box_intensity_3x3'].values
        
        mae_mean = np.mean(np.abs(gt_matched_intensities - det_matched_means))
        rmse_mean = np.sqrt(np.mean((gt_matched_intensities - det_matched_means)**2))
        
        mae_box = np.mean(np.abs(gt_matched_intensities - det_matched_box))
        rmse_box = np.sqrt(np.mean((gt_matched_intensities - det_matched_box)**2))
        
        print("\nIntensity Error (Mean Intensity vs Ground Truth):")
        print(f"  Mean Absolute Error (MAE): {mae_mean:.4f}")
        print(f"  Root Mean Square Error (RMSE): {rmse_mean:.4f}")
        
        print("\nIntensity Error (3x3 Box Intensity vs Ground Truth):")
        print(f"  Mean Absolute Error (MAE): {mae_box:.4f}")
        print(f"  Root Mean Square Error (RMSE): {rmse_box:.4f}")
        
        # Check correlation
        r_mean = np.corrcoef(gt_matched_intensities, det_matched_means)[0, 1]
        r_box = np.corrcoef(gt_matched_intensities, det_matched_box)[0, 1]
        print(f"\nCorrelation Coefficient (R) for Mean Intensity: {r_mean:.5f}")
        print(f"Correlation Coefficient (R) for 3x3 Box Intensity: {r_box:.5f}")

def create_visual_verification(image_path, gt_path, det_path, crop_x=2000, crop_y=2000, crop_size=300):
    print("\nGenerating visual verification crop...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image for visualization!")
        return
        
    df_gt = pd.read_csv(gt_path)
    df_det = pd.read_csv(det_path)
    
    # Filter detections and ground truth within crop box
    gt_crop = df_gt[
        (df_gt['x'] >= crop_x) & (df_gt['x'] < crop_x + crop_size) &
        (df_gt['y'] >= crop_y) & (df_gt['y'] < crop_y + crop_size)
    ]
    det_crop = df_det[
        (df_det['x'] >= crop_x) & (df_det['x'] < crop_x + crop_size) &
        (df_det['y'] >= crop_y) & (df_det['y'] < crop_y + crop_size)
    ]
    
    # Crop original image and convert to color for drawing
    img_crop = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    img_color = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    
    # Draw true positions as green dots
    for _, row in gt_crop.iterrows():
        x = int(row['x'] - crop_x)
        y = int(row['y'] - crop_y)
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)  # Green dot
        
    # Draw detected positions as red open circles
    for _, row in det_crop.iterrows():
        x = int(row['x'] - crop_x)
        y = int(row['y'] - crop_y)
        cv2.circle(img_color, (x, y), 4, (0, 0, 255), 1)  # Red circle
        
    # Save the output image
    out_dir = "C:/Users/chuya/.gemini/antigravity/brain/791c8f54-1705-4b80-864f-cb0af0737e3d"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "detection_overlay.png")
    cv2.imwrite(out_path, img_color)
    print(f"Saved visual verification crop to {out_path}")

if __name__ == "__main__":
    gt_csv = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/ground_truth.csv"
    det_csv = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/results.csv"
    image_png = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/synthetic_pillars.png"
    
    evaluate(gt_csv, det_csv)
    create_visual_verification(image_png, gt_csv, det_csv)
