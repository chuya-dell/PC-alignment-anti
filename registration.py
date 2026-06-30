import numpy as np
import cv2
import pandas as pd
from scipy.spatial import KDTree
import os

def load_image_unicode(path):
    """Unicode/Japanese path safe image read using numpy and cv2."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None

def detect_grooves(img):
    """
    Detect vertical and horizontal groove landmarks from the microscope image.
    Returns (groove_x, groove_y) coords.
    """
    h, w = img.shape
    
    # 1. Vertical Groove (X coordinate)
    # Average columns vertically in the central y-band [500, 1500] to avoid edge artifacts
    x_profile = np.mean(img[500:1500, 0:300], axis=0)
    # Smooth profile to reduce pixel-level noise (moving average of size 15)
    x_profile_smooth = np.convolve(x_profile, np.ones(15)/15, mode='same')
    # Find global minimum in safe range [30, 270]
    groove_x = 30 + np.argmin(x_profile_smooth[30:270])
    
    # 2. Horizontal Groove (Y coordinate)
    # Average rows horizontally in the right-side x-band [1000, 1900]
    y_profile = np.mean(img[0:300, 1000:1900], axis=1)
    y_profile_smooth = np.convolve(y_profile, np.ones(15)/15, mode='same')
    # Find global minimum in safe range [30, 270]
    groove_y = 30 + np.argmin(y_profile_smooth[30:270])
    
    return float(groove_x), float(groove_y)

def calculate_coarse_shift(ref_img_path, tgt_img_path):
    """
    Detect grooves in both reference and target, and compute the coarse translation offset.
    Returns (dx, dy) to shift the target to align with reference.
    """
    ref_img = load_image_unicode(ref_img_path)
    tgt_img = load_image_unicode(tgt_img_path)
    
    if ref_img is None or tgt_img is None:
        raise ValueError("Failed to load reference or target image.")
        
    ref_x, ref_y = detect_grooves(ref_img)
    tgt_x, tgt_y = detect_grooves(tgt_img)
    
    dx = ref_x - tgt_x
    dy = ref_y - tgt_y
    
    print(f"Landmark detection:")
    print(f"  Reference grooves: X={ref_x:.1f}, Y={ref_y:.1f}")
    print(f"  Target grooves   : X={tgt_x:.1f}, Y={tgt_y:.1f}")
    print(f"  Coarse Shift (dx, dy): ({dx:.1f}, {dy:.1f})")
    
    return dx, dy

def fine_alignment_icp(ref_pts, tgt_pts_coarse, max_iter=100, tolerance=1e-6, max_match_dist=3.0):
    """
    Iterative Closest Point (ICP) registration between reference and coarse-aligned target points.
    Estimates a rigid transform (rotation & translation).
    Returns (H_fine, aligned_pts) where H_fine is a 2x3 affine matrix.
    """
    aligned_pts = tgt_pts_coarse.copy()
    ref_tree = KDTree(ref_pts)
    
    # Cumulative affine transformation matrix (3x3)
    T_total = np.eye(3)
    
    prev_err = float('inf')
    
    for i in range(max_iter):
        # Find closest reference point for each target point
        distances, indices = ref_tree.query(aligned_pts, k=1)
        
        # Outlier rejection: only keep pairs within a threshold distance
        valid = distances < max_match_dist
        num_valid = np.sum(valid)
        if num_valid < 10:
            print(f"ICP warning: Too few matching points ({num_valid}). Aborting.")
            break
            
        src = aligned_pts[valid]
        dst = ref_pts[indices[valid]]
        
        # Compute centroids
        mu_src = np.mean(src, axis=0)
        mu_dst = np.mean(dst, axis=0)
        
        # Center points
        src_centered = src - mu_src
        dst_centered = dst - mu_dst
        
        # Covariance matrix
        H_cov = src_centered.T @ dst_centered
        
        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H_cov)
        R = Vt.T @ U.T
        
        # Special reflection handling
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
            
        t = mu_dst - R @ mu_src
        
        # Construct this step's transform matrix
        T_step = np.eye(3)
        T_step[0:2, 0:2] = R
        T_step[0:2, 2] = t
        
        # Apply step transform to all target points
        aligned_pts = (R @ aligned_pts.T).T + t
        
        # Accumulate transform
        T_total = T_step @ T_total
        
        # Check convergence
        mean_err = np.mean(distances[valid])
        if abs(prev_err - mean_err) < tolerance:
            break
        prev_err = mean_err
        
    return T_total[0:2, :], aligned_pts

def align_and_match_dataframes(df_ref, df_tgt, ref_img_path, tgt_img_path):
    """
    Performs full 2-stage alignment on target dataframe coordinates to match reference.
    Returns:
      df_tgt_aligned: df_tgt with updated 'x', 'y' coordinates, and added 'matched_ref_id' and 'alignment_distance'.
      H_final: Final 2x3 affine transformation matrix.
    """
    # 1. Coarse shift using physical landmarks (grooves)
    dx_coarse, dy_coarse = calculate_coarse_shift(ref_img_path, tgt_img_path)
    
    # Apply coarse shift
    ref_coords = df_ref[['x', 'y']].values
    tgt_coords = df_tgt[['x', 'y']].values
    
    tgt_coords_coarse = tgt_coords.copy()
    tgt_coords_coarse[:, 0] += dx_coarse
    tgt_coords_coarse[:, 1] += dy_coarse
    
    # 2. Fine shift using ICP point-matching
    print("Running Fine Registration (ICP)...")
    H_fine, tgt_coords_fine = fine_alignment_icp(ref_coords, tgt_coords_coarse)
    
    # Combine Coarse translation and Fine rigid transform to get H_final
    # H_fine maps (x_coarse, y_coarse) -> (x_fine, y_fine)
    # x_coarse = x + dx_coarse,  y_coarse = y + dy_coarse
    # We can write H_final that directly maps (x, y) -> (x_fine, y_fine)
    # H_fine = [R | t]
    # H_final = [R | R * [dx_coarse, dy_coarse]^T + t]
    R_fine = H_fine[0:2, 0:2]
    t_fine = H_fine[0:2, 2]
    t_coarse = np.array([dx_coarse, dy_coarse])
    
    t_final = R_fine @ t_coarse + t_fine
    
    H_final = np.zeros((2, 3))
    H_final[0:2, 0:2] = R_fine
    H_final[0:2, 2] = t_final
    
    print(f"Final Affine Transform matrix (H_final):")
    print(f"  R: {R_fine[0,0]:.6f} {R_fine[0,1]:.6f} | {R_fine[1,0]:.6f} {R_fine[1,1]:.6f}")
    print(f"  t (dx, dy): ({t_final[0]:.4f}, {t_final[1]:.4f})")
    
    # Create final aligned DataFrame
    df_tgt_aligned = df_tgt.copy()
    df_tgt_aligned['x'] = tgt_coords_fine[:, 0]
    df_tgt_aligned['y'] = tgt_coords_fine[:, 1]
    
    # Find matching reference IDs for each aligned pillar
    ref_tree = KDTree(ref_coords)
    distances, indices = ref_tree.query(tgt_coords_fine, k=1)
    
    df_tgt_aligned['matched_ref_id'] = df_ref['pillar_id'].values[indices]
    df_tgt_aligned['alignment_distance'] = distances
    
    # Set matched_ref_id to -1 if distance is too large (e.g. > 1.5 pixels) indicating no true match
    no_match = distances > 1.5
    df_tgt_aligned.loc[no_match, 'matched_ref_id'] = -1
    
    matched_count = np.sum(~no_match)
    print(f"Alignment completed: {matched_count}/{len(df_tgt)} pillars matched within 1.5 pixels ({matched_count/len(df_tgt)*100:.2f}%)")
    
    return df_tgt_aligned, H_final
