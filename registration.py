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
    Returns globally stable (groove_x, groove_y) coordinates.
    """
    h, w = img.shape
    
    # Sub-pixel interpolation helper
    def get_subpixel_min(profile, min_idx):
        if 0 < min_idx < len(profile) - 1:
            y0, y1, y2 = profile[min_idx-1], profile[min_idx], profile[min_idx+1]
            denom = (y0 - 2*y1 + y2)
            if denom != 0:
                return min_idx + 0.5 * (y0 - y2) / denom
        return float(min_idx)
        
    # 1. Vertical Groove (X coordinate)
    # Average columns vertically in the central y-band [500, 1500] to avoid edge artifacts
    x_profile = np.mean(img[500:1500, 0:300], axis=0)
    # Remove shading gradient by subtracting a 51-pixel moving average
    x_detrend = x_profile - np.convolve(x_profile, np.ones(51)/51, mode='same')
    x_profile_smooth = np.convolve(x_detrend, np.ones(15)/15, mode='same')
    groove_x_idx = 30 + np.argmin(x_profile_smooth[30:270])
    groove_x = get_subpixel_min(x_profile_smooth, groove_x_idx)
    
    # 2. Horizontal Groove (Y coordinate)
    # Average rows horizontally in the right-side x-band [1000, 1900]
    y_profile = np.mean(img[0:300, 1000:1900], axis=1)
    # Remove shading gradient
    y_detrend = y_profile - np.convolve(y_profile, np.ones(51)/51, mode='same')
    y_profile_smooth = np.convolve(y_detrend, np.ones(15)/15, mode='same')
    groove_y_idx = 30 + np.argmin(y_profile_smooth[30:270])
    groove_y = get_subpixel_min(y_profile_smooth, groove_y_idx)
    
    return groove_x, groove_y

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
    print(f"  Reference grooves: X={ref_x:.2f}, Y={ref_y:.2f}")
    print(f"  Target grooves   : X={tgt_x:.2f}, Y={tgt_y:.2f}")
    print(f"  Coarse Shift (dx, dy): ({dx:.2f}, {dy:.2f})")
    
    return dx, dy

def find_grid_orientation(pts):
    """
    Finds dominant grid orientation angle (in radians) in [-30, 30] degrees
    by computing the mode of nearest-neighbor vector angles. Extremely robust.
    """
    tree = KDTree(pts)
    # Query 7 closest neighbors for each point
    distances, indices = tree.query(pts, k=7)
    
    angles_list = []
    
    for i in range(len(pts)):
        p = pts[i]
        for j in range(1, 7): # Skip index 0 (itself)
            d = distances[i, j]
            # HCP grid spacing is ~5.5px, so query neighboring vectors in [4.5, 6.5]px
            if 4.5 <= d <= 6.5:
                q = pts[indices[i, j]]
                # Compute vector angle
                angle_rad = np.arctan2(q[1] - p[1], q[0] - p[0])
                angle_deg = np.degrees(angle_rad)
                # Fold into [-30, 30] degree range (hexagonal 60-degree symmetry)
                angle_fold = (angle_deg + 30.0) % 60.0 - 30.0
                angles_list.append(angle_fold)
                
    if len(angles_list) == 0:
        return 0.0
        
    # Build histogram of angles with very high resolution (0.02 degrees bins)
    bins = np.arange(-30.0, 30.0, 0.02)
    hist, bin_edges = np.histogram(angles_list, bins=bins)
    
    # Smooth histogram to find the true continuous peak (moving average of size 25)
    hist_smooth = np.convolve(hist, np.ones(25)/25, mode='same')
    
    best_idx = np.argmax(hist_smooth)
    best_angle_deg = 0.5 * (bin_edges[best_idx] + bin_edges[best_idx + 1])
    
    return np.radians(best_angle_deg)

def fine_alignment_icp(ref_pts, tgt_pts_coarse, max_iter=100, tolerance=1e-6, return_diagnostics=False):
    """
    Iterative Closest Point (ICP) registration between reference and coarse-aligned target points.
    Estimates a 2nd-order quadratic polynomial transform to correct non-linear lens distortion.
    Uses a tight distance threshold (1.8px) from the start to prevent grid aliasing (matching with neighbors).
    """
    aligned_pts = tgt_pts_coarse.copy()
    ref_tree = KDTree(ref_pts)
    
    # Keep copy of the starting coarse target coordinates
    orig_pts = tgt_pts_coarse.copy()
    
    prev_err = float('inf')
    
    # Best-fit coefficients
    C_best = np.zeros((6, 2))
    
    converged = False
    iter_count = 0
    
    for i in range(max_iter):
        iter_count = i + 1
        # Two-stage dynamic thresholding:
        # First 3 iterations use 3.5px to capture the remaining coarse translation.
        # Remaining iterations use 1.5px to lock onto the sub-pixel grid and avoid grid aliasing.
        dist_thresh = 3.5 if i < 3 else 1.5
        
        # Find closest reference point for each target point
        distances, indices = ref_tree.query(aligned_pts, k=1)
        
        # Outlier rejection using the dynamic threshold
        valid = distances < dist_thresh
        num_valid = np.sum(valid)
        if num_valid < 20:
            break
            
        src = orig_pts[valid]
        dst = ref_pts[indices[valid]]
        
        # Estimate 2nd-order polynomial transform (6 coefficients for x and y)
        x = src[:, 0]
        y = src[:, 1]
        X_mat = np.column_stack([x, y, np.ones_like(x), x**2, y**2, x*y])
        Y_mat = dst
        
        # Solve the linear system directly using least squares
        C_best, _, _, _ = np.linalg.lstsq(X_mat, Y_mat, rcond=None)
        
        # Apply the estimated transform to all original points to get new aligned coordinates
        x_all = orig_pts[:, 0]
        y_all = orig_pts[:, 1]
        X_all = np.column_stack([x_all, y_all, np.ones_like(x_all), x_all**2, y_all**2, x_all*y_all])
        aligned_pts = X_all @ C_best
        
        # Check convergence
        mean_err = np.mean(distances[valid])
        if abs(prev_err - mean_err) < tolerance:
            converged = True
            break
        prev_err = mean_err
        
    # Extract linear affine part [A | t] for summary representation (first 3 coefficients)
    H_summary = C_best.T[:, 0:3]
    
    if return_diagnostics:
        return H_summary, aligned_pts, iter_count, converged
    else:
        return H_summary, aligned_pts

def fine_alignment_local_refinement(ref_pts, tgt_pts_coarse, img_w=2048, img_h=2044, grid_n=8):
    """
    Performs local translation refinement on an grid_n x grid_n layout.
    First runs global Affine ICP, then estimates local median offsets (dx, dy)
    for each grid cell to correct local lens distortion. Extremely robust.
    """
    print("  Calculating global Affine ICP transform...")
    H_global, aligned_pts = fine_alignment_icp(ref_pts, tgt_pts_coarse, max_iter=60)
    
    cell_w = img_w / grid_n
    cell_h = img_h / grid_n
    
    ref_tree = KDTree(ref_pts)
    
    # We will refine aligned_pts in-place
    refined_pts = aligned_pts.copy()
    
    print(f"  Refining local translations on a {grid_n}x{grid_n} grid...")
    for r in range(grid_n):
        for c in range(grid_n):
            # Define cell bounding box (aligned coordinates)
            x_min, x_max = c * cell_w, (c + 1) * cell_w
            y_min, y_max = r * cell_h, (r + 1) * cell_h
            
            # Find indices of points falling into this cell
            in_cell_mask = (aligned_pts[:, 0] >= x_min) & (aligned_pts[:, 0] < x_max) & \
                           (aligned_pts[:, 1] >= y_min) & (aligned_pts[:, 1] < y_max)
            
            cell_idx = np.where(in_cell_mask)[0]
            if len(cell_idx) == 0:
                continue
                
            cell_pts = aligned_pts[cell_idx]
            
            # Find nearest neighbors in the reference set
            distances, indices = ref_tree.query(cell_pts, k=1)
            
            # Keep only close matches to estimate the local shift (threshold 2.0px)
            valid = distances < 2.0
            if np.sum(valid) < 15:
                # Fallback: do not adjust translation if matches are too sparse
                continue
                
            # Compute median shift to avoid outlier effects
            src_valid = cell_pts[valid]
            dst_valid = ref_pts[indices[valid]]
            
            shifts = dst_valid - src_valid
            median_shift = np.median(shifts, axis=0) # [dx, dy]
            
            # Apply local shift correction
            refined_pts[cell_idx] += median_shift
            
    return H_global, refined_pts

def align_and_match_dataframes(df_ref, df_tgt, ref_img_path, tgt_img_path, return_diagnostics=False):
    """
    Performs full 2-stage alignment on target dataframe coordinates to match reference.
    Stage 1: Landmark translation and vertical groove rotation (Stage 1.5)
    Stage 2: Subpixel Quadratic Polynomial ICP
    """
    # 1. Coarse translation using landmarks, and rotation using grid orientation variance maximization
    dx_coarse, dy_coarse = calculate_coarse_shift(ref_img_path, tgt_img_path)
    
    ref_coords = df_ref[['x', 'y']].values
    tgt_coords = df_tgt[['x', 'y']].values
    
    print("Estimating grid dominant orientations...")
    ref_grid_angle = find_grid_orientation(ref_coords)
    tgt_grid_angle = find_grid_orientation(tgt_coords)
    
    # Target orientation subtract reference orientation
    theta_coarse = tgt_grid_angle - ref_grid_angle
    # Fold to [-30, 30] deg range due to 60-deg symmetry
    theta_coarse = (theta_coarse + np.pi/6) % (np.pi/3) - np.pi/6
    
    print(f"  Grid Orientations - Reference: {np.degrees(ref_grid_angle):.4f}°, Target: {np.degrees(tgt_grid_angle):.4f}°")
    print(f"  Coarse Rotation angle (theta): {np.degrees(theta_coarse):.4f}°")
    
    # Apply coarse rotation around image center (cx, cy) = (1024, 1022)
    cx, cy = 1024.0, 1022.0
    c = np.array([cx, cy])
    
    cos_t = np.cos(theta_coarse)
    sin_t = np.sin(theta_coarse)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    
    # Rotate tgt_coords
    tgt_coords_shifted = tgt_coords - c
    tgt_coords_rot = (R @ tgt_coords_shifted.T).T + c
    
    # Apply translation
    tgt_coords_coarse = tgt_coords_rot.copy()
    tgt_coords_coarse[:, 0] += dx_coarse
    tgt_coords_coarse[:, 1] += dy_coarse
    
    # 2. Fine shift using global 2nd-order polynomial ICP
    print("Running Global Polynomial Registration...")
    if return_diagnostics:
        H_fine, tgt_coords_fine, iter_count, converged = fine_alignment_icp(
            ref_coords, tgt_coords_coarse, return_diagnostics=True
        )
    else:
        H_fine, tgt_coords_fine = fine_alignment_icp(ref_coords, tgt_coords_coarse)
        iter_count, converged = -1, False
    
    # Combine Coarse translation/rotation and Fine polynomial linear part for H_final
    A_fine = H_fine[0:2, 0:2]
    t_fine = H_fine[0:2, 2]
    
    # Mathematical synthesis of final affine matrix [A_final | t_final]
    # p_aligned = A_fine * R * p + A_fine * (c - R * c + t_coarse) + t_fine
    A_final = A_fine @ R
    t_coarse_vec = np.array([dx_coarse, dy_coarse])
    t_final = A_fine @ (c - R @ c + t_coarse_vec) + t_fine
    
    H_final = np.zeros((2, 3))
    H_final[0:2, 0:2] = A_final
    H_final[0:2, 2] = t_final
    
    print(f"Global Affine Transform matrix (H_final):")
    print(f"  A: {A_final[0,0]:.6f} {A_final[0,1]:.6f} | {A_final[1,0]:.6f} {A_final[1,1]:.6f}")
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
    
    if return_diagnostics:
        return df_tgt_aligned, H_final, iter_count, converged, dx_coarse, dy_coarse
    else:
        return df_tgt_aligned, H_final
