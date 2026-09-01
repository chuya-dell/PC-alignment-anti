import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import pandas as pd
import numpy as np
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm

def load_image_unicode(path):
    """Unicode/Japanese path safe image read using numpy and cv2."""
    try:
        n = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None

import argparse
from path_resolver import resolve_gdrive_path

def main():
    parser = argparse.ArgumentParser(description="Grid Difference Analysis Pipeline")
    parser.add_argument("--img-dir", required=True, help="Directory containing raw microscope TIFF images")
    parser.add_argument("--pillar-dir", required=True, help="Directory containing aligned pillar CSV files")
    parser.add_argument("--output-dir", default=None, help="Output directory for grid analysis results (default: <pillar-dir>/grid_analysis)")
    parser.add_argument("--grid-size", type=float, default=6.29, help="Grid size in pixels (default: 6.29)")
    args = parser.parse_args()

    img_dir = resolve_gdrive_path(args.img_dir)
    pillar_dir = resolve_gdrive_path(args.pillar_dir)
    
    if args.output_dir is None:
        output_dir = os.path.join(pillar_dir, "grid_analysis")
    else:
        output_dir = resolve_gdrive_path(args.output_dir, create_if_missing=True)
        
    os.makedirs(output_dir, exist_ok=True)
    grid_size = args.grid_size

    print("Starting Grid Image Difference Analysis...")
    print(f"Image Directory:  {img_dir}")
    print(f"Pillar Directory: {pillar_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Grid Size:        {grid_size} px\n")

    files = os.listdir(pillar_dir)
    aligned_files = [f for f in files if f.endswith('_aligned_to_pre.csv')]
    
    # 自然順ソート (1-1, 1-2, ..., 8-8)
    def extract_key(filename):
def process_single_grid_task(task_args):
    f_aligned = task_args['f_aligned']
    img_dir = task_args['img_dir']
    pillar_dir = task_args['pillar_dir']
    output_dir = task_args['output_dir']
    grid_size = task_args['grid_size']
    all_img_files = task_args['all_img_files']

    base = f_aligned[:-19]

    pre_img_name = [img for img in all_img_files if img.startswith(f"{base}-0") and img.lower().endswith('.tif')]
    post_img_name = [img for img in all_img_files if img.startswith(f"{base}-1") and img.lower().endswith('.tif')]
    
    if not pre_img_name or not post_img_name:
        pre_img_name = [img for img in all_img_files if f"{base}-0" in img and img.lower().endswith('.tif')]
        post_img_name = [img for img in all_img_files if f"{base}-1" in img and img.lower().endswith('.tif')]
        
    if not pre_img_name or not post_img_name:
        return False
        
    pre_img_path = os.path.join(img_dir, pre_img_name[0])
    post_img_path = os.path.join(img_dir, post_img_name[0])
    
    aligned_path = os.path.join(pillar_dir, f_aligned)
    post_path = os.path.join(pillar_dir, f"{base}_pillars_post.csv")
    pre_path = os.path.join(pillar_dir, f"{base}_pillars_pre.csv")
    
    if not os.path.exists(post_path) or not os.path.exists(pre_path):
        return False
        
    df_aligned = pd.read_csv(aligned_path)
    df_post_orig = pd.read_csv(post_path)
    df_pre = pd.read_csv(pre_path)
    
    matched = df_aligned[df_aligned['matched_ref_id'] != -1].copy()
    if len(matched) < 10:
        return False
        
    matched_coords = pd.merge(
        matched[['pillar_id', 'matched_ref_id']].rename(columns={'pillar_id': 'aligned_id'}),
        df_post_orig[['pillar_id', 'x', 'y']].rename(columns={'x': 'x_post', 'y': 'y_post'}),
        left_on='aligned_id',
        right_on='pillar_id'
    )
    
    pair_coords = pd.merge(
        matched_coords,
        df_pre[['pillar_id', 'x', 'y']].rename(columns={'x': 'x_pre', 'y': 'y_pre'}),
        left_on='matched_ref_id',
        right_on='pillar_id'
    )
    
    src_pts = pair_coords[['x_post', 'y_post']].values.astype(np.float32)
    dst_pts = pair_coords[['x_pre', 'y_pre']].values.astype(np.float32)
    
    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        return False
        
    pre_img = load_image_unicode(pre_img_path)
    post_img = load_image_unicode(post_img_path)
    
    if pre_img is None or post_img is None:
        return False
        
    h, w = pre_img.shape
    post_aligned = cv2.warpPerspective(post_img, H, (w, h), flags=cv2.INTER_LINEAR)
    post_mask = cv2.warpPerspective(np.ones_like(post_img, dtype=np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    common_mask = (post_mask > 0) & (pre_img > 0)
    
    kernel_scratch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pre_open = cv2.morphologyEx(pre_img, cv2.MORPH_OPEN, kernel_scratch)
    thresh_pre = np.mean(pre_open) + 2.0 * np.std(pre_open)
    _, pre_scratch_mask = cv2.threshold(pre_open, thresh_pre, 255, cv2.THRESH_BINARY)
    
    post_open = cv2.morphologyEx(post_aligned, cv2.MORPH_OPEN, kernel_scratch)
    thresh_post = np.mean(post_open) + 2.0 * np.std(post_open)
    _, post_scratch_mask = cv2.threshold(post_open, thresh_post, 255, cv2.THRESH_BINARY)
    
    scratch_mask = (pre_scratch_mask > 0) | (post_scratch_mask > 0)
    valid_mask = common_mask & (~scratch_mask)
    
    diff_img = post_aligned.astype(np.float32) - pre_img.astype(np.float32)
    
    cols = int(np.floor(w / grid_size))
    rows = int(np.floor(h / grid_size))
    
    grid_records = []
    heatmap_data = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            x_start = int(np.floor(c * grid_size))
            x_end = int(np.floor((c + 1) * grid_size))
            y_start = int(np.floor(r * grid_size))
            y_end = int(np.floor((r + 1) * grid_size))
            
            block_diff = diff_img[y_start:y_end, x_start:x_end]
            block_mask = valid_mask[y_start:y_end, x_start:x_end]
            
            valid_pixels = block_diff[block_mask]
            
            if len(valid_pixels) > 0:
                mean_diff = float(np.mean(valid_pixels))
                std_diff = float(np.std(valid_pixels))
                pixel_count = int(len(valid_pixels))
            else:
                mean_diff = np.nan
                std_diff = np.nan
                pixel_count = 0
                
            grid_records.append({
                'Col': c + 1,
                'Row': r + 1,
                'Mean': mean_diff,
                'Std': std_diff,
                'PixelCount': pixel_count
            })
            
            heatmap_data[r, c] = mean_diff if not np.isnan(mean_diff) else 0.0
            
    df_grid = pd.DataFrame(grid_records)
    csv_out_path = os.path.join(output_dir, f"{base}_grid_analysis.csv")
    df_grid.to_csv(csv_out_path, index=False)
    
    v_min, v_max = -10.0, 10.0
    clipped = np.clip(heatmap_data, v_min, v_max)
    scaled = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    
    grid_mask = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if grid_records[idx]['PixelCount'] > 0:
                grid_mask[r, c] = 255
                
    heatmap_color[grid_mask == 0] = [0, 0, 0]
    
    display_scale = int(np.floor(2048 / cols))
    if display_scale > 1:
        h_out = rows * display_scale
        w_out = cols * display_scale
        heatmap_resized = cv2.resize(heatmap_color, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
    else:
        heatmap_resized = heatmap_color
        
    img_out_path = os.path.join(output_dir, f"{base}_grid_heatmap.png")
    _, ext = os.path.splitext(img_out_path)
    is_success, buffer = cv2.imencode(ext, heatmap_resized)
    if is_success:
        buffer.tofile(img_out_path)
    return True

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    parser = argparse.ArgumentParser(description="Grid Difference Analysis Pipeline")
    parser.add_argument("--img-dir", required=True, help="Directory containing raw microscope TIFF images")
    parser.add_argument("--pillar-dir", required=True, help="Directory containing aligned pillar CSV files")
    parser.add_argument("--output-dir", default=None, help="Output directory for grid analysis results (default: <pillar-dir>/grid_analysis)")
    parser.add_argument("--grid-size", type=float, default=6.29, help="Grid size in pixels (default: 6.29)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel CPU worker processes")
    args = parser.parse_args()

    img_dir = resolve_gdrive_path(args.img_dir)
    pillar_dir = resolve_gdrive_path(args.pillar_dir)
    
    if args.output_dir is None:
        output_dir = os.path.join(pillar_dir, "grid_analysis")
    else:
        output_dir = resolve_gdrive_path(args.output_dir, create_if_missing=True)
        
    os.makedirs(output_dir, exist_ok=True)
    grid_size = args.grid_size
    workers = args.workers if args.workers else max(1, os.cpu_count() - 2)

    print("Starting Grid Image Difference Analysis (Parallel Execution)...")
    print(f"Image Directory:  {img_dir}")
    print(f"Pillar Directory: {pillar_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Grid Size:        {grid_size} px")
    print(f"CPU Workers:      {workers}\n")

    files = os.listdir(pillar_dir)
    aligned_files = [f for f in files if f.endswith('_aligned_to_pre.csv')]

    def extract_key(filename):
        base = filename[:-19]
        parts = base.split('-')
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return (999, 999)
            
    aligned_files.sort(key=extract_key)
    print(f"Found {len(aligned_files)} aligned pairs to process in parallel.")
    all_img_files = os.listdir(img_dir)

    task_list = [{
        'f_aligned': f,
        'img_dir': img_dir,
        'pillar_dir': pillar_dir,
        'output_dir': output_dir,
        'grid_size': grid_size,
        'all_img_files': all_img_files
    } for f in aligned_files]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_grid_task, task) for task in task_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel Grid Analysis ({workers} CPU Cores)"):
            pass

    print(f"\nGrid analysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
