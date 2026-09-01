import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import re
import argparse
import pandas as pd
import numpy as np
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm
import time

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import analyze_image
from registration import align_and_match_dataframes

from path_resolver import resolve_gdrive_path

def group_files(input_dir):
    """
    input_dir内の全ファイルを条件・セット・連番でグルーピング。
    戻り値: dict of (condition, set_num) -> {'pre': path, 'post': path}
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return {}

    files = os.listdir(input_dir)
    print(f"Scanning {len(files)} files in {input_dir}...")

    groups = {}

    for f in files:
        if not f.lower().endswith('.tif'):
            continue

        # 拡張子を除去したベース名を取得
        base, ext = os.path.splitext(f)
        
        # 重複/コピーを表すカッコ "(1)" などを除去
        base_clean = re.sub(r'\s*\(\d+\)\s*', '', base)
        # 末尾のドットやハイフンをトリム
        base_clean = base_clean.rstrip('.-')

        # ハイフンで分割
        parts = base_clean.split('-')
        if len(parts) < 3:
            print(f"Warning: Filename '{f}' does not match 'Condition-Set-Sequence' pattern. Skipped.")
            continue

        cond = parts[0]
        set_num = parts[1]
        seq_num = parts[2]

        if seq_num not in ['0', '1']:
            print(f"Warning: Unknown sequence number '{seq_num}' in file '{f}'. Skipped.")
            continue

        key = (cond, set_num)
        if key not in groups:
            groups[key] = {'pre': None, 'post': None}

        role = 'pre' if seq_num == '0' else 'post'
        path = os.path.join(input_dir, f)

        if groups[key][role] is not None:
            # 重複発見の警告
            print(f"Warning: Duplicate file found for {role} of condition {cond}, set {set_num}:")
            print(f"  Existing: {groups[key][role]}")
            print(f"  New:      {path} (Skipped)")
        else:
            groups[key][role] = path

    return groups

def auto_detect_invert(img_path):
    """
    画像の平均輝度を調べ、背景が明るい明視野画像 (invert=True) か、
    背景が暗い暗視野画像 (invert=False) かを自動判定する。
    """
    try:
        # 日本語パス安全な読み込み
        n = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        mean_val = np.mean(img)
        # 127より大きければ明視野とみなして反転を有効にする
        return mean_val > 127
    except Exception as e:
        print(f"Error auto-detecting contrast invert for '{img_path}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pillar Alignment Batch Execution Pipeline")
    parser.add_argument("--input-dir", required=True, help="Input directory containing image dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--method", default="peak", choices=["blob", "peak"], help="Pillar detection method: 'peak' (default, local maxima for HCP) or 'blob' (connected components)")
    parser.add_argument("--min-dist", type=int, default=5, help="Minimum distance between peaks (for 'peak' method)")
    parser.add_argument("--threshold", type=float, default=None, help="Absolute detection threshold (for 'peak' method)")
    parser.add_argument("--invert", action="store_true", help="Manually force invert contrast during pillar detection")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable automatic contrast invert detection")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pairs to process (for testing)")
    
    args = parser.parse_args()

    # パスの自動解決
def process_pair_task(task_args):
    pair = task_args['pair']
    output_dir = task_args['output_dir']
    method = task_args['method']
    min_dist = task_args['min_dist']
    threshold = task_args['threshold']
    invert = task_args['invert']
    no_auto_invert = task_args['no_auto_invert']

    pair_name = pair['name']
    pre_img_path = pair['pre_path']
    post_img_path = pair['post_path']

    inv = invert
    if not no_auto_invert:
        inv = auto_detect_invert(pre_img_path)

    record = {
        'pair_name': pair_name,
        'pre_pillars': 0,
        'post_pillars': 0,
        'matched_pillars': 0,
        'match_rate': 0.0,
        'coarse_dx': 0.0,
        'coarse_dy': 0.0,
        'icp_iterations': 0,
        'icp_converged': False,
        'icp_rmse': 0.0,
        'status': 'FAILED',
        'error_message': ''
    }

    try:
        df_pre = analyze_image(
            image_path=pre_img_path,
            method=method,
            min_area=3,
            max_area=100,
            top_hat_size=15,
            min_dist=min_dist,
            threshold=threshold,
            invert=inv
        )
        df_post = analyze_image(
            image_path=post_img_path,
            method=method,
            min_area=3,
            max_area=100,
            top_hat_size=15,
            min_dist=min_dist,
            threshold=threshold,
            invert=inv
        )

        record['pre_pillars'] = len(df_pre)
        record['post_pillars'] = len(df_post)

        pre_csv_out = os.path.join(output_dir, f"{pair_name}_pillars_pre.csv")
        post_csv_out = os.path.join(output_dir, f"{pair_name}_pillars_post.csv")
        
        df_pre.to_csv(pre_csv_out, index=False)
        df_post.to_csv(post_csv_out, index=False)

        if len(df_pre) < 10 or len(df_post) < 10:
            raise ValueError(f"Too few pillars detected (pre: {len(df_pre)}, post: {len(df_post)}). Cannot align.")

        df_aligned, H_final, iter_count, converged, dx_coarse, dy_coarse = align_and_match_dataframes(
            df_ref=df_pre,
            df_tgt=df_post,
            ref_img_path=pre_img_path,
            tgt_img_path=post_img_path,
            return_diagnostics=True
        )

        aligned_csv_out = os.path.join(output_dir, f"{pair_name}_aligned_to_pre.csv")
        df_aligned.to_csv(aligned_csv_out, index=False)

        dists = df_aligned['alignment_distance'].values
        matched_mask = df_aligned['matched_ref_id'] != -1
        matched_dists = dists[matched_mask]
        
        num_matched = len(matched_dists)
        match_rate = (num_matched / len(df_post)) * 100.0 if len(df_post) > 0 else 0.0
        rmse = np.sqrt(np.mean(matched_dists**2)) if num_matched > 0 else 0.0

        record['matched_pillars'] = num_matched
        record['match_rate'] = match_rate
        record['coarse_dx'] = dx_coarse
        record['coarse_dy'] = dy_coarse
        record['icp_iterations'] = iter_count
        record['icp_converged'] = converged
        record['icp_rmse'] = rmse
        
        if match_rate < 70.0:
            record['status'] = 'WARNING'
            record['error_message'] = f'Low match rate: {match_rate:.2f}%'
        else:
            record['status'] = 'SUCCESS'

    except Exception as e:
        record['status'] = 'FAILED'
        record['error_message'] = str(e)

    return record

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    parser = argparse.ArgumentParser(description="Pillar Alignment Batch Execution Pipeline")
    parser.add_argument("--input-dir", required=True, help="Input directory containing image dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--method", default="peak", choices=["blob", "peak"], help="Pillar detection method: 'peak' or 'blob'")
    parser.add_argument("--min-dist", type=int, default=5, help="Minimum distance between peaks")
    parser.add_argument("--threshold", type=float, default=None, help="Absolute detection threshold")
    parser.add_argument("--invert", action="store_true", help="Manually force invert contrast during pillar detection")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable automatic contrast invert detection")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pairs to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel CPU worker processes")
    
    args = parser.parse_args()

    input_dir = resolve_gdrive_path(args.input_dir)
    output_dir = resolve_gdrive_path(args.output_dir)

    print(f"Resolved Input Directory:  {input_dir}")
    print(f"Resolved Output Directory: {output_dir}")

    groups = group_files(input_dir)
    if not groups:
        print("No valid file pairs found.")
        return

    valid_pairs = []
    for (cond, set_num), paths in groups.items():
        pair_name = f"{cond}-{set_num}"
        if paths['pre'] is None or paths['post'] is None:
            continue
            
        valid_pairs.append({
            'name': pair_name,
            'cond': cond,
            'set': set_num,
            'pre_path': paths['pre'],
            'post_path': paths['post']
        })

    try:
        valid_pairs.sort(key=lambda x: (int(x['cond']), int(x['set'])))
    except ValueError:
        valid_pairs.sort(key=lambda x: x['name'])

    if args.limit is not None:
        print(f"Limiting execution to the first {args.limit} pairs.")
        valid_pairs = valid_pairs[:args.limit]

    workers = args.workers if args.workers else max(1, os.cpu_count() - 2)
    print(f"Total pairs to process: {len(valid_pairs)}")
    print(f"Executing batch alignment in parallel using {workers} CPU workers...")
    os.makedirs(output_dir, exist_ok=True)

    task_list = [{
        'pair': pair,
        'output_dir': output_dir,
        'method': args.method,
        'min_dist': args.min_dist,
        'threshold': args.threshold,
        'invert': args.invert,
        'no_auto_invert': args.no_auto_invert
    } for pair in valid_pairs]

    summary_records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_pair_task, task) for task in task_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Parallel Alignment ({workers} CPU Cores)"):
            try:
                rec = future.result()
                summary_records.append(rec)
            except Exception as e:
                print(f"Error processing pair task: {e}")

    try:
        summary_records.sort(key=lambda x: (int(x['pair_name'].split('-')[0]), int(x['pair_name'].split('-')[1])))
    except Exception:
        summary_records.sort(key=lambda x: x['pair_name'])

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "="*60)
    print(f" Batch processing completed!")
    print(f" Quality summary saved to: {summary_csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
