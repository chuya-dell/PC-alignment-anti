import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import time

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import analyze_image
from registration import align_and_match_dataframes

def resolve_gdrive_path(path):
    """
    Windows環境でGドライブのマウントパスをFドライブに置換して解決する。
    """
    if path is None:
        return None
    
    # バックスラッシュをスラッシュに置換して処理しやすくする
    resolved = path.replace('\\', '/')
    
    # G:/マイドライブ or G:/My Drive を F:/GoogleDrive_local に置換 (大文字小文字不問)
    resolved = re.sub(r'^[Gg]:/マイドライブ', 'F:/GoogleDrive_local', resolved)
    resolved = re.sub(r'^[Gg]:/My Drive', 'F:/GoogleDrive_local', resolved)
    
    return resolved

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
    parser = argparse.ArgumentParser(description="High-Density Pillar Alignment Batch Pipeline")
    parser.add_argument("--input-dir", required=True, help="Input directory containing image dataset")
    parser.add_argument("--output-dir", default=None, help="Output directory for results (default: input-dir/../foranti_high_density)")
    # 高密度・極限検出のためのハイパーパラメータのデフォルト値をチューニング
    parser.add_argument("--method", default="peak", choices=["blob", "peak"], help="Pillar detection method (default: peak)")
    parser.add_argument("--min-dist", type=int, default=3, help="Minimum peak separation (default: 3 for maximum density)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection contrast threshold (default: 0.2 to capture weak pillars)")
    parser.add_argument("--invert", action="store_true", help="Manually force invert contrast during pillar detection")
    parser.add_argument("--no-auto-invert", action="store_true", help="Disable automatic contrast invert detection")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pairs to process (for testing)")
    
    args = parser.parse_args()

    # パスの自動解決
    input_dir = resolve_gdrive_path(args.input_dir)
    
    # 出力先デフォルトの設定
    if args.output_dir is None:
        # 入力ディレクトリの親に foranti_high_density を作る
        parent_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(parent_dir, "foranti_high_density")
    else:
        output_dir = resolve_gdrive_path(args.output_dir)

    print(f"Resolved Input Directory:  {input_dir}")
    print(f"Resolved Output Directory: {output_dir}")
    print(f"High-Density Param: method={args.method}, min-dist={args.min_dist}, threshold={args.threshold}")

    groups = group_files(input_dir)
    if not groups:
        print("No valid file pairs found.")
        return

    # 有効なペア（pre/post両方が揃っているもの）を抽出
    valid_pairs = []
    for (cond, set_num), paths in groups.items():
        pair_name = f"{cond}-{set_num}"
        if paths['pre'] is None or paths['post'] is None:
            print(f"Warning: Incomplete pair for {pair_name}. pre: {paths['pre']}, post: {paths['post']}. Skipped.")
            continue
            
        valid_pairs.append({
            'name': pair_name,
            'cond': cond,
            'set': set_num,
            'pre_path': paths['pre'],
            'post_path': paths['post']
        })

    # 自然な数値順ソート（例: 1-1, 1-2, ..., 2-1...）
    try:
        valid_pairs.sort(key=lambda x: (int(x['cond']), int(x['set'])))
    except ValueError:
        valid_pairs.sort(key=lambda x: x['name'])

    if args.limit is not None:
        print(f"Limiting execution to the first {args.limit} pairs.")
        valid_pairs = valid_pairs[:args.limit]

    print(f"Total pairs to process: {len(valid_pairs)}")
    os.makedirs(output_dir, exist_ok=True)

    summary_records = []

    # 各ペアのバッチ処理ループ
    for pair in tqdm(valid_pairs, desc="Processing High-Density Alignment Batch"):
        pair_name = pair['name']
        pre_img_path = pair['pre_path']
        post_img_path = pair['post_path']

        print(f"\n" + "="*60)
        print(f" Processing Pair (High-Density): {pair_name}")
        print(f"   pre : {os.path.basename(pre_img_path)}")
        print(f"   post: {os.path.basename(post_img_path)}")
        print(f""+"="*60)

        # コントラスト反転の判定
        invert = args.invert
        if not args.no_auto_invert:
            invert = auto_detect_invert(pre_img_path)
            print(f"   Auto-invert detection: {invert}")

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
            # 1. ピラー座標検出 (pre)
            print(f"   Detecting pillars in pre image (Target: Max count)...")
            df_pre = analyze_image(
                image_path=pre_img_path,
                method=args.method,
                min_area=3,
                max_area=100,
                top_hat_size=15,
                min_dist=args.min_dist,
                threshold=args.threshold,
                invert=invert
            )
            
            # 2. ピラー座標検出 (post)
            print(f"   Detecting pillars in post image (Target: Max count)...")
            df_post = analyze_image(
                image_path=post_img_path,
                method=args.method,
                min_area=3,
                max_area=100,
                top_hat_size=15,
                min_dist=args.min_dist,
                threshold=args.threshold,
                invert=invert
            )

            record['pre_pillars'] = len(df_pre)
            record['post_pillars'] = len(df_post)

            # 検出結果の保存
            pre_csv_out = os.path.join(output_dir, f"{pair_name}_pillars_pre.csv")
            post_csv_out = os.path.join(output_dir, f"{pair_name}_pillars_post.csv")
            
            df_pre.to_csv(pre_csv_out, index=False)
            df_post.to_csv(post_csv_out, index=False)

            if len(df_pre) < 10 or len(df_post) < 10:
                raise ValueError(f"Too few pillars detected (pre: {len(df_pre)}, post: {len(df_post)}). Cannot align.")

            # 3. 2段階アライメントの実行 (ICPから詳細データを取得)
            print(f"   Running registration pipeline...")
            df_aligned, H_final, iter_count, converged, dx_coarse, dy_coarse = align_and_match_dataframes(
                df_ref=df_pre,
                df_tgt=df_post,
                ref_img_path=pre_img_path,
                tgt_img_path=post_img_path,
                return_diagnostics=True
            )

            # アライメント済み座標の保存
            aligned_csv_out = os.path.join(output_dir, f"{pair_name}_aligned_to_pre.csv")
            df_aligned.to_csv(aligned_csv_out, index=False)

            # 4. アライメント品質の評価
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
            
            # マッチ率によるステータスの判定
            if match_rate < 70.0:
                record['status'] = 'WARNING'
                record['error_message'] = f'Low match rate: {match_rate:.2f}%'
                print(f"   [WARNING] Low alignment precision. Match rate: {match_rate:.2f}%")
            else:
                record['status'] = 'SUCCESS'
                print(f"   [SUCCESS] Match rate: {match_rate:.2f}%, RMSE: {rmse:.4f} px (ICP iterations: {iter_count})")

        except Exception as e:
            # 傷検出エラー等が発生しても、全体を止めずに次へ
            err_msg = str(e)
            record['status'] = 'FAILED'
            record['error_message'] = err_msg
            print(f"   [FAILED] Alignment failed: {err_msg}")

        summary_records.append(record)

    # summary.csv の書き出し
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "="*60)
    print(f" High-Density Batch processing completed!")
    print(f" Quality summary saved to: {summary_csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
