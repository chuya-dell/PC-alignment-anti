import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_batch_alignment import resolve_gdrive_path

def find_alignment_pairs(data_dir):
    """
    data_dir内からアライメント済みのファイルペアを探索。
    戻り値: dict of (cond, set_num) -> {'pre_csv': path, 'aligned_csv': path}
    """
    files = os.listdir(data_dir)
    pairs = {}
    
    # aligned_to_pre.csv を探す
    for f in files:
        if f.endswith('_aligned_to_pre.csv'):
            base = f[:-19] # '_aligned_to_pre.csv' の部分をカット (例: '1-1')
            parts = base.split('-')
            if len(parts) >= 2:
                cond, set_num = parts[0], parts[1]
                key = (cond, set_num)
                
                aligned_path = os.path.join(data_dir, f)
                pre_path = os.path.join(data_dir, f"{base}_pillars_pre.csv")
                
                if os.path.exists(pre_path):
                    pairs[key] = {
                        'name': base,
                        'cond': cond,
                        'set': set_num,
                        'pre_csv': pre_path,
                        'aligned_csv': aligned_path
                    }
                else:
                    print(f"Warning: Pre CSV not found for aligned data: {f}")
                    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Pillar Intensity Comparison Pipeline (Pre vs Post SAM)")
    parser.add_argument("--data-dir", default=None, help="Directory containing alignment CSVs (default: foranti_high_density)")
    parser.add_argument("--output-dir", default=None, help="Directory to save intensity comparison results")
    
    args = parser.parse_args()
    
    # デフォルトの入力フォルダ設定 (前回の出力先)
    if args.data_dir is None:
        # F:/GoogleDrive_local のパスを構成
        data_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti_high_density"
    else:
        data_dir = resolve_gdrive_path(args.data_dir)
        
    # デフォルトの出力フォルダ設定
    if args.output_dir is None:
        parent_dir = os.path.dirname(data_dir)
        output_dir = os.path.join(parent_dir, "foranti_intensity_comparison")
    else:
        output_dir = resolve_gdrive_path(args.output_dir)
        
    print(f"Resolved Data Directory:   {data_dir}")
    print(f"Resolved Output Directory: {output_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return
        
    pairs = find_alignment_pairs(data_dir)
    if not pairs:
        print("No valid alignment CSV pairs found.")
        return
        
    valid_pairs = list(pairs.values())
    # 数値順ソート
    try:
        valid_pairs.sort(key=lambda x: (int(x['cond']), int(x['set'])))
    except ValueError:
        valid_pairs.sort(key=lambda x: x['name'])
        
    print(f"Total pairs to process: {len(valid_pairs)}")
    os.makedirs(output_dir, exist_ok=True)
    
    summary_records = []
    
    for pair in tqdm(valid_pairs, desc="Comparing Intensities"):
        name = pair['name']
        pre_csv = pair['pre_csv']
        aligned_csv = pair['aligned_csv']
        
        record = {
            'pair_name': name,
            'cond': pair['cond'],
            'set': pair['set'],
            'matched_pillars': 0,
            'mean_intensity_pre': 0.0,
            'mean_intensity_post': 0.0,
            'mean_intensity_diff': 0.0,
            'mean_intensity_ratio': 1.0,
            'median_intensity_ratio': 1.0,
            'std_intensity_diff': 0.0,
            'status': 'FAILED'
        }
        
        try:
            # 1. データの読み込み
            df_pre = pd.read_csv(pre_csv)
            df_aligned = pd.read_csv(aligned_csv)
            
            # 2. マッチしたピラーのみを抽出
            matched_df = df_aligned[df_aligned['matched_ref_id'] != -1].copy()
            
            if len(matched_df) == 0:
                raise ValueError("No matched pillars found in aligned CSV.")
                
            # 3. pre側の輝度情報をIDでマージ
            # matched_df の x, y は post のアライメント後の座標
            # df_pre の x, y は pre の元の座標
            # 分かりやすくするために列名を整理してマージ
            df_pre_sub = df_pre[['pillar_id', 'x', 'y', 'mean_intensity', 'box_intensity_3x3']].rename(
                columns={
                    'x': 'x_pre',
                    'y': 'y_pre',
                    'mean_intensity': 'mean_intensity_pre',
                    'box_intensity_3x3': 'box_intensity_3x3_pre'
                }
            )
            
            merged_df = pd.merge(
                matched_df,
                df_pre_sub,
                left_on='matched_ref_id',
                right_on='pillar_id'
            ).rename(
                columns={
                    'x': 'x_post_aligned',
                    'y': 'y_post_aligned',
                    'mean_intensity': 'mean_intensity_post',
                    'box_intensity_3x3': 'box_intensity_3x3_post'
                }
            )
            
            # 4. 輝度差および輝度比の計算
            # mean_intensity (単一ピクセルの重心値など) と box_intensity_3x3 (3x3局所平均値) の両方で計算
            merged_df['intensity_diff'] = merged_df['mean_intensity_post'] - merged_df['mean_intensity_pre']
            merged_df['intensity_ratio'] = merged_df['mean_intensity_post'] / merged_df['mean_intensity_pre']
            
            merged_df['box_intensity_diff_3x3'] = merged_df['box_intensity_3x3_post'] - merged_df['box_intensity_3x3_pre']
            merged_df['box_intensity_ratio_3x3'] = merged_df['box_intensity_3x3_post'] / merged_df['box_intensity_3x3_pre']
            
            # 保存するカラムの選択と整理
            out_cols = [
                'matched_ref_id', 'x_pre', 'y_pre', 'x_post_aligned', 'y_post_aligned', 'alignment_distance',
                'mean_intensity_pre', 'mean_intensity_post', 'intensity_diff', 'intensity_ratio',
                'box_intensity_3x3_pre', 'box_intensity_3x3_post', 'box_intensity_diff_3x3', 'box_intensity_ratio_3x3'
            ]
            final_df = merged_df[out_cols]
            
            # 5. CSVファイルへ書き出し
            out_csv_path = os.path.join(output_dir, f"{name}_intensity_diff.csv")
            final_df.to_csv(out_csv_path, index=False)
            
            # 6. 統計情報の集計
            record['matched_pillars'] = len(final_df)
            record['mean_intensity_pre'] = float(final_df['mean_intensity_pre'].mean())
            record['mean_intensity_post'] = float(final_df['mean_intensity_post'].mean())
            record['mean_intensity_diff'] = float(final_df['intensity_diff'].mean())
            record['mean_intensity_ratio'] = float(final_df['intensity_ratio'].mean())
            record['median_intensity_ratio'] = float(final_df['intensity_ratio'].median())
            record['std_intensity_diff'] = float(final_df['intensity_diff'].std())
            record['status'] = 'SUCCESS'
            
        except Exception as e:
            print(f"Error processing pair {name}: {e}")
            record['status'] = 'FAILED'
            
        summary_records.append(record)
        
    # summaryファイルの保存
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "intensity_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "="*60)
    print(" Intensity comparison batch processing completed!")
    print(f" Summary saved to: {summary_csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
