import os
import sys
import pandas as pd
import numpy as np

from registration import align_and_match_dataframes

def main():
    raw_folder = "G:/マイドライブ/1.実験データ_gdrive/5.生データ D/260630 sam dna/位置合わせ"
    outputs_dir = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/test_outputs"
    
    ref_img_path = os.path.join(raw_folder, "1-0.tif")
    ref_csv_path = os.path.join(outputs_dir, "1-0_pillars.csv")
    
    if not os.path.exists(ref_csv_path):
        print(f"Error: Reference pillars CSV '{ref_csv_path}' not found. Run run_detection_all.py first.")
        return
        
    df_ref = pd.read_csv(ref_csv_path)
    
    targets = ["1-1", "1-2", "1-3"]
    
    for tgt_name in targets:
        print(f"\n==================================================")
        print(f" アライメント開始: 基準(1-0) <--- ターゲット({tgt_name})")
        print(f"==================================================")
        
        tgt_img_path = os.path.join(raw_folder, f"{tgt_name}.tif")
        tgt_csv_path = os.path.join(outputs_dir, f"{tgt_name}_pillars.csv")
        
        if not os.path.exists(tgt_csv_path):
            print(f"Error: Target pillars CSV '{tgt_csv_path}' not found. Run run_detection_all.py first.")
            continue
            
        df_tgt = pd.read_csv(tgt_csv_path)
        
        # Run 2-stage alignment
        try:
            df_aligned, H = align_and_match_dataframes(
                df_ref=df_ref,
                df_tgt=df_tgt,
                ref_img_path=ref_img_path,
                tgt_img_path=tgt_img_path
            )
            
            # Save the aligned DataFrame
            out_csv_path = os.path.join(outputs_dir, f"{tgt_name}_aligned_to_1-0.csv")
            df_aligned.to_csv(out_csv_path, index=False)
            print(f"アライメント結果を保存しました: {out_csv_path}")
            
            # Evaluate registration quality
            dists = df_aligned['alignment_distance'].values
            matched_mask = df_aligned['matched_ref_id'] != -1
            matched_dists = dists[matched_mask]
            
            print(f"\n--- アライメント品質評価 ---")
            print(f"総ターゲットピラー数: {len(df_tgt)}")
            print(f"マッチ成功数 (距離 <= 1.5px): {len(matched_dists)} ({len(matched_dists)/len(df_tgt)*100:.2f}%)")
            
            if len(matched_dists) > 0:
                print(f"マッチペアの平均距離(RMSE): {np.sqrt(np.mean(matched_dists**2)):.4f} px")
                print(f"マッチペアの平均距離: {np.mean(matched_dists):.4f} px")
                print(f"マッチペアの中央値距離: {np.median(matched_dists):.4f} px")
                print(f"マッチペアの最大距離: {np.max(matched_dists):.4f} px")
                
                # Check pixel-level precision percentage
                matched_05 = np.sum(matched_dists < 0.5)
                print(f"サブピクセル精度内 (距離 < 0.5px) に収まる割合: {matched_05}/{len(df_tgt)} ({matched_05/len(df_tgt)*100:.2f}%)")
            else:
                print("警告: マッチングしたペアがありません。")
                
        except Exception as e:
            print(f"Error during alignment: {e}")

if __name__ == "__main__":
    main()
