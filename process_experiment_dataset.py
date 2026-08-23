import os
import glob
import re
import argparse
import time
import pandas as pd
import numpy as np

from analyzer import analyze_image
from registration import align_and_match_dataframes, create_grid_estimated_dataframe
from voronoi_analyzer import analyze_voronoi
from generate_excel_summary import analyze_intensities_excel

def is_bg_file_name(filename, bg_prefixes):
    base = os.path.basename(filename)
    str_prefixes = [str(b).strip() for b in bg_prefixes]
    for bg in str_prefixes:
        # Match pattern like "13-1-0.tif" where 13 is the major series number
        pattern = rf'^{re.escape(bg)}[\-_\s\.]|(?:^|[\-_bB][gG]?|No\.?)\s*{re.escape(bg)}(?:[\-_\s\.]|$)'
        if re.search(pattern, base, re.IGNORECASE) or base.startswith(bg + "-") or base.startswith(bg + "_"):
            return True
    return False

def process_full_experiment_folder(exp_dir, bg_prefixes, pitch=5.5):
    """
    Complete End-to-End Pipeline for Plasmon Pillar Arrays:
    1. Pillar Detection on all TIF images
    2. Alignment & Grid Estimation against reference BG image
    3. Voronoi Spatial & Coordination Analysis
    4. Pattern A & Pattern B 3sigma/5sigma Excel Summary Generation
    """
    t_start = time.time()
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    exp_name = os.path.basename(exp_dir)
    print(f"\n========================================================")
    print(f" 実験データ全自動一括解析パイプライン")
    print(f" 対象フォルダ: {exp_dir}")
    print(f" 基準識別子  : {bg_prefixes}")
    print(f"========================================================\n")

    # Find all TIF images
    valid_exts = ('.tif', '.tiff', '.png')
    all_files = sorted(os.listdir(exp_dir))
    image_files = [f for f in all_files if f.lower().endswith(valid_exts) and not f.endswith(('_overlay.png', '_histogram.png', '_heatmap.png'))]

    print(f"検出TIF画像数: {len(image_files)} 件")

    if not image_files:
        print("エラー: 処理対象の画像ファイルが見つかりません。")
        return

    # Step 1: Run Pillar Detection on all TIF images
    print("\n--- [STEP 1/4] 全画像のピラー検出処理 (analyzer.py) ---")
    detected_csvs = {}
    for img_file in image_files:
        img_path = os.path.join(exp_dir, img_file)
        base_name, _ = os.path.splitext(img_file)
        csv_path = os.path.join(exp_dir, f"{base_name}_pillars.csv")
        
        if os.path.exists(csv_path):
            print(f"  [既存読み込み] {base_name}_pillars.csv")
            df = pd.read_csv(csv_path)
        else:
            print(f"  [検出中] {img_file} ...")
            df = analyze_image(
                image_path=img_path,
                tile_size=2000,
                overlap=50,
                method="peak",
                min_dist=4,
                top_hat_size=15
            )
            df.to_csv(csv_path, index=False)
            print(f"  -> 保存完了 ({len(df):,} ピラー検出)")

        detected_csvs[img_file] = csv_path

    # Step 2: Group background (reference) files and target files
    bg_img_files = [f for f in image_files if is_bg_file_name(f, bg_prefixes)]
    tgt_img_files = [f for f in image_files if f not in bg_img_files]

    print(f"\n--- 基準(BG)画像: {len(bg_img_files)} 件 / 測定(Target)画像: {len(tgt_img_files)} 件 ---")

    if not bg_img_files:
        print(f"警告: 基準識別子 {bg_prefixes} に該当する背景画像が見つかりません。全画像でアライメントを試みます。")
        ref_img_file = image_files[0]
    else:
        ref_img_file = bg_img_files[0] # Pick primary reference image

    print(f"メインリファレンス画像: {ref_img_file}")
    ref_img_path = os.path.join(exp_dir, ref_img_file)
    ref_csv_path = detected_csvs[ref_img_file]
    df_ref = pd.read_csv(ref_csv_path)

    # Step 3: Registration & Alignment (Pattern A & Pattern B generation)
    print("\n--- [STEP 2/4] 位置合わせ・全基準格子投影 (registration.py) ---")
    for tgt_file in image_files:
        base_name, _ = os.path.splitext(tgt_file)
        tgt_img_path = os.path.join(exp_dir, tgt_file)
        tgt_csv_path = detected_csvs[tgt_file]
        
        aligned_csv_path = os.path.join(exp_dir, f"{base_name}_aligned.csv")
        estimated_csv_path = os.path.join(exp_dir, f"{base_name}_estimated_grid.csv")

        if os.path.exists(aligned_csv_path) and os.path.exists(estimated_csv_path):
            print(f"  [既存読み込み] {base_name} アライメントCSV完了済み")
            continue

        if tgt_file == ref_img_file:
            # Self-reference
            df_tgt = pd.read_csv(tgt_csv_path)
            df_tgt['matched_ref_id'] = df_tgt['pillar_id'] if 'pillar_id' in df_tgt.columns else np.arange(len(df_tgt))
            df_tgt['alignment_distance'] = 0.0
            df_tgt.to_csv(aligned_csv_path, index=False)
            
            # Pattern B self
            df_tgt['is_estimated'] = True
            df_tgt.to_csv(estimated_csv_path, index=False)
            continue

        print(f"  [アライメント処理] 基準({ref_img_file}) <-- {tgt_file}")
        try:
            df_tgt = pd.read_csv(tgt_csv_path)
            df_aligned, H_final = align_and_match_dataframes(
                df_ref=df_ref,
                df_tgt=df_tgt,
                ref_img_path=ref_img_path,
                tgt_img_path=tgt_img_path
            )
            df_aligned.to_csv(aligned_csv_path, index=False)

            # Pattern B: Estimate all reference grid coordinates on target image
            df_est = create_grid_estimated_dataframe(
                df_ref=df_ref,
                H_final=H_final,
                tgt_img_path=tgt_img_path
            )
            df_est.to_csv(estimated_csv_path, index=False)

        except Exception as e:
            print(f"  [アライメントスキップ/注意] {tgt_file}: {e}")

    # Step 4: Voronoi Spatial Analysis
    print("\n--- [STEP 3/4] ボロノイ空間解析 & 欠陥ヒートマップ (voronoi_analyzer.py) ---")
    all_csvs_for_voronoi = sorted(glob.glob(os.path.join(exp_dir, "*_aligned.csv")) + glob.glob(os.path.join(exp_dir, "*_estimated_grid.csv")))
    if not all_csvs_for_voronoi:
        all_csvs_for_voronoi = list(detected_csvs.values())

    for cpath in all_csvs_for_voronoi:
        cbase = os.path.splitext(os.path.basename(cpath))[0]
        v_out_dir = os.path.join(exp_dir, f"voronoi_{cbase}")
        try:
            analyze_voronoi(csv_path=cpath, output_dir=v_out_dir, margin=50)
        except Exception as e:
            print(f"  ボロノイ解析注意 ({cbase}): {e}")

    # Step 5: Excel Summary Generation (Pattern A & Pattern B side-by-side)
    print("\n--- [STEP 4/4] 3σ/5σ エクセル集計表の生成 (generate_excel_summary.py) ---")
    excel_out_path = os.path.join(exp_dir, "intensity_summary_3s_5s.xlsx")
    try:
        analyze_intensities_excel(
            input_dir=exp_dir,
            output_excel_path=excel_out_path,
            sigmas=[3, 5],
            bg_prefixes=bg_prefixes
        )
    except Exception as e:
        print(f" エクセル生成注意: {e}")

    t_total = time.time() - t_start
    print(f"\n========================================================")
    print(f" 実験データ {exp_name} の全自動解析完了！（所要時間: {t_total:.2f} 秒）")
    print(f" エクセル集計表: {excel_out_path}")
    print(f"========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Plasmon Pillar Array Dataset Processing")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing experiment TIF images")
    parser.add_argument("--bg", type=str, nargs="+", required=True, help="Background sequence identifiers (e.g. 13 or 10 11 or 8)")

    args = parser.parse_args()
    process_full_experiment_folder(args.dir, bg_prefixes=args.bg)
