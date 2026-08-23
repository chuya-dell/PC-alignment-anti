import os
import glob
import argparse
import time
from voronoi_analyzer import analyze_voronoi
from generate_excel_summary import analyze_intensities_excel

def run_batch_analysis(target_dir, sigmas=[3, 5], margin=50):
    """
    Run BOTH Voronoi Spatial Analysis AND Background (0-series) Excel Summary
    on all CSV files in the target directory.
    """
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
    # Filter out generated summary CSVs
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(("voronoi_", "summary_"))]

    print(f"\n========================================================")
    print(f" 統合自動解析（ボロノイ解析 ＋ バックグラウンド集計）")
    print(f" 対象フォルダ: {os.path.abspath(target_dir)}")
    print(f" 検出CSV数  : {len(csv_files)} 件")
    print(f"========================================================\n")

    t_start = time.time()

    # 1. 各CSVファイルに対してボロノイ解析を実行
    print("--- [1/2] 各連番データのボロノイ空間解析を実行中 ---")
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        print(f"\n>> ボロノイ解析中: {base_name}")
        out_sub_dir = os.path.join(target_dir, "voronoi_" + os.path.splitext(base_name)[0])
        try:
            analyze_voronoi(csv_path=csv_file, output_dir=out_sub_dir, margin=margin)
        except Exception as e:
            print(f"   ボロノイ解析の注意 ({base_name}): {e}")

    # 2. バックグラウンド (0番台) 基準のエクセル集計を出力 (3σ, 5σ)
    print("\n--- [2/2] バックグラウンド (0番台) 基準の Mean + 3σ/5σ エクセル集計を出力中 ---")
    excel_path = os.path.join(target_dir, "intensity_summary_3s_5s.xlsx")
    try:
        analyze_intensities_excel(input_dir=target_dir, output_excel_path=excel_path, sigmas=sigmas)
    except Exception as e:
        print(f" エクセル集計の注意: {e}")

    t_total = time.time() - t_start
    print(f"\n========================================================")
    print(f" 全統合解析が完了しました！（所要時間: {t_total:.2f} 秒）")
    print(f" 1. ボロノイ解析結果 : 各 CSV 用サブフォルダ")
    print(f" 2. エクセル集計表   : {os.path.abspath(excel_path)}")
    print(f"========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BOTH Voronoi Spatial Analysis & Background (0-series) Excel Summary")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing sequence CSV files")
    parser.add_argument("--sigmas", type=int, nargs="+", default=[3, 5], help="n values for Mean + n*sigma (default: 3 5)")
    parser.add_argument("--margin", type=int, default=50, help="ROI boundary margin in pixels for Voronoi (default: 50)")

    args = parser.parse_args()
    run_batch_analysis(args.dir, sigmas=args.sigmas, margin=args.margin)
