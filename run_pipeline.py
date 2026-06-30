import os
import sys
import argparse
import time

# Add the script's directory to sys.path to allow imports on virtual/synced drives
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from analyzer import analyze_image
from visualize import create_detection_overlay
from summarize import generate_summary

def run_pipeline():
    parser = argparse.ArgumentParser(description="Plasmon Analysis Pipeline Orchestrator")
    parser.add_argument("--dir", type=str, required=True, help="Path to the experiment folder (e.g., experiments/exp_01)")
    parser.add_argument("--image", type=str, default=None, help="Filename of the image. If omitted, auto-detects the single image in the folder.")
    
    # Analysis arguments
    parser.add_argument("--method", type=str, default="peak", choices=["blob", "peak"], help="Detection method: 'blob' or 'peak' (default: peak)")
    parser.add_argument("--min-dist", type=int, default=4, help="Minimum peak distance in pixels (default: 4)")
    parser.add_argument("--min-area", type=int, default=3, help="Min area for blob detection (default: 3)")
    parser.add_argument("--max-area", type=int, default=100, help="Max area for blob detection (default: 100)")
    parser.add_argument("--top-hat", type=int, default=15, help="Background top-hat filter size (default: 15)")
    parser.add_argument("--threshold", type=float, default=None, help="Manual intensity threshold (default: auto)")
    
    # Visualization arguments
    parser.add_argument("--radius", type=int, default=1, help="Overlay dot radius (default: 1)")
    parser.add_argument("--thickness", type=int, default=-1, help="Overlay dot thickness (-1 for solid, default: -1)")
    parser.add_argument("--crop-size", type=int, default=800, help="Size of visual overlay crop (default: 800)")
    
    # Summary arguments
    parser.add_argument("--size", type=float, default=1.0, help="Marker size for the heatmap plot (default: 1.0)")
    
    args = parser.parse_args()
    
    # Setup paths and output naming based on experiment folder name
    exp_dir = args.dir
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    
    # Auto-detect image file if not specified
    image_file = args.image
    if image_file is None:
        valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Error: Experiment directory '{exp_dir}' does not exist.")
            
        all_files = os.listdir(exp_dir)
        image_files = [
            f for f in all_files 
            if f.lower().endswith(valid_extensions) and 
            not any(suffix in f.lower() for suffix in ('_overlay.png', '_histogram.png', '_heatmap.png'))
        ]
        
        if len(image_files) == 1:
            image_file = image_files[0]
            print(f"自動検出: フォルダ内に画像ファイルが1つだけ見つかったため、'{image_file}' を使用します。")
        elif len(image_files) > 1:
            # Try to match folder name
            matched = [f for f in image_files if os.path.splitext(f)[0] == exp_name]
            if len(matched) == 1:
                image_file = matched[0]
                print(f"自動検出: フォルダ名と一致する画像 '{image_file}' を使用します。")
            else:
                raise ValueError(
                    f"エラー: フォルダ '{exp_dir}' 内に複数の画像が見つかりました {image_files}。\n"
                    f"--image 引数で画像ファイル名を明示的に指定してください。"
                )
        else:
            raise FileNotFoundError(
                f"エラー: フォルダ '{exp_dir}' 内に画像ファイル ({', '.join(valid_extensions)}) が見つかりません。"
            )
            
    image_path = os.path.join(exp_dir, image_file)
    
    # Extract the base image name without extension for output prefixes
    image_name, _ = os.path.splitext(image_file)
    
    csv_path = os.path.join(exp_dir, f"{image_name}_results.csv")
    overlay_path = os.path.join(exp_dir, f"{image_name}_overlay.png")
    hist_path = os.path.join(exp_dir, f"{image_name}_histogram.png")
    heatmap_path = os.path.join(exp_dir, f"{image_name}_heatmap.png")
    
    # Validation
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
            
    print(f"\n================ パイプライン処理開始 ================")
    print(f"対象フォルダ: {exp_dir}")
    print(f"解析画像   : {image_path}")
    print(f"====================================================\n")
    
    t_start = time.time()
    
    # Step 1: Run Analysis
    print("[STEP 1/3] 画像解析を実行中...")
    df_results = analyze_image(
        image_path=image_path,
        tile_size=2000,
        overlap=50,
        min_area=args.min_area,
        max_area=args.max_area,
        top_hat_size=args.top_hat,
        method=args.method,
        min_dist=args.min_dist,
        threshold=args.threshold
    )
    
    # Save CSV
    df_results.to_csv(csv_path, index=False)
    print(f"-> 解析データを保存しました: {csv_path}")
    
    if df_results.empty:
        print("ピラーが検出されなかったため、可視化と集計をスキップします。")
        return
        
    # Step 2: Create Overlay Visualization
    print("\n[STEP 2/3] 検出位置の重ね合わせ画像を作成中...")
    create_detection_overlay(
        image_path=image_path,
        csv_path=csv_path,
        output_path=overlay_path,
        crop=True,
        crop_size=args.crop_size,
        radius=args.radius,
        thickness=args.thickness
    )
    
    # Step 3: Generate Summary statistics & plots
    print("\n[STEP 3/3] 統計結果およびグラフ（ヒストグラム・ヒートマップ）を生成中...")
    generate_summary(
        csv_path=csv_path,
        histogram_path=hist_path,
        heatmap_path=heatmap_path,
        marker_size=args.size
    )
    
    t_total = time.time() - t_start
    print(f"\n================ パイプライン処理完了 ================")
    print(f"総処理時間: {t_total:.2f} 秒")
    print(f"出力ファイルはすべて以下のフォルダに保存されました:")
    print(f"-> {os.path.abspath(exp_dir)}")
    print(f"====================================================\n")

if __name__ == "__main__":
    run_pipeline()
