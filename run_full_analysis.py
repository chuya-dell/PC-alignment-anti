import os
import sys
import subprocess
import argparse
from path_resolver import resolve_gdrive_path

def run_dataset_pipeline(input_dir, output_base_dir, limit=None):
    input_path = resolve_gdrive_path(input_dir)
    if not os.path.exists(input_path):
        print(f"Error: Input dataset directory does not exist: {input_dir}")
        return False
        
    dataset_name = os.path.basename(input_path.rstrip('/\\'))
    output_dir = resolve_gdrive_path(os.path.join(output_base_dir, dataset_name), create_if_missing=True)
    
    pillar_out = os.path.join(output_dir, "pillar_analysis").replace('\\', '/')
    grid_out = os.path.join(output_dir, "grid_analysis").replace('\\', '/')
    
    os.makedirs(pillar_out, exist_ok=True)
    os.makedirs(grid_out, exist_ok=True)
    
    python_exe = sys.executable
    
    print("=" * 70)
    print(f" STARTING FULL PIPELINE FOR DATASET: {dataset_name}")
    print(f"   Input Directory : {input_path}")
    print(f"   Output Directory: {output_dir}")
    print("=" * 70)
    
    # 1. Pillar Alignment Batch
    cmd1 = [python_exe, "run_batch_alignment.py", "--input-dir", input_path, "--output-dir", pillar_out]
    if limit:
        cmd1.extend(["--limit", str(limit)])
    print(f"\n[Step 1/5] Running Pillar Alignment...")
    res1 = subprocess.run(cmd1)
    if res1.returncode != 0:
        print(f"Error in Step 1 for dataset {dataset_name}")
        return False
        
    # 2. Pillar Intensity Comparison
    cmd2 = [python_exe, "run_intensity_comparison.py", "--data-dir", pillar_out, "--output-dir", pillar_out]
    print(f"\n[Step 2/5] Running Pillar Intensity Comparison...")
    res2 = subprocess.run(cmd2)
    
    # 3. Export Pillar Excel Report
    cmd3 = [python_exe, "export_intensity_excel.py", pillar_out]
    print(f"\n[Step 3/5] Exporting Pillar Intensity Excel Report...")
    res3 = subprocess.run(cmd3)
    
    # 4. Grid Difference Analysis
    cmd4 = [python_exe, "run_grid_difference_analysis.py", "--img-dir", input_path, "--pillar-dir", pillar_out, "--output-dir", grid_out]
    print(f"\n[Step 4/5] Running Grid Difference Analysis & Heatmap Generation...")
    res4 = subprocess.run(cmd4)
    
    # 5. Export Grid Excel Report
    cmd5 = [python_exe, "export_grid_intensity_excel.py", grid_out]
    print(f"\n[Step 5/5] Exporting Grid Intensity Excel Report...")
    res5 = subprocess.run(cmd5)
    
    print("\n" + "=" * 70)
    print(f" COMPLETED PIPELINE FOR DATASET: {dataset_name}")
    print(f" Results saved in: {output_dir}")
    print("=" * 70)
    return True

def main():
    parser = argparse.ArgumentParser(description="Master Analysis Pipeline for Plasmonic Crystal Datasets")
    parser.add_argument("--input-dir", required=True, help="Input directory containing raw TIFF images")
    parser.add_argument("--output-base", default="G:/マイドライブ/5.解析結果_remo", help="Base output directory for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs (for testing)")
    args = parser.parse_args()
    
    run_dataset_pipeline(args.input_dir, args.output_base, args.limit)

if __name__ == "__main__":
    main()
