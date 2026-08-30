import pandas as pd
import os
import glob

# Check the original summary Excel file
base_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"
excel_path = os.path.join(base_dir, "intensity_summary_3s_5s.xlsx")

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path, sheet_name=0)
    print("Columns:", df.columns.tolist()[:8])
    if 'N_total' in df.columns:
        print("Average N_total per FOV:", df['N_total'].mean())
else:
    print(f"Excel not found: {excel_path}")

# Check the aligned_to_pre.csv
csv_files = glob.glob(os.path.join(base_dir, "*_aligned_to_pre.csv"))
if csv_files:
    df_csv = pd.read_csv(csv_files[0])
    print(f"Rows in {os.path.basename(csv_files[0])}:", len(df_csv))
    print("Columns:", df_csv.columns.tolist())
