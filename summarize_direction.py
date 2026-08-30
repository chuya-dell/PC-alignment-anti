import pandas as pd
import numpy as np
import os
import glob

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
excel_files = glob.glob(os.path.join(out_dir, "*.xlsx"))
out_md = os.path.join(out_dir, "Grid_Percent_Summary.md")

with open(out_md, "w", encoding="utf-8") as f_out:
    f_out.write("# 実験データ要約（Grid % 値一覧）\n\n")
    f_out.write("閾値（ブランク平均 - 3σ）を超えて「有意に暗くなった」ピラーの割合（**Neg Grid%**）を中心にまとめた表です。\n")
    f_out.write("各値は、複数の視野（FOV）における平均値 ± 標準偏差（SD）を示しています。\n\n")
    
    for f in excel_files:
        if "_negative" in f: continue
        name = os.path.basename(f).replace('.xlsx', '')
        f_out.write(f"## {name}\n")
        
        try:
            xl = pd.ExcelFile(f)
            blank_sheet = None
            for s in xl.sheet_names:
                if "Blank" in s or "0 M" in s:
                    blank_sheet = s
                    break
            
            mean_b, std_b = 0, 0
            if blank_sheet:
                df_blank = xl.parse(blank_sheet)
                all_blanks = []
                for col in df_blank.columns:
                    all_blanks.extend(df_blank[col].dropna().values)
                if len(all_blanks) > 0:
                    mean_b = np.mean(all_blanks)
                    std_b = np.std(all_blanks)
                    
            thresh_pos = mean_b + 3 * std_b
            thresh_neg = mean_b - 3 * std_b
            
            f_out.write(f"Blank (All pillars) -> Mean: {mean_b:.3f}%, SD: {std_b:.3f}% (Thresh: < {thresh_neg:.3f}%)\n\n")
            f_out.write("| Condition | Mean Delta ± SD (%) | Pos Grid% ± SD (%) | Neg Grid% ± SD (%) |\n")
            f_out.write("| :--- | :--- | :--- | :--- |\n")
                    
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                grid_pos = []
                grid_neg = []
                means = []
                for col in df.columns:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        means.append(np.mean(vals))
                        grid_pos.append(np.sum(vals > thresh_pos) / len(vals) * 100.0)
                        grid_neg.append(np.sum(vals < thresh_neg) / len(vals) * 100.0)
                
                om = np.mean(means) if means else 0
                om_sd = np.std(means) if means else 0
                ogp = np.mean(grid_pos) if grid_pos else 0
                ogp_sd = np.std(grid_pos) if grid_pos else 0
                ogn = np.mean(grid_neg) if grid_neg else 0
                ogn_sd = np.std(grid_neg) if grid_neg else 0
                
                f_out.write(f"| {sheet} | {om:.3f} ± {om_sd:.3f} | {ogp:.2f} ± {ogp_sd:.2f} | {ogn:.2f} ± {ogn_sd:.2f} |\n")
            f_out.write("\n")
                
        except Exception as e:
            f_out.write(f"Error reading: {e}\n\n")
