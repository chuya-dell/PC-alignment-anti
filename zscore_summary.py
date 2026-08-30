import pandas as pd
import numpy as np
import os
import glob

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
excel_files = glob.glob(os.path.join(out_dir, "*.xlsx"))
out_md = os.path.join(out_dir, "ZScore_Summary.md")

with open(out_md, "w", encoding="utf-8") as f_out:
    f_out.write("# 実験データ再評価（Z値 ＆ n数フィルター版）\n\n")
    f_out.write("ブランクのばらつき（σ）が日ごとに異なる問題を解消するため、各濃度の平均輝度変化を **Z値（Z-score）** に変換しました。\n")
    f_out.write("計算式: `Z = (Condition Mean - Blank Mean) / Blank Std`\n")
    f_out.write("※暗くなる反応（マイナス方向）がシグナルである場合、**Z値がマイナスに大きいほど強いシグナル** を意味します。\n")
    f_out.write("※視野数（FOV）が 1 の条件は、異常値（アーティファクト）の可能性が高いため `[n=1 除外]` としてフラグを立てています。\n\n")
    
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
            
            if std_b == 0:
                f_out.write("Blank Std is 0, cannot calculate Z-score.\n\n")
                continue
                
            f_out.write(f"Blank stats -> Mean: {mean_b:.3f}%, SD: {std_b:.3f}%\n\n")
            f_out.write("| Condition | FOV (n) | Mean Delta | Z-Score (vs Blank) | Neg Grid% |\n")
            f_out.write("| :--- | :--- | :--- | :--- | :--- |\n")
                    
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                grid_neg = []
                means = []
                for col in df.columns:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        means.append(np.mean(vals))
                        grid_neg.append(np.sum(vals < (mean_b - 3*std_b)) / len(vals) * 100.0)
                
                n = len(means)
                if n == 0: continue
                
                om = np.mean(means)
                ogn = np.mean(grid_neg)
                z_score = (om - mean_b) / std_b
                
                # Flag n=1
                if n == 1:
                    f_out.write(f"| {sheet} | **{n} [n=1 除外]** | {om:.3f}% | **{z_score:.2f}** | {ogn:.2f}% |\n")
                else:
                    # Highlight significant Z-scores (e.g. < -1.0)
                    z_str = f"**{z_score:.2f}**" if z_score < -1.0 else f"{z_score:.2f}"
                    f_out.write(f"| {sheet} | {n} | {om:.3f}% | {z_str} | {ogn:.2f}% |\n")
            f_out.write("\n")
                
        except Exception as e:
            f_out.write(f"Error reading: {e}\n\n")
