import subprocess
import os

python_exe = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Scripts\python.exe"
script = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\dose_response_analyzer_v4.py"
out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

runs = [
    # 1. p200 with mask
    ("SAM", "260706", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200", "p200_mask_ON", False),
    # 2. p200 without mask
    ("SAM", "260706", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200", "p200_mask_OFF", True),
    # 3. p50 with mask (using 260828 SAM or DNA? Let's use 260828 SAM)
    ("SAM", "260828", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM", "p50_mask_ON", False),
    # 4. p50 without mask
    ("SAM", "260828", r"G:\マイドライブ\1.実験データ_gdrive\4.生Forceデータ\4.生データ D\260828-p50-SAM", "p50_mask_OFF", True)
]

for assay, date, d, name, no_mask in runs:
    # fix typo in 260828-p50-SAM path
    if "Force" in d:
        d = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"
        
    cmd = [
        python_exe, script,
        "--assay", assay,
        "--date", date,
        "--dir", d,
        "--out", f"{out_dir}\\{name}.png"
    ]
    if no_mask:
        cmd.append("--no-mask")
        
    print(f"\n================ Running {name} ================")
    subprocess.run(cmd)
    
print("All Done!")
