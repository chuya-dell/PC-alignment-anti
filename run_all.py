import subprocess

python_exe = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Scripts\python.exe"
script = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\dose_response_analyzer_qc_v3.py"
out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

runs = [
    ("SAM", "260828", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM", "SAM_QC_v3_260828"),
    ("DNA", "260828", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-dna", "DNA_QC_v3_260828"),
    ("DNA", "260827", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260827_pp50_dna", "DNA_QC_v3_260827"),
    ("SAM", "260826", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260826-p50-sam", "SAM_QC_v3_260826"),
    ("DNA", "260825", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260825_p50_dna", "DNA_QC_v3_260825"),
    ("SAM", "260824", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260824_p50_SHC6OH", "SAM_QC_v3_260824"),
    ("SAM", "2607071", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1", "SAM_QC_v3_260707_1"),
    ("SAM", "2607072", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2", "SAM_QC_v3_260707_2"),
    ("SAM", "260706", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200", "SAM_QC_v3_260706_p200"),
    ("DNA", "260822", r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna", "DNA_QC_v3_260822")
]

for assay, date, d, out in runs:
    print(f"Running {out}...")
    subprocess.run([
        python_exe, script,
        "--assay", assay,
        "--date", date,
        "--dir", d,
        "--out", f"{out_dir}\\{out}.png"
    ])
print("Done!")
