@echo off
set PYTHON=C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Scripts\python.exe
set SCRIPT=C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\dose_response_analyzer_qc_v3.py
set OUT_DIR=C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572

echo Running SAM_QC_v3_260828
%PYTHON% %SCRIPT% --assay SAM --date 260828 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM" --out "%OUT_DIR%\SAM_QC_v3_260828.png"

echo Running DNA_QC_v3_260828
%PYTHON% %SCRIPT% --assay DNA --date 260828 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-dna" --out "%OUT_DIR%\DNA_QC_v3_260828.png"

echo Running DNA_QC_v3_260827
%PYTHON% %SCRIPT% --assay DNA --date 260827 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260827_pp50_dna" --out "%OUT_DIR%\DNA_QC_v3_260827.png"

echo Running SAM_QC_v3_260826
%PYTHON% %SCRIPT% --assay SAM --date 260826 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260826-p50-sam" --out "%OUT_DIR%\SAM_QC_v3_260826.png"

echo Running DNA_QC_v3_260825
%PYTHON% %SCRIPT% --assay DNA --date 260825 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260825_p50_dna" --out "%OUT_DIR%\DNA_QC_v3_260825.png"

echo Running SAM_QC_v3_260824
%PYTHON% %SCRIPT% --assay SAM --date 260824 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260824_p50_SHC6OH" --out "%OUT_DIR%\SAM_QC_v3_260824.png"

echo Running SAM_QC_v3_260707_1
%PYTHON% %SCRIPT% --assay SAM --date 2607071 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1" --out "%OUT_DIR%\SAM_QC_v3_260707_1.png"

echo Running SAM_QC_v3_260707_2
%PYTHON% %SCRIPT% --assay SAM --date 2607072 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2" --out "%OUT_DIR%\SAM_QC_v3_260707_2.png"

echo Running SAM_QC_v3_260706_p200
%PYTHON% %SCRIPT% --assay SAM --date 260706 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200" --out "%OUT_DIR%\SAM_QC_v3_260706_p200.png"

echo Running DNA_QC_v3_260822
%PYTHON% %SCRIPT% --assay DNA --date 260822 --dir "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna" --out "%OUT_DIR%\DNA_QC_v3_260822.png"

echo All runs completed!
