$python = "C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\.venv\Scripts\python.exe"
$script = "C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\dose_response_analyzer_qc_v3.py"
$out_dir = "C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

function Run-Analyzer ($assay, $date, $dir, $out_name) {
    Write-Host "Running $out_name ..."
    $args = @(
        $script,
        "--assay", $assay,
        "--date", $date,
        "--dir", "`"$dir`"",
        "--out", "`"$out_dir\$out_name.png`""
    )
    $proc = Start-Process -FilePath $python -ArgumentList $args -NoNewWindow -Wait -PassThru
    Write-Host "Exit code: $($proc.ExitCode)"
}

Run-Analyzer "SAM" "260828" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM" "SAM_QC_v3_260828"
Run-Analyzer "DNA" "260828" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-dna" "DNA_QC_v3_260828"
Run-Analyzer "DNA" "260827" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260827_pp50_dna" "DNA_QC_v3_260827"
Run-Analyzer "SAM" "260826" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260826-p50-sam" "SAM_QC_v3_260826"
Run-Analyzer "DNA" "260825" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260825_p50_dna" "DNA_QC_v3_260825"
Run-Analyzer "SAM" "260824" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260824_p50_SHC6OH" "SAM_QC_v3_260824"
Run-Analyzer "SAM" "2607071" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1" "SAM_QC_v3_260707_1"
Run-Analyzer "SAM" "2607072" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2" "SAM_QC_v3_260707_2"
Run-Analyzer "SAM" "260706" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200" "SAM_QC_v3_260706_p200"
Run-Analyzer "DNA" "260822" "G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna" "DNA_QC_v3_260822"

Write-Host "All runs completed!"
