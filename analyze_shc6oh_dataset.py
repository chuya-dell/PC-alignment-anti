import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

input_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\2060824_p50_SHC6OH"

# Concentration mapping from user notes
CONC_MAP = {
    '0':  ('0 M (Blank)', 0.0),
    '14': ('0 M (Blank 2)', 0.0),
    '10': ('1 fM', 1e-15),
    '8':  ('10 fM', 1e-14),
    '13': ('10 fM (Dup)', 1e-14),
    '5':  ('100 fM', 1e-13),
    '4':  ('1 pM', 1e-12),
    '12': ('1 pM (Dup)', 1e-12),
    '3':  ('10 pM', 1e-11),
    '11': ('10 pM (Leak?)', 1e-11),
    '2':  ('100 pM (Leak?)', 1e-10),
    '1':  ('1 nM', 1e-9)
}

# Defect observation notes from user (Sample-Trial)
DEFECT_NOTES = {
    '0-1': '染みとごみ',
    '0-4': 'ごみ',
    '0-5': 'ごみ',
    '0-6': 'ごみ',
    '0-7': '染み',
    '0-8': '染みとごみ（巨大大ゴミ）',
    '3-4': 'ごみ',
    '3-6': '染み（大）',
    '3-7': '染み',
    '3-8': '染み',
    '4-1': '染み',
    '4-2': 'ごみ',
    '4-5': 'ごみ',
    '4-6': '染み',
    '8-1': '染みとごみ',
    '8-5': '染み',
    '8-7': 'ごみ',
    '8-8': 'ごみと染み',
    '10-2': 'ごみと染み',
    '10-4': 'ごみ',
    '10-6': '染み',
    '10-8': '染みとごみ',
    '11-1': 'ごみ',
    '11-2': 'ごみ',
    '11-3': '染み',
    '11-4': '傷',
    '12-8': '染み',
    '13-4': 'ごみ',
    '14-1': '染み',
    '14-6': '染み'
}

def is_defect_file(filename):
    # Match pattern like "0-1-0_aligned.csv" or "3-4-1_estimated_grid.csv"
    m = re.match(r'^(\d+)\-(\d+)', os.path.basename(filename))
    if m:
        key = f"{m.group(1)}-{m.group(2)}"
        if key in DEFECT_NOTES:
            return DEFECT_NOTES[key]
    return None

def process_shc6oh_full():
    aligned_csvs = sorted(glob.glob(os.path.join(input_dir, "*_aligned.csv")))
    estimated_csvs = sorted(glob.glob(os.path.join(input_dir, "*_estimated_grid.csv")))
    
    if not aligned_csvs:
        print("CSVファイルがまだ生成されていません。処理待機中...")
        return
        
    # Collect clean BG values (Series 0 and 14 without defects)
    bg_vals = []
    bg_files = [f for f in aligned_csvs if os.path.basename(f).startswith(("0-", "14-"))]
    
    for f in bg_files:
        defect_reason = is_defect_file(f)
        if defect_reason:
            continue
        df_bg = pd.read_csv(f)
        bg_vals.extend(df_bg['mean_intensity'].dropna().values)
        
    bg_arr = np.array(bg_vals)
    bg_mean = np.mean(bg_arr)
    bg_std = np.std(bg_arr)
    thresh_3s = bg_mean + 3 * bg_std
    thresh_5s = bg_mean + 5 * bg_std
    
    print("\n=======================================================")
    print(" ■ 2060824_p50_SHC6OH ブランク基準値（クリーンデータ抽出）")
    print("=======================================================")
    print(f"  ・使用ブランククリーン数: {len(bg_files) - len([f for f in bg_files if is_defect_file(f)])} 枚")
    print(f"  ・集計基準ピラー総数    : {len(bg_arr):,} 個")
    print(f"  ・ブランク平均輝度 (μ) : {bg_mean:.4f}")
    print(f"  ・ブランク標準偏差 (σ) : {bg_std:.4f}")
    print(f"  ・判定閾値 (μ + 3σ)   : {thresh_3s:.4f}")
    print(f"  ・判定閾値 (μ + 5σ)   : {thresh_5s:.4f}")
    
    all_csvs = sorted(aligned_csvs + estimated_csvs)
    summary_rows = []
    audit_rows = []
    
    for cpath in all_csvs:
        fname = os.path.basename(cpath)
        series = re.split(r'[_\-\.]', fname)[0]
        
        if 'aligned' in fname:
            pattern = 'Pattern A (実測同定ペア)'
        elif 'estimated_grid' in fname:
            pattern = 'Pattern B (全基準格子推定)'
        else:
            pattern = '単一画像全検出'
            
        df = pd.read_csv(cpath)
        vals = df['mean_intensity'].dropna().values
        n_total = len(vals)
        
        if n_total == 0:
            continue
            
        mu = np.mean(vals)
        std = np.std(vals)
        delta_m = mu - bg_mean
        
        c3s = np.sum(vals > thresh_3s)
        r3s = (c3s / n_total) * 100
        
        c5s = np.sum(vals > thresh_5s)
        r5s = (c5s / n_total) * 100
        
        row = {
            'ファイル名': fname,
            'サンプル系列': series,
            '評価パターン': pattern,
            '規格化母数 [総ピラー数 N_total]': n_total,
            '平均輝度 (μ)': mu,
            '輝度標準偏差 (σ)': std,
            'BG比輝度変化 (ΔMean)': delta_m,
            '3σ超過数 [個]': c3s,
            '3σ超過規格化割合 [% = (超過数/N_total)*100]': r3s,
            '5σ超過数 [個]': c5s,
            '5σ超過規格化割合 [% = (超過数/N_total)*100]': r5s
        }
        
        summary_rows.append(row)
        
        # Check exclusion rules
        defect_note = is_defect_file(fname)
        if defect_note:
            audit_rows.append({
                '対象ファイル名': fname,
                'サンプル系列': series,
                '評価パターン': pattern,
                '実測平均輝度': mu,
                '除外基準カテゴリー': '【基準1】実験ノート欠陥記録（ゴミ・染み・傷）',
                '定量的根拠・数理証明': f'実験記録に明記: {defect_note}'
            })
        elif 'Pattern A' in pattern and n_total < 500:
            audit_rows.append({
                '対象ファイル名': fname,
                'サンプル系列': series,
                '評価パターン': pattern,
                '実測平均輝度': mu,
                '除外基準カテゴリー': '【基準2】アライメント幾何学的不全（同定数極小）',
                '定量的根拠・数理証明': f'1対1マッチ数 N={n_total} < 500'
            })
        elif delta_m < -5.0:
            audit_rows.append({
                '対象ファイル名': fname,
                'サンプル系列': series,
                '評価パターン': pattern,
                '実測平均輝度': mu,
                '除外基準カテゴリー': '【基準3】撮影不全・コントラスト極度低下（焦点ズレ）',
                '定量的根拠・数理証明': f'ネガティブコントロール比 ΔMean={delta_m:.2f} < -5.0'
            })
            
    df_summary = pd.DataFrame(summary_rows)
    df_audit = pd.DataFrame(audit_rows)
    
    # Filter clean rows
    excluded_files = set(df_audit['対象ファイル名'])
    df_clean = df_summary[~df_summary['ファイル名'].isin(excluded_files)].copy()
    
    # Add concentration details
    df_clean['濃度表記'] = df_clean['サンプル系列'].apply(lambda s: CONC_MAP.get(str(s), ('Unknown', None))[0])
    df_clean['濃度(M)'] = df_clean['サンプル系列'].apply(lambda s: CONC_MAP.get(str(s), ('Unknown', None))[1])
    
    df_series_clean = df_clean.groupby(['サンプル系列', '濃度表記', '濃度(M)', '評価パターン']).agg({
        '規格化母数 [総ピラー数 N_total]': 'mean',
        '平均輝度 (μ)': 'mean',
        '輝度標準偏差 (σ)': 'mean',
        'BG比輝度変化 (ΔMean)': 'mean',
        '3σ超過数 [個]': 'mean',
        '3σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean',
        '5σ超過数 [個]': 'mean',
        '5σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean'
    }).reset_index().sort_values(by='濃度(M)')
    
    # Save Excel output
    excel_out = os.path.join(input_dir, "intensity_summary_3s_5s.xlsx")
    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='全データ規格化サマリー', index=False)
        df_series_clean.to_excel(writer, sheet_name='科学的除外クリーンサマリー', index=False)
        df_audit.to_excel(writer, sheet_name='科学的除外根拠証明シート', index=False)
        
    print(f"  [完了] エクセル保存完了: {excel_out}")
    print("\n--- 科学的除外クリーン濃度別比較結果 ---")
    print(df_series_clean[['サンプル系列', '濃度表記', '評価パターン', 'BG比輝度変化 (ΔMean)', '5σ超過規格化割合 [% = (超過数/N_total)*100]']].to_string(index=False))

    # Plot dose-response curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    patterns = [
        ('Pattern A', 'Pattern A (Matched Pairs)', 'o-', '#1f77b4'),
        ('Pattern B', 'Pattern B (All Grid Projected)', 's--', '#ff7f0e')
    ]
    
    df_valid = df_series_clean[df_series_clean['濃度(M)'].notnull()].copy()
    
    for p_label, p_legend, fmt, color in patterns:
        sub = df_valid[df_valid['評価パターン'].str.contains(p_label)]
        if len(sub) == 0:
            continue
            
        x_vals = [x if x > 0 else 1e-16 for x in sub['濃度(M)']]
        y_5s = sub['5σ超過規格化割合 [% = (超過数/N_total)*100]'].values
        y_delta = sub['BG比輝度変化 (ΔMean)'].values
        
        ax1.plot(x_vals, y_5s, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)
        ax2.plot(x_vals, y_delta, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)
        
    ticks = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    tick_labels = ['0 M\n(Blank)', '1 fM', '10 fM', '100 fM', '1 pM', '10 pM', '100 pM', '1 nM']
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('5-Sigma Exceeding Pillar Ratio [%]', fontsize=11, fontweight='bold')
    ax1.set_title('2060824_p50_SHC6OH: 5-Sigma Ratio [%] Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels, rotation=35, fontsize=9)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Intensity Change (Delta Mean)', fontsize=11, fontweight='bold')
    ax2.set_title('2060824_p50_SHC6OH: Delta Mean Intensity Curve', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels, rotation=35, fontsize=9)
    
    plt.tight_layout()
    
    out_png = os.path.join(input_dir, "2060824_p50_SHC6OH_dose_response_curve.png")
    scratch_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer", "2060824_p50_SHC6OH_dose_response_curve.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", "2060824_p50_SHC6OH_dose_response_curve.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(scratch_out, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    print(f"  [完了] グラフ出力完了: {out_png}")

if __name__ == "__main__":
    process_shc6oh_full()
