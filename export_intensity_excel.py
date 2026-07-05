import pandas as pd
import numpy as np
import os
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
import time

def main():
    data_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti"
    output_xlsx = "./pillar_intensity_analysis_120k.xlsx"
    
    print("Starting Optimized Excel export pipeline (1-120k index in col A)...")
    print(f"Source Folder: {data_dir}")
    print(f"Target Excel:  {output_xlsx}\n")
    
    # 8つの条件 (1〜8) ごとにデータを保持
    cond_sheets = {str(c): {} for c in range(1, 9)}
    
    # 1. foranti フォルダ内のファイルをスキャンして処理
    files = os.listdir(data_dir)
    aligned_files = [f for f in files if f.endswith('_aligned_to_pre.csv')]
    
    # 自然順にソート (1-1, 1-2, ..., 8-8)
    def extract_key(filename):
        base = filename[:-19]
        parts = base.split('-')
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return (999, 999)
            
    aligned_files.sort(key=extract_key)
    print(f"Found {len(aligned_files)} aligned CSV files to process.")
    
    for f in aligned_files:
        base = f[:-19] # '1-2'
        parts = base.split('-')
        if len(parts) < 2:
            continue
        cond, set_num = parts[0], parts[1]
        
        if cond not in cond_sheets:
            continue
            
        aligned_path = os.path.join(data_dir, f)
        pre_path = os.path.join(data_dir, f"{base}_pillars_pre.csv")
        
        if not os.path.exists(pre_path):
            continue
            
        # データロード
        df_aligned = pd.read_csv(aligned_path)
        df_pre = pd.read_csv(pre_path)
        
        # 共通部分の抽出 (matched_ref_id != -1)
        matched_df = df_aligned[df_aligned['matched_ref_id'] != -1].copy()
        excluded_count = len(df_aligned) - len(matched_df)
        
        # post側の 'pillar_id' をドロップしてマージ時の名前重複を防ぐ
        if 'pillar_id' in matched_df.columns:
            matched_df = matched_df.drop(columns=['pillar_id'])
            
        # preデータとマージして輝度変化を算出
        df_pre_sub = df_pre[['pillar_id', 'box_intensity_3x3']].rename(
            columns={'box_intensity_3x3': 'box_intensity_3x3_pre'}
        )
        
        merged = pd.merge(
            matched_df,
            df_pre_sub,
            left_on='matched_ref_id',
            right_on='pillar_id'
        )
        
        # 輝度変化量 (post - pre) の算出
        merged['intensity_change'] = merged['box_intensity_3x3'] - merged['box_intensity_3x3_pre']
        
        # 出力データの整理 (基準側IDと輝度変化量)
        result_df = merged[['pillar_id', 'intensity_change']].copy()
        result_df = result_df.sort_values('pillar_id').reset_index(drop=True)
        
        cond_sheets[cond][set_num] = {
            'total_count': len(result_df),
            'excluded_count': excluded_count,
            'data': result_df
        }
        
    # 2. Pandas を使って 120,000行の基礎 DataFrame を一気に Excel へ高速書き出し (リトライ機能付き)
    print("\nWriting main data columns (A-I) to Excel sheets via Pandas...")
    
    writer = None
    for attempt in range(1, 21):
        try:
            writer = pd.ExcelWriter(output_xlsx, engine='openpyxl')
            break
        except PermissionError:
            print(f"Warning: Excel file is locked (Permission denied). Attempt {attempt}/20. Please close the Excel file if it is open...")
            time.sleep(3)
            
    if writer is None:
        raise PermissionError(f"Could not open target Excel file because it is locked by another process: {output_xlsx}")
        
    with writer:
        for cond_num in range(1, 9):
            cond_str = str(cond_num)
            sheet_name = f"Condition {cond_str}"
            
            # A列に 1〜120,000 の連番をあらかじめ生成
            df_cond = pd.DataFrame({'': np.arange(1, 120001)}) # ヘッダーも空白にして連番だけにする
            
            sets_data = cond_sheets[cond_str]
            for s_idx in range(1, 9):
                set_str = str(s_idx)
                col_name = f"Set {set_str}"
                
                if set_str in sets_data:
                    # 輝度変化データ
                    values = sets_data[set_str]['data']['intensity_change'].values
                    # 120,000行に引き伸ばして足りない部分を NaN (空白) で埋める
                    padded_values = np.full(120000, np.nan)
                    padded_values[:len(values)] = values
                    df_cond[col_name] = padded_values
                else:
                    df_cond[col_name] = np.full(120000, np.nan)
                    
            # インデックス無し、ヘッダーもそのまま書き出し
            df_cond.to_excel(writer, sheet_name=sheet_name, index=False)
            
    # 3. openpyxl を使って、K列・L列の統計情報テーブルの追加と書式スタイリングを行う
    print("Applying styling and adding Set statistics to K-L columns...")
    
    wb = None
    for attempt in range(1, 21):
        try:
            wb = openpyxl.load_workbook(output_xlsx)
            break
        except PermissionError:
            print(f"Warning: Excel file is locked during post-styling open. Attempt {attempt}/20. Retrying...")
            time.sleep(3)
            
    if wb is None:
        raise PermissionError(f"Could not read Excel file because it is locked by Excel: {output_xlsx}")
    
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    normal_font = Font(name="Segoe UI", size=9)
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid") # Steel Blue
    
    for cond_num in range(1, 9):
        cond_str = str(cond_num)
        sheet_name = f"Condition {cond_str}"
        ws = wb[sheet_name]
        
        # グリッドラインを表示する設定 (Excelで枠線を表示させる)
        ws.views.sheetView[0].showGridLines = True
        
        # K列, L列 of headers (セット統計情報の縦書き用)
        cell_k_head = ws.cell(row=1, column=11, value="Set") # K列 (11)
        cell_l_head = ws.cell(row=1, column=12, value="N (Pillars)") # L列 (12)
        for cell in [cell_k_head, cell_l_head]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            
        sets_data = cond_sheets[cond_str]
        
        # A列(連番)の1行目のヘッダーセルを取得してスタイル調整 (空文字列に設定されている)
        cell_a_head = ws.cell(row=1, column=1)
        cell_a_head.value = 1 # 1行目から連番にするため、ヘッダー行だったセルを数値の「1」に上書き
        cell_a_head.font = normal_font
        cell_a_head.alignment = Alignment(horizontal="center")
        
        # B〜I列の1行目のヘッダーセルのフォント/アライメントの調整 (値は Set 1 〜 Set 8)
        # 1行目から輝度データにするため、ヘッダー行だったセルの値を対応するデータの最初の値で上書きする
        for s_idx in range(1, 9):
            set_str = str(s_idx)
            col_idx = s_idx + 1 # B列 (2) 〜 I列 (9)
            cell_data_head = ws.cell(row=1, column=col_idx)
            
            # 元々のヘッダーセル(行1)を、データの最初の値（1つ目の輝度変化量）に書き換える
            if set_str in sets_data and len(sets_data[set_str]['data']) > 0:
                first_val = float(sets_data[set_str]['data']['intensity_change'].iloc[0])
                cell_data_head.value = first_val
                cell_data_head.number_format = '0.00'
            else:
                cell_data_head.value = None # データがない場合は空白
                
            cell_data_head.font = normal_font
            cell_data_head.alignment = Alignment(horizontal="right")
            
            # J列・K列(統計テーブル)の2行目〜9行目にセット情報を縦書きで追加
            row_info = s_idx + 1
            c_k = ws.cell(row=row_info, column=11, value=f"Set {set_str}")
            
            if set_str in sets_data:
                total_count = sets_data[set_str]['total_count']
                c_l = ws.cell(row=row_info, column=12, value=total_count)
                c_l.number_format = '#,##0'
                c_l.alignment = Alignment(horizontal="right")
            else:
                c_l = ws.cell(row=row_info, column=12, value="No Data")
                c_l.alignment = Alignment(horizontal="center")
                
            c_k.font = normal_font
            c_l.font = normal_font
            c_k.alignment = Alignment(horizontal="center")
            
        # 列幅設定
        ws.column_dimensions['A'].width = 10
        for s_idx in range(1, 9):
            col_letter = get_column_letter(s_idx + 1)
            ws.column_dimensions[col_letter].width = 12
            
        ws.column_dimensions['J'].width = 3 # 空き列
        ws.column_dimensions['K'].width = 10
        ws.column_dimensions['L'].width = 15
        
    for attempt in range(1, 21):
        try:
            wb.save(output_xlsx)
            print(f"\nSuccessfully generated Optimized Excel file at:\n{output_xlsx}")
            break
        except PermissionError:
            print(f"Warning: Excel file is locked during save. Attempt {attempt}/20. Please close the Excel file...")
            time.sleep(3)

if __name__ == "__main__":
    main()
