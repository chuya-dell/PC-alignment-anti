import os
import re
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def load_image_unicode(path):
    """Unicode/Japanese path safe image read using numpy and cv2."""
    try:
        n = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None

def main():
    # パス設定
    # オリジナル画像が格納されているディレクトリ
    img_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/df"
    # アライメント済みピラーCSVが格納されているディレクトリ
    pillar_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti"
    # グリッド解析結果の出力ディレクトリ
    output_dir = os.path.join(pillar_dir, "grid_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    grid_size = 6.29 # グリッド1辺のピクセルサイズ (ImageJマクロのデフォルト)
    
    print("Starting Grid Image Difference Analysis...")
    print(f"Image Directory:  {img_dir}")
    print(f"Pillar Directory: {pillar_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Grid Size:        {grid_size} px\n")
    
    # アライメントデータファイルをスキャン
    files = os.listdir(pillar_dir)
    aligned_files = [f for f in files if f.endswith('_aligned_to_pre.csv')]
    
    # 自然順ソート (1-1, 1-2, ..., 8-8)
    def extract_key(filename):
        base = filename[:-19]
        parts = base.split('-')
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return (999, 999)
            
    aligned_files.sort(key=extract_key)
    print(f"Found {len(aligned_files)} aligned pairs to process.")
    
    # 画像ファイル名定義のためのヘルパー
    # 例: "1-1-0" または "1-1-1" を含むファイルを検索
    all_img_files = os.listdir(img_dir)
    
    for f_aligned in tqdm(aligned_files, desc="Grid Analysis Batch"):
        base = f_aligned[:-19] # '1-1'
        
        # このペアに対応するオリジナル画像(pre/post)を特定
        pre_img_name = [img for img in all_img_files if img.startswith(f"{base}-0") and img.lower().endswith('.tif')]
        post_img_name = [img for img in all_img_files if img.startswith(f"{base}-1") and img.lower().endswith('.tif')]
        
        if not pre_img_name or not post_img_name:
            # カッコ付きコピー名などでのフォールバック検索
            pre_img_name = [img for img in all_img_files if f"{base}-0" in img and img.lower().endswith('.tif')]
            post_img_name = [img for img in all_img_files if f"{base}-1" in img and img.lower().endswith('.tif')]
            
        if not pre_img_name or not post_img_name:
            print(f"Warning: Images not found for pair {base}. Skipped.")
            continue
            
        pre_img_path = os.path.join(img_dir, pre_img_name[0])
        post_img_path = os.path.join(img_dir, post_img_name[0])
        
        # ピラー座標データをロード
        aligned_path = os.path.join(pillar_dir, f_aligned)
        post_path = os.path.join(pillar_dir, f"{base}_pillars_post.csv")
        pre_path = os.path.join(pillar_dir, f"{base}_pillars_pre.csv")
        
        if not os.path.exists(post_path) or not os.path.exists(pre_path):
            print(f"Warning: Original pillar CSVs not found for {base}. Skipped.")
            continue
            
        df_aligned = pd.read_csv(aligned_path)
        df_post_orig = pd.read_csv(post_path)
        df_pre = pd.read_csv(pre_path)
        
        # 1. 一致したピラーペアを抽出して、アライメント前とプレ側の対応座標を得る
        matched = df_aligned[df_aligned['matched_ref_id'] != -1].copy()
        if len(matched) < 10:
            print(f"Warning: Too few matched pillars ({len(matched)}) for {base}. Skipped.")
            continue
            
        # ポスト側の元のIDを取得して、元の座標 (x_orig, y_orig) とマージする
        # アライメントファイルにはアライメント後の (x, y) が入っているため、元のポスト座標 (df_post_orig) からロードする
        matched_coords = pd.merge(
            matched[['pillar_id', 'matched_ref_id']].rename(columns={'pillar_id': 'aligned_id'}),
            df_post_orig[['pillar_id', 'x', 'y']].rename(columns={'x': 'x_post', 'y': 'y_post'}),
            left_on='aligned_id',
            right_on='pillar_id'
        )
        
        # プレ側の対応座標 (x_pre, y_pre) とマージ
        pair_coords = pd.merge(
            matched_coords,
            df_pre[['pillar_id', 'x', 'y']].rename(columns={'x': 'x_pre', 'y': 'y_pre'}),
            left_on='matched_ref_id',
            right_on='pillar_id'
        )
        
        # 対応点座標の抽出 (RANSAC推定用)
        src_pts = pair_coords[['x_post', 'y_post']].values.astype(np.float32)
        dst_pts = pair_coords[['x_pre', 'y_pre']].values.astype(np.float32)
        
        # 2. 画像全体の射影アライメント行列 (Homography H) を頑健に推定
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if H is None:
            print(f"Warning: Homography estimation failed for {base}. Skipped.")
            continue
            
        # 3. 画像のロード
        pre_img = load_image_unicode(pre_img_path)
        post_img = load_image_unicode(post_img_path)
        
        if pre_img is None or post_img is None:
            continue
            
        h, w = pre_img.shape
        
        # 4. ポスト画像をプレ画像に合わせてワープ (サブピクセル精度アライメント)
        post_aligned = cv2.warpPerspective(post_img, H, (w, h), flags=cv2.INTER_LINEAR)
        
        # 5. 共通領域マスクの計算 (食み出し領域を除外)
        # ポスト画像の枠マスクをワープさせて有効範囲を作る
        post_mask = cv2.warpPerspective(np.ones_like(post_img, dtype=np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
        common_mask = (post_mask > 0) & (pre_img > 0)
        
        # 5.5 傷およびゴミ領域の自動検出と除外マスクの計算
        # 直径15ピクセルの楕円カーネルによるモルフォロジー・オープニング処理
        # これによりピラー(小さな明るい点)が消え、大きな傷やランドマーク溝などのみが高輝度構造として残る
        kernel_scratch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # pre画像の傷
        pre_open = cv2.morphologyEx(pre_img, cv2.MORPH_OPEN, kernel_scratch)
        thresh_pre = np.mean(pre_open) + 2.0 * np.std(pre_open)
        _, pre_scratch_mask = cv2.threshold(pre_open, thresh_pre, 255, cv2.THRESH_BINARY)
        
        # post画像 (アライメント後) の傷
        post_open = cv2.morphologyEx(post_aligned, cv2.MORPH_OPEN, kernel_scratch)
        thresh_post = np.mean(post_open) + 2.0 * np.std(post_open)
        _, post_scratch_mask = cv2.threshold(post_open, thresh_post, 255, cv2.THRESH_BINARY)
        
        # 傷マスク (pre または post のいずれかで傷・太い溝がある領域)
        scratch_mask = (pre_scratch_mask > 0) | (post_scratch_mask > 0)
        
        # 共通領域かつ、傷がない領域の最終有効マスク
        valid_mask = common_mask & (~scratch_mask)
        
        # 6. 差分画像の計算 (post - pre)
        diff_img = post_aligned.astype(np.float32) - pre_img.astype(np.float32)
        
        # 7. グリッド分割解析 (G = 6.29)
        cols = int(np.floor(w / grid_size))
        rows = int(np.floor(h / grid_size))
        
        grid_records = []
        heatmap_data = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            for c in range(cols):
                x_start = int(np.floor(c * grid_size))
                x_end = int(np.floor((c + 1) * grid_size))
                y_start = int(np.floor(r * grid_size))
                y_end = int(np.floor((r + 1) * grid_size))
                
                # ブロック内の輝度差とマスク (傷なし共通領域のみ)
                block_diff = diff_img[y_start:y_end, x_start:x_end]
                block_mask = valid_mask[y_start:y_end, x_start:x_end]
                
                valid_pixels = block_diff[block_mask]
                
                if len(valid_pixels) > 0:
                    mean_diff = float(np.mean(valid_pixels))
                    std_diff = float(np.std(valid_pixels))
                    pixel_count = int(len(valid_pixels))
                else:
                    mean_diff = np.nan
                    std_diff = np.nan
                    pixel_count = 0
                    
                grid_records.append({
                    'Col': c + 1,
                    'Row': r + 1,
                    'Mean': mean_diff,
                    'Std': std_diff,
                    'PixelCount': pixel_count
                })
                
                # ヒートマップ用の配列 (NaNは0にしてマッピング)
                heatmap_data[r, c] = mean_diff if not np.isnan(mean_diff) else 0.0
                
        # 8. CSV出力の作成と保存
        df_grid = pd.DataFrame(grid_records)
        csv_out_path = os.path.join(output_dir, f"{base}_grid_analysis.csv")
        df_grid.to_csv(csv_out_path, index=False)
        
        # 9. 可視化ヒートマップ画像 (ImageJのFireカラールックアップテーブルに相当) の生成
        # -10 から +10 の範囲を 0〜255 にスケーリング
        v_min, v_max = -10.0, 10.0
        clipped = np.clip(heatmap_data, v_min, v_max)
        scaled = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
        
        # COLORMAP_JET または COLORMAP_MAGMA (ImageJのFireに類似) を適用
        heatmap_color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        
        # NaNの部分（共通領域外）を黒 (0, 0, 0) でマスクする
        # グリッド解像度でのマスクを作成
        grid_mask = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if grid_records[idx]['PixelCount'] > 0:
                    grid_mask[r, c] = 255
                    
        # マスク適用
        heatmap_color[grid_mask == 0] = [0, 0, 0]
        
        # ImageJマクロと同様にピクセル補間なしで高解像度化 (2048幅付近)
        display_scale = int(np.floor(2048 / cols))
        if display_scale > 1:
            h_out = rows * display_scale
            w_out = cols * display_scale
            heatmap_resized = cv2.resize(heatmap_color, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        else:
            heatmap_resized = heatmap_color
            
        img_out_path = os.path.join(output_dir, f"{base}_grid_heatmap.png")
        # 日本語パス安全な保存
        _, ext = os.path.splitext(img_out_path)
        is_success, buffer = cv2.imencode(ext, heatmap_resized)
        if is_success:
            buffer.tofile(img_out_path)

    print(f"\nGrid analysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
