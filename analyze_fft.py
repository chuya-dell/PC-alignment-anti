import cv2
import numpy as np
import os
import argparse

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """Windows環境で日本語文字を含むパスから画像を安全に読み込む。"""
    try:
        n = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"Error reading file with numpy/cv2: {path}, error: {e}")
        return None

def imwrite_unicode(path, img):
    """Windows環境で日本語文字を含むパスへ画像を安全に書き込む。"""
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            n.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"Error writing file with numpy/cv2: {path}, error: {e}")
        return False

def analyze_fft(image_path, output_dir=None):
    # 画像の読み込み
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
        
    h, w = img.shape
    print(f"Image: {os.path.basename(image_path)} ({w}x{h})")
    
    # 2D FFT用に中央の 1024x1024 領域をクロップ (2のべき乗サイズが高速かつ対称性が良い)
    size = 1024
    cy, cx = h // 2, w // 2
    img_crop = img[cy - size//2:cy + size//2, cx - size//2:cx + size//2]
    
    # ハミング窓を適用して端部のリーケージ（線状のノイズ）を抑制
    window = np.outer(np.hamming(size), np.hamming(size))
    img_windowed = (img_crop.astype(float) - np.mean(img_crop)) * window
    
    # 2D FFTの実行
    f_transform = np.fft.fft2(img_windowed)
    f_shift = np.fft.fftshift(f_transform)
    
    # パワースペクトル (対数スケール)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-5)
    
    # 直流成分 (中心)
    cy_f, cx_f = size // 2, size // 2
    
    # HCP格子のピッチが約5〜8pxと仮定すると、周波数Rは 1024 / ピッチ = 128〜204 ピクセル付近になる。
    # 探索範囲：中心から距離 [80, 250] の回折環の範囲
    y_indices, x_indices = np.indices((size, size))
    dist_from_center = np.sqrt((x_indices - cx_f)**2 + (y_indices - cy_f)**2)
    
    mask = (dist_from_center >= 80) & (dist_from_center <= 250)
    spectrum_masked = magnitude_spectrum.copy()
    spectrum_masked[~mask] = 0
    
    # 最大ピークの座標を特定 (第一回折スポットの1つ)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(spectrum_masked)
    
    # 中心からの距離 R
    R = np.sqrt((max_loc[0] - cx_f)**2 + (max_loc[1] - cy_f)**2)
    
    # 実空間における周期 (ピッチ) の計算
    pitch = size / R
    
    # 最大ピークの角度 (基準角)
    base_angle = np.arctan2(max_loc[1] - cy_f, max_loc[0] - cx_f)
    base_angle_deg = np.degrees(base_angle)
    
    # 六方格子の規則性検証: 60度間隔の6つのスポットの輝度をチェック
    angles = base_angle + np.radians([0, 60, 120, 180, 240, 300])
    spot_intensities = []
    spot_coords = []
    
    for a in angles:
        px = int(round(cx_f + R * np.cos(a)))
        py = int(round(cy_f + R * np.sin(a)))
        spot_coords.append((px, py))
        
        # スポット周囲 3x3 の平均強度
        val = np.mean(magnitude_spectrum[py-1:py+2, px-1:px+2])
        spot_intensities.append(val)
        
    # 背景領域の平均輝度（スポットがない場所）を測定してコントラスト比を出す
    bg_mask = mask.copy()
    for px, py in spot_coords:
        cv2.circle(bg_mask.astype(np.uint8), (px, py), 20, 0, -1)
    bg_mean = np.mean(magnitude_spectrum[bg_mask == 1])
    
    print("\n--- FFT Analysis Result ---")
    print(f"  Detected回折環半径 R : {R:.2f} pixels")
    print(f"  計算された格子周期 (ピッチ): {pitch:.3f} pixels")
    print(f"  ベース配向角 (theta)     : {base_angle_deg:.2f}°")
    print(f"  背景ノイズ平均強度       : {bg_mean:.2f}")
    print("  6つの回折スポット強度:")
    for i, a in enumerate(np.degrees(angles)):
        a_fold = (a + 180) % 360 - 180
        contrast = spot_intensities[i] - bg_mean
        print(f"    角度 {a_fold:+.1f}° : 強度 {spot_intensities[i]:.2f} (背景比: {contrast:+.2f})")
        
    # 六方格子の判定
    avg_spot_intensity = np.mean(spot_intensities)
    contrast_ratio = avg_spot_intensity - bg_mean
    is_periodic = contrast_ratio > 15.0 # 差分が15dB以上あれば明確な周期構造と判定
    
    print(f"  平均スポット強度         : {avg_spot_intensity:.2f}")
    print(f"  スポット/背景コントラスト: {contrast_ratio:.2f} dB")
    print(f"  -> 判定結果: {'明確な六方格子周期構造あり' if is_periodic else '周期構造は不明瞭'}")
    
    # 2. 可視化画像の作成・保存
    # パワースペクトル画像を 0-255 に正規化
    mag_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_color = cv2.cvtColor(mag_normalized, cv2.COLOR_GRAY2BGR)
    
    # 回折環を描画 (赤色)
    cv2.circle(mag_color, (cx_f, cy_f), int(round(R)), (0, 0, 255), 1)
    
    # 6つの回折スポットを描画 (緑色サークル)
    for i, (px, py) in enumerate(spot_coords):
        cv2.circle(mag_color, (px, py), 8, (0, 255, 0), 2)
        cv2.putText(mag_color, f"{i+1}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # 保存先の設定
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_fft_spectrum.png")
    
    imwrite_unicode(out_path, mag_color)
    print(f"  FFTスペクトル画像を保存しました: {out_path}")
    
    # 画像全体にピッチpitchのHCP格子を敷き詰めた場合の理論ピラー数
    # 六方格子の単位胞面積 S_unit = pitch^2 * sin(60deg)
    # 理論最大ピラー数 N_max = 画像面積 / S_unit
    area_img = w * h
    area_unit = (pitch**2) * np.sin(np.radians(60))
    n_max_theoretical = area_img / area_unit
    print(f"  画像全体（{w}x{h}）に格子を敷き詰めた場合の理論最大ピラー数: {n_max_theoretical:,.0f} 個")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image FFT Periodicity Analyzer")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--out-dir", default=None, help="Directory to save the FFT spectrum image")
    
    args = parser.parse_args()
    
    # 日本語マウントパスの解決
    from run_batch_alignment import resolve_gdrive_path
    image_path = resolve_gdrive_path(args.image)
    out_dir = resolve_gdrive_path(args.out_dir)
    
    analyze_fft(image_path, out_dir)
