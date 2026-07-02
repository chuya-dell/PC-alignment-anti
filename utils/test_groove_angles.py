import os
import cv2
import numpy as np

def load_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def find_groove_pos(img_strip, search_range=(50, 400), axis='x'):
    # Profile along the search axis
    if axis == 'x':
        profile = np.mean(img_strip, axis=0) # average vertically
    else:
        profile = np.mean(img_strip, axis=1) # average horizontally
        
    # Smooth profile
    smoothed = np.convolve(profile, np.ones(15)/15, mode='same')
    
    # Restrict to search range
    search_profile = smoothed[search_range[0]:search_range[1]]
    min_idx_local = np.argmin(search_profile)
    min_idx = search_range[0] + min_idx_local
    
    # Subpixel resolution using quadratic fit
    if 0 < min_idx < len(profile) - 1:
        y0, y1, y2 = profile[min_idx-1], profile[min_idx], profile[min_idx+1]
        denom = (y0 - 2*y1 + y2)
        if denom != 0:
            offset = 0.5 * (y0 - y2) / denom
            return min_idx + offset
    return float(min_idx)

def get_image_groove_angles(img_path):
    img = load_image(img_path)
    h, w = img.shape
    
    # 1. Vertical Groove (X position) at Y_upper (200 to 700) and Y_lower (1300 to 1800)
    strip_upper = img[200:700, :]
    strip_lower = img[1300:1800, :]
    
    x_upper = find_groove_pos(strip_upper, search_range=(50, 300), axis='x')
    x_lower = find_groove_pos(strip_lower, search_range=(50, 300), axis='x')
    
    # Vertical distance between center of strips is 1100 px (from 450 to 1550)
    dy_vert = 1100.0
    angle_vert = np.degrees(np.arctan2(x_lower - x_upper, dy_vert))
    
    # 2. Horizontal Groove (Y position) at X_left (200 to 700) and X_right (1300 to 1800)
    # The horizontal groove is near Y = 100 to 200, but only on the right-top side of the image
    # Note: Horizontal groove is only in the right half of the image!
    # Let's check X_mid (1000 to 1300) and X_right (1500 to 1800)
    strip_left = img[:, 1000:1300]
    strip_right = img[:, 1500:1800]
    
    y_left = find_groove_pos(strip_left, search_range=(50, 250), axis='y')
    y_right = find_groove_pos(strip_right, search_range=(50, 250), axis='y')
    
    dx_horiz = 500.0
    angle_horiz = np.degrees(np.arctan2(y_right - y_left, dx_horiz))
    
    return {
        "x_upper": x_upper,
        "x_lower": x_lower,
        "angle_vert": angle_vert,
        "y_left": y_left,
        "y_right": y_right,
        "angle_horiz": angle_horiz
    }

def main():
    folder = "G:/マイドライブ/1.実験データ_gdrive/5.生データ D/260630 sam dna/位置合わせ"
    
    images = ["1-0.tif", "1-1.tif", "1-2.tif", "1-3.tif"]
    
    for img_name in images:
        path = os.path.join(folder, img_name)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        res = get_image_groove_angles(path)
        print(f"\nImage: {img_name}")
        print(f"  Vertical Groove: upper_X={res['x_upper']:.2f}, lower_X={res['x_lower']:.2f} | Angle={res['angle_vert']:.4f}°")
        print(f"  Horizontal Groove: left_Y={res['y_left']:.2f}, right_Y={res['y_right']:.2f} | Angle={res['angle_horiz']:.4f}°")

if __name__ == "__main__":
    main()
