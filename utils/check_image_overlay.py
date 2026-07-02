import os
import cv2
import numpy as np

def load_image(path):
    # Support Japanese path and spaces
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def generate_overlay():
    folder = "G:/マイドライブ/1.実験データ_gdrive/5.生データ D/260630 sam dna/位置合わせ"
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
    
    ref_img_path = os.path.join(folder, "1-0.tif")
    tgt_img_path = os.path.join(folder, "1-1.tif")
    
    if not os.path.exists(ref_img_path) or not os.path.exists(tgt_img_path):
        print(f"Original images not found at: {folder}")
        return
        
    ref_img = load_image(ref_img_path)
    tgt_img = load_image(tgt_img_path)
    
    # Precise H_final estimated by the latest polynomial ICP pipeline
    H_final = np.array([
        [0.999620, 0.000291, 80.7818],
        [-0.000690, 0.999454, 51.3862]
    ])
    
    h, w = ref_img.shape
    
    # Warp target image using the estimated transform
    tgt_warped = cv2.warpAffine(tgt_img, H_final, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    
    # Create RGB composite image (Ref = Green, Warped Target = Magenta)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = ref_img # Green channel
    overlay[:, :, 0] = tgt_warped # Blue channel
    overlay[:, :, 2] = tgt_warped # Red channel
    
    # Crop three regions: Left, Center, Right (400x400 px)
    crops = {
        "left": (overlay[800:1200, 200:600], "Left Crop"),
        "center": (overlay[800:1200, 800:1200], "Center Crop"),
        "right": (overlay[800:1200, 1400:1800], "Right Crop")
    }
    
    # Combine crops side-by-side into a single image
    combined = np.hstack([crop for crop, _ in crops.values()])
    
    # Save combined crop
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "image_overlay_check.png")
    cv2.imwrite(out_path, combined)
    print(f"Saved image overlay check to: {out_path}")

if __name__ == "__main__":
    generate_overlay()
