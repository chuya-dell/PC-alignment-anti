import pandas as pd
import cv2
import argparse
import os

def create_detection_overlay(image_path, csv_path, output_path, crop=False, crop_size=500, radius=2, thickness=-1):
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        # Try reading in grayscale and converting to BGR
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Drawing {len(df):,} detection markers...")
    for _, row in df.iterrows():
        x = int(round(row['x']))
        y = int(round(row['y']))
        # Draw a small circle (red, custom radius and thickness)
        cv2.circle(img, (x, y), radius, (0, 0, 255), thickness)
        
    if crop:
        # Save a crop of the center to make visual inspection easy
        h, w, _ = img.shape
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - crop_size // 2)
        y2 = min(h, cy + crop_size // 2)
        x1 = max(0, cx - crop_size // 2)
        x2 = min(w, cx + crop_size // 2)
        
        cropped_img = img[y1:y2, x1:x2]
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved cropped overlay to {output_path}")
    else:
        # Save the full image
        cv2.imwrite(output_path, img)
        print(f"Saved full overlay to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize detection results on the original image")
    parser.add_argument("--image", type=str, required=True, help="Path to the original image")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to the results CSV")
    parser.add_argument("--output", type=str, default="overlay_crop.png", help="Path to save the output image")
    parser.add_argument("--crop", action="store_true", default=True, help="Whether to crop a region for visualization")
    parser.add_argument("--crop-size", type=int, default=800, help="Size of the crop")
    parser.add_argument("--radius", type=int, default=2, help="Radius of the overlay circle")
    parser.add_argument("--thickness", type=int, default=-1, help="Thickness of the circle (-1 for filled dot)")
    
    args = parser.parse_args()
    
    # Run visualization
    create_detection_overlay(
        image_path=args.image,
        csv_path=args.csv,
        output_path=args.output,
        crop=args.crop,
        crop_size=args.crop_size,
        radius=args.radius,
        thickness=args.thickness
    )
