import numpy as np
import cv2
import pandas as pd
import time
import os

def generate_synthetic_pillars(width=10000, height=10000, spacing=10, jitter_max=2.0, noise_std=5.0):
    print(f"Generating synthetic image ({width}x{height}) with spacing={spacing}...")
    start_time = time.time()
    
    # Initialize background
    # Add a low-frequency gradient (background variation)
    Y_grid, X_grid = np.mgrid[0:height, 0:width]
    background = 10 + 15 * np.sin(X_grid / (width / 2.0)) * np.cos(Y_grid / (height / 2.0))
    img = background.astype(np.float32)
    
    # Generate pillar centroids
    x_coords = np.arange(spacing // 2, width - spacing // 2, spacing)
    y_coords = np.arange(spacing // 2, height - spacing // 2, spacing)
    
    X_centers, Y_centers = np.meshgrid(x_coords, y_coords)
    X = X_centers.flatten().astype(np.float32)
    Y = Y_centers.flatten().astype(np.float32)
    
    # Add random jitter
    np.random.seed(42)  # For reproducibility
    X += np.random.uniform(-jitter_max, jitter_max, size=X.shape)
    Y += np.random.uniform(-jitter_max, jitter_max, size=Y.shape)
    
    # Clip coordinates to be safely inside the image boundaries (with 2-pixel margin)
    X = np.clip(X, 2, width - 3)
    Y = np.clip(Y, 2, height - 3)
    
    # Generate peak intensities
    intensities = np.random.uniform(80.0, 240.0, size=X.shape)
    
    # Convert coordinates to integers for fast NumPy drawing
    xi = np.round(X).astype(np.int32)
    yi = np.round(Y).astype(np.int32)
    
    # Draw pillars onto the image using vectorized operations
    # Simple Gaussian-like shape over 3x3 pixels
    img[yi, xi] += intensities
    img[yi - 1, xi] += intensities * 0.6
    img[yi + 1, xi] += intensities * 0.6
    img[yi, xi - 1] += intensities * 0.6
    img[yi, xi + 1] += intensities * 0.6
    img[yi - 1, xi - 1] += intensities * 0.3
    img[yi - 1, xi + 1] += intensities * 0.3
    img[yi + 1, xi - 1] += intensities * 0.3
    img[yi + 1, xi + 1] += intensities * 0.3
    
    # Clip the image to valid 0-255 range
    img = np.clip(img, 0, 255)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Create DataFrame for ground truth
    # Compute the theoretical mean intensity of the 3x3 footprint
    # Sum of footprint coefficients = 1 + 4*0.6 + 4*0.3 = 1 + 2.4 + 1.2 = 4.6
    # So the mean value on the 9 pixels is: background + (intensity * 4.6) / 9
    bg_vals = background[yi, xi]
    mean_intensities = bg_vals + (intensities * 4.6) / 9.0
    
    df = pd.DataFrame({
        'pillar_id': np.arange(len(X)),
        'x': X,
        'y': Y,
        'true_intensity': mean_intensities
    })
    
    print(f"Generated {len(df):,} pillars in {time.time() - start_time:.2f} seconds.")
    return img, df

if __name__ == "__main__":
    os.makedirs("C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer", exist_ok=True)
    
    # We will generate a 10,000x10,000 image for full benchmarking
    # If the user wants a quick test, they can run a smaller size
    img, df = generate_synthetic_pillars(width=10000, height=10000, spacing=10)
    
    image_path = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/synthetic_pillars.png"
    csv_path = "C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/ground_truth.csv"
    
    print("Saving image to disk (this might take a few seconds due to PNG compression)...")
    t0 = time.time()
    cv2.imwrite(image_path, img)
    print(f"Saved image to {image_path} in {time.time() - t0:.2f} seconds.")
    
    print("Saving ground truth data...")
    df.to_csv(csv_path, index=False)
    print(f"Saved ground truth to {csv_path}")
