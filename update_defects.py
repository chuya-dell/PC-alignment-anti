import pandas as pd
import os

data = [
    # SAM 8/24
    (260824, 0, 1, 'Stain', 'suspected'), # from previous context
    (260824, 0, 4, 'StainDust', 'clear'),
    (260824, 0, 5, 'Dust', 'clear'),
    (260824, 0, 6, 'Dust', 'clear'),
    (260824, 0, 8, 'Dust_Huge', 'clear'),
    (260824, 3, 4, 'Dust', 'clear'),
    (260824, 3, 6, 'Stain_Large', 'clear'),
    (260824, 3, 7, 'Stain', 'clear'),
    (260824, 3, 8, 'Stain', 'clear'),
    (260824, 4, 1, 'Stain', 'clear'),
    (260824, 4, 2, 'Dust', 'clear'),
    (260824, 4, 5, 'Dust', 'clear'),
    (260824, 4, 6, 'Stain', 'clear'),
    (260824, 8, 1, 'StainDust', 'clear'),
    (260824, 8, 5, 'Stain', 'clear'),
    (260824, 8, 7, 'Dust', 'clear'),
    (260824, 8, 8, 'StainDust', 'clear'),
    (260824, 10, 2, 'StainDust', 'clear'),
    (260824, 10, 4, 'Dust', 'clear'),
    (260824, 10, 6, 'Stain', 'clear'),
    (260824, 10, 8, 'StainDust', 'clear'),
    (260824, 11, 1, 'Dust', 'clear'),
    (260824, 11, 2, 'Dust', 'clear'),
    (260824, 11, 3, 'Stain', 'clear'),
    (260824, 11, 4, 'Scratch', 'clear'),
    (260824, 12, 8, 'Stain', 'clear'),
    (260824, 13, 4, 'Dust', 'clear'),
    (260824, 14, 1, 'Stain', 'clear'),
    (260824, 14, 6, 'Stain', 'clear'),
    
    # DNA 8/25
    (260825, 0, 7, 'Stain', 'clear'),
    (260825, 4, 1, 'Dust', 'clear'),
    (260825, 4, 2, 'Stain_Huge', 'clear'),
    (260825, 4, 4, 'Stain_Large', 'clear'),
    (260825, 4, 7, 'Dust', 'clear'),
    (260825, 5, 1, 'Scratch_Vert', 'clear'),
    (260825, 5, 4, 'Stain_BL', 'clear'),
    (260825, 5, 8, 'Scratch', 'clear'),
    (260825, 6, 3, 'Dust', 'clear'),
    (260825, 6, 8, 'Stain_Soak', 'clear'),
    (260825, 7, 8, 'Stain_Huge_Exclude', 'clear'),
    (260825, 8, 3, 'Dust', 'clear'),
    (260825, 8, 6, 'Dust_Scatter', 'clear'),
    (260825, 8, 7, 'Dust_Scatter', 'clear'),
    (260825, 9, 1, 'Dust', 'clear'),
    (260825, 9, 6, 'Dust_Large', 'clear'),
    (260825, 9, 8, 'Dust', 'clear'),
]

rows = []
for d, s, p, t, c in data:
    radius = 40
    if 'dust' in t.lower() and 'stain' not in t.lower():
        radius = 16
        
    rows.append({
        'Date': d, 'SampleID': s, 'PositionID': p,
        'FieldFileName': f"{s}-{p}-0.tif",
        'DefectType': t, 'Shape': 'circle',
        'X_px': 0, 'Y_px': 0, 'W_px': '', 'H_px': '',
        'Radius_px': radius,
        'IsPre': 'TRUE', 'IsPost': 'TRUE',
        'Confidence': c,
        'AnnotatedBeforeAnalysis': 'TRUE'
    })

df = pd.DataFrame(rows)

out_path = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\defect_coordinates.csv"
header_comment = "#[CoordinateSystem: ImageJ (Top-Left Origin, Y-down), Raw Pre-Image Space]\n"

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(header_comment)
    df.to_csv(f, index=False, lineterminator='\n')

print("Updated defect_coordinates.csv")
