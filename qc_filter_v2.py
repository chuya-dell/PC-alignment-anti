# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import roifile

LEDGER_PATH = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv"
DEFECTS_PATH = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\defect_coordinates.csv"

# Preload data if available
try:
    df_ledger = pd.read_csv(LEDGER_PATH)
except:
    df_ledger = pd.DataFrame()

try:
    df_defects = pd.read_csv(DEFECTS_PATH, comment='#') # Skip header comments like [CoordinateSystem...]
except:
    df_defects = pd.DataFrame()

def get_sample_info(date_str, assay_type, sample_id):
    """
    Returns L1 status and metadata for a sample.
    Returns: (is_valid, concentration, run_id, reason, substrate_id)
    """
    if df_ledger.empty:
        return True, 0.0, 1, "LEDGER_NOT_FOUND", 1
        
    date_val = int(date_str) if str(date_str).isdigit() else date_str
    sub = df_ledger[(df_ledger['Date'] == date_val) & (df_ledger['SampleID'] == sample_id)]
    
    if len(sub) == 0:
        return True, 0.0, 1, "NOT_IN_LEDGER", 1
        
    row = sub.iloc[0]
    
    is_leak = str(row.get('WellLeakSuspect', 'FALSE')).upper() == 'TRUE'
    is_leak_wet = str(row.get('WellStatus', 'normal')).lower() == 'leak_wet'
    conc = float(row.get('Concentration_M', 0.0))
    run_id = row.get('ReplicateOf', 1)
    if pd.isna(run_id) or str(run_id).strip() == '':
        run_id = 1
        
    substrate_id = row.get('SubstrateID', 1)
    if pd.isna(substrate_id):
        substrate_id = 1
    
    if is_leak or is_leak_wet:
        return False, conc, run_id, "L1_WELL_LEAK_SUSPECT", substrate_id
        
    return True, conc, run_id, "VALID", substrate_id

def get_l3_mask(date_str, sample_id, pos_id, xi, yi, img_width=4096, img_height=4096, scratch_mask=None, roi_dir=None):
    """
    Applies L3 masking to an array of (xi, yi) coordinates.
    - scratch_mask: boolean 2D array of same size as image. True means the pixel is in the scratch zone.
    - roi_dir: Directory to look for ImageJ ROI .zip files named e.g. "260824_0_1_ROI.zip"
    
    Returns: (valid_mask, masked_ratio)
    valid_mask: boolean 1D array same length as xi, True means keep, False means exclude.
    masked_ratio: float (0.0 to 1.0) representing fraction of *valid overlap area* that was masked.
    """
    n_pts = len(xi)
    valid_mask = np.ones(n_pts, dtype=bool)
    
    total_area = float(img_width * img_height)
    if scratch_mask is not None:
        valid_overlap_area = float(total_area - np.sum(scratch_mask))
    else:
        valid_overlap_area = total_area
        
    masked_area = 0.0
    
    # 0. Check for Auto Mask (.npy) first
    auto_mask_path = rf"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\auto_masks\{date_str}_{sample_id}-{pos_id}_mask.npy"
    if os.path.exists(auto_mask_path):
        auto_mask = np.load(auto_mask_path)
        if scratch_mask is not None:
            auto_mask[scratch_mask] = 0
            
        masked_area += np.sum(auto_mask)
        in_mask = auto_mask[yi, xi] > 0
        valid_mask &= (~in_mask)
        
        masked_ratio = min(1.0, masked_area / valid_overlap_area) if valid_overlap_area > 0 else 1.0
        return valid_mask, masked_ratio
    
    # 1. Check for ROI zip files first
    if roi_dir:
        roi_filename = f"{date_str}_{sample_id}_{pos_id}_ROI.zip"
        roi_path = os.path.join(roi_dir, roi_filename)
        if os.path.exists(roi_path):
            rois = roifile.ImagejRoi.fromfile(roi_path)
            if not isinstance(rois, list):
                rois = [rois]
                
            for roi in rois:
                # Basic bounding box check for ROIs (assuming polygon or rect)
                # For a true pixel-perfect mask we'd render the ROI to a 2D array, 
                # but for points we can just check if they fall inside.
                # Here we use bounding box as a fast approximation, 
                # or render to mask using cv2.fillPoly if we extract coordinates.
                coords = roi.coordinates()
                if coords is not None and len(coords) > 0:
                    coords = coords.astype(np.int32)
                    # Render polygon to a blank mask to count area and filter points
                    poly_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [coords], 1)
                    
                    # Subtract scratch region from the defect mask to not double-count area
                    if scratch_mask is not None:
                        poly_mask[scratch_mask] = 0
                        
                    masked_area += np.sum(poly_mask)
                    
                    # Filter points
                    in_poly = poly_mask[yi, xi] == 1
                    valid_mask &= (~in_poly)
                    
            masked_ratio = min(1.0, masked_area / valid_overlap_area)
            return valid_mask, masked_ratio
    
    # 2. Check CSV coordinates
    date_val = int(date_str) if str(date_str).isdigit() else date_str
    if not df_defects.empty:
        sub = df_defects[(df_defects['Date'] == date_val) & 
                         (df_defects['SampleID'] == sample_id) & 
                         (df_defects['PositionID'] == pos_id)]
                         
        if len(sub) > 0:
            for _, row in sub.iterrows():
                dx = row.get('X_px', 0)
                dy = row.get('Y_px', 0)
                
                # Default radii
                r = row.get('Radius_px', np.nan)
                if pd.isna(r):
                    r = 16 if str(row.get('DefectType')).upper() == 'DUST' else 40
                else:
                    r = float(r)
                    
                # If unannotated (0,0), escalate to 100% mask
                if dx == 0 and dy == 0:
                    return np.zeros(n_pts, dtype=bool), 1.0
                    
                dist_sq = (xi - dx)**2 + (yi - dy)**2
                in_circle = dist_sq <= r**2
                
                valid_mask &= (~in_circle)
                # Roughly approximate area (ignoring overlaps and scratch for simple circles)
                masked_area += np.pi * (r**2)
                
    masked_ratio = min(1.0, masked_area / valid_overlap_area) if valid_overlap_area > 0 else 1.0
    return valid_mask, masked_ratio
