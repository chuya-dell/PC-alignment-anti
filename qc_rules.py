# -*- coding: utf-8 -*-
"""
Quality Control (QC) rules for Plasmon Image Analysis.
Based on manual visual inspection notes (2026-08-25).
"""

SAM_RULES = {
    0: {'status': 'EXCLUDE_SAMPLE', 'exclude_positions': [1,4,5,6,7,8], 'conc': 0},
    1: {'status': 'INCLUDE', 'exclude_positions': [], 'conc': 1e-9},
    2: {'status': 'QC_FLAGGED', 'exclude_positions': [], 'conc': 100e-12}, # Well leak suspect
    3: {'status': 'QC_FLAGGED', 'exclude_positions': [4,6,7,8], 'conc': 10e-12},
    4: {'status': 'QC_FLAGGED', 'exclude_positions': [1,2,5,6], 'conc': 1e-12},
    5: {'status': 'INCLUDE', 'exclude_positions': [], 'conc': 100e-15},
    8: {'status': 'EXCLUDE_SAMPLE', 'exclude_positions': [1,5,7,8], 'conc': 10e-15}, # Replaced by 13
    10: {'status': 'QC_FLAGGED', 'exclude_positions': [2,4,6,8], 'conc': 1e-15},
    11: {'status': 'EXCLUDE_SAMPLE', 'exclude_positions': [1,2,3,4], 'conc': 10e-12},
    12: {'status': 'INCLUDE', 'exclude_positions': [8], 'conc': 1e-12},
    13: {'status': 'INCLUDE', 'exclude_positions': [4], 'conc': 10e-15},
    14: {'status': 'INCLUDE', 'exclude_positions': [1,6], 'conc': 0},
    15: {'status': 'EXCLUDE_SAMPLE', 'exclude_positions': [], 'conc': 0}
}

DNA_RULES = {
    0: {'status': 'INCLUDE', 'exclude_positions': [7], 'conc': 0, 'run_id': 1},
    1: {'status': 'INCLUDE', 'exclude_positions': [], 'conc': 1e-9, 'run_id': 1},
    2: {'status': 'INCLUDE', 'exclude_positions': [], 'conc': 100e-12, 'run_id': 1}, # Substrate 2 suspect but unknown
    3: {'status': 'INCLUDE', 'exclude_positions': [], 'conc': 10e-12, 'run_id': 1},
    4: {'status': 'QC_FLAGGED', 'exclude_positions': [1,2,4,7], 'conc': 1e-12, 'run_id': 1},
    5: {'status': 'QC_FLAGGED', 'exclude_positions': [1,4,8], 'conc': 100e-15, 'run_id': 1},
    6: {'status': 'INCLUDE', 'exclude_positions': [3,8], 'conc': 10e-15, 'run_id': 1},
    7: {'status': 'INCLUDE', 'exclude_positions': [8], 'conc': 1e-15, 'run_id': 1},
    8: {'status': 'QC_FLAGGED', 'exclude_positions': [3,6,7], 'conc': 1e-12, 'run_id': 2},
    9: {'status': 'QC_FLAGGED', 'exclude_positions': [1,6,8], 'conc': 100e-15, 'run_id': 2}
}

def get_position_status(assay_type, sample_id, position_id):
    """
    Returns boolean (True if position should be included in main analysis, False if excluded)
    and string reason/status.
    """
    rules = SAM_RULES if assay_type == 'SAM' else DNA_RULES
    
    if sample_id not in rules:
        return True, "UNKNOWN_SAMPLE"
        
    rule = rules[sample_id]
    
    if rule['status'] == 'EXCLUDE_SAMPLE':
        return False, "EXCLUDE_SAMPLE"
        
    if position_id in rule['exclude_positions']:
        return False, "EXCLUDE_POSITION_DEFECT"
        
    # If the sample is QC_FLAGGED (e.g., well leak, unreliable), we still process the non-excluded positions
    # but we flag them so they can be plotted separately.
    if rule['status'] == 'QC_FLAGGED':
        return True, "QC_FLAGGED"
        
    return True, "INCLUDE"

def get_sample_metadata(assay_type, sample_id):
    rules = SAM_RULES if assay_type == 'SAM' else DNA_RULES
    return rules.get(sample_id, {})
