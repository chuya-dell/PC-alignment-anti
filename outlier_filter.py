import pandas as pd
import numpy as np
import os
import re

# Explicit exclusion list based on experiment notes and abnormal physical artifacts
EXCLUDE_MAP = {
    '260706_sam_p200': ['8'],            # No.8: 1 pM (失敗・ゴミ)
    '260707_sam_p100_1': ['8', '6'],     # No.8: 0 M(疑問), No.6: 10 fM (顕著な画像欠損・異常低下)
    '260707_sam_p100_2': ['7']           # No.7: 1 fM (post測定時に剥離)
}

def is_excluded_sample(filename, dataset_name):
    base = os.path.basename(filename)
    series = re.split(r'[_\-\.]', str(base))[0]
    ex_list = EXCLUDE_MAP.get(dataset_name, [])
    return str(series) in [str(x) for x in ex_list]
