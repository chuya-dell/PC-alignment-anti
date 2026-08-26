import pandas as pd
import os

ledger_path = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv"
df = pd.read_csv(ledger_path)

new_ledger = [
    (260826, 'SAM', 1, 1e-9, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, ''),
    (260826, 'SAM', 2, 1e-10, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, ''),
    (260826, 'SAM', 3, 0, '', 'missing', 'unknown', 'FALSE', 'FALSE', '', '', 1, 'surface rough'),
    (260826, 'SAM', 4, 1e-11, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replaced by 12'),
    (260826, 'SAM', 5, 1e-12, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replaced by 10'),
    (260826, 'SAM', 6, 1e-13, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, ''),
    (260826, 'SAM', 7, 1e-14, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replaced by 11'),
    (260826, 'SAM', 8, 1e-15, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, ''),
    (260826, 'SAM', 9, 0, '', 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'NC'),
    (260826, 'SAM', 10, 1e-12, 5, 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replacement for 5'),
    (260826, 'SAM', 11, 1e-14, 7, 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replacement for 7'),
    (260826, 'SAM', 12, 1e-11, 4, 'valid', 'normal', 'FALSE', 'FALSE', '', '', 1, 'replacement for 4'),
]

new_df = pd.DataFrame(new_ledger, columns=df.columns)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv(ledger_path, index=False)
print("Updated ledger")

# Defect updates
defects = [
    (1, 2, 'Dust_Stain'),
    (4, 1, 'Dust'), (4, 2, 'Dust'), (4, 3, 'Dust'), (4, 4, 'Dust'), (4, 5, 'Stain'), (4, 6, 'Stain'),
    (5, 5, 'Stain'), (5, 6, 'Dust_Scratch'), (5, 7, 'Dust'),
    (6, 1, 'Dust'), (6, 2, 'Dust'), (6, 3, 'Dust'), (6, 4, 'Dust'), (6, 5, 'Dust'), (6, 6, 'Dust'), (6, 7, 'Stain'), (6, 8, 'Dust'),
    (7, 3, 'Dust'), (7, 7, 'Structural_Defect'),
    (8, 1, 'Dust'),
    (9, 1, 'Dust'), (9, 2, 'Dust'), (9, 3, 'Dust'), (9, 4, 'Dust'), (9, 5, 'Dust'), (9, 6, 'Stain'), (9, 8, 'Dust_Stain'),
    (10, 1, 'Stain_Dust'), (10, 3, 'Stain'), (10, 5, 'Dust'), (10, 7, 'Dust'),
    (11, 1, 'Stain'), (11, 3, 'Dust'), (11, 4, 'Stain'), (11, 6, 'Stain'),
    (12, 2, 'Dust_Stain'), (12, 3, 'Dust'), (12, 4, 'Stain'), (12, 5, 'Stain'), (12, 8, 'Dust')
]

defect_path = r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\defect_coordinates.csv"
with open(defect_path, 'r', encoding='utf-8') as f:
    content = f.read()

rows = []
for s, p, t in defects:
    r = 16 if 'dust' in t.lower() and 'stain' not in t.lower() else 40
    rows.append(f"260826,{s},{p},{s}-{p}-0.tif,{t},circle,0,0,,,{(r)},TRUE,TRUE,clear,TRUE")

with open(defect_path, 'w', encoding='utf-8') as f:
    f.write(content.strip() + '\n' + '\n'.join(rows) + '\n')
    
print("Updated defects")
