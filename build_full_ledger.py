import pandas as pd

data = []

def add(date, exp, sid, conc, rep, status='valid', reason=''):
    data.append({
        'Date': date, 'ExperimentType': exp, 'SampleID': sid, 
        'Concentration_M': conc, 'ReplicateOf': rep, 'NumberStatus': status,
        'WellStatus': 'normal', 'DriedOut': 'FALSE', 'AdjacentContamination': 'FALSE',
        'WellPosition': '', 'ImagingDateTime': '', 'SubstrateID': 1, 'Notes': reason
    })

# 260828 SAM (Estimated)
for i, c in zip([1,2,3,4,5,6,7,8], [1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 0]):
    add('260828', 'SAM', i, c, '', 'valid', 'Estimated from standard series')

# 260828 DNA mismatch
add('260828', 'DNA', 1, 1e-9, '')
add('260828', 'DNA', 2, 1e-10, '')
add('260828', 'DNA', 9, 1e-11, 3, 'valid', 'Replacement for 3')
add('260828', 'DNA', 4, 1e-12, '')
add('260828', 'DNA', 10, 1e-14, 6, 'valid', 'Replacement for 6')
add('260828', 'DNA', 7, 1e-15, '')
add('260828', 'DNA', 8, 0, '')
add('260828', 'DNA', 11, -1, '', 'mismatch', 'Mismatch control 11') # -1 to ignore in dose-response
add('260828', 'DNA', 12, -1, '', 'mismatch', 'Mismatch control 12')

# 260827 DNA
add('260827', 'DNA', 1, 1e-9, '', 'missing', 'To be excluded/redone')
add('260827', 'DNA', 2, 1e-10, '')
add('260827', 'DNA', 3, 1e-11, '')
add('260827', 'DNA', 4, 1e-12, '')
add('260827', 'DNA', 5, 1e-13, '')
add('260827', 'DNA', 6, 1e-14, '', 'missing', 'To be excluded')
add('260827', 'DNA', 7, 1e-15, '')
add('260827', 'DNA', 0, 0, '')
add('260827', 'DNA', 9, 0, '', 'missing', 'To be excluded')
add('260827', 'DNA', 10, 0, '', 'missing', 'To be excluded')
add('260827', 'DNA', 11, 0, '', 'valid', 'Changed to blank')

# 260826 SAM
add('260826', 'SAM', 1, 1e-9, '')
add('260826', 'SAM', 2, 1e-10, '')
add('260826', 'SAM', 3, 0, '', 'missing', 'Skip')
add('260826', 'SAM', 12, 1e-11, 4, 'valid', 'Replaces 4')
add('260826', 'SAM', 10, 1e-12, 5, 'valid', 'Replaces 5')
add('260826', 'SAM', 6, 1e-13, '')
add('260826', 'SAM', 11, 1e-14, 7, 'valid', 'Replaces 7')
add('260826', 'SAM', 8, 1e-15, '')
add('260826', 'SAM', 9, 0, '')

# 260825 DNA
for i, c in zip([0,1,2,3,4,5,6,7], [0, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]):
    add('260825', 'DNA', i, c, '')
add('260825', 'DNA', 8, 1e-12, 4, 'valid', 'Remeasure 4')
add('260825', 'DNA', 9, 1e-13, 5, 'valid', 'Remeasure 5')

# 260824 SAM
add('260824', 'SAM', 0, 0, '', 'valid', 'Blank')
add('260824', 'SAM', 1, 1e-9, '')
add('260824', 'SAM', 2, 1e-10, '')
add('260824', 'SAM', 3, 1e-11, '')
add('260824', 'SAM', 4, 1e-12, '')
add('260824', 'SAM', 5, 1e-13, '')
add('260824', 'SAM', 8, 1e-14, '')
add('260824', 'SAM', 10, 1e-15, '')
add('260824', 'SAM', 11, 1e-11, 3)
add('260824', 'SAM', 12, 1e-12, 4)
add('260824', 'SAM', 13, 1e-14, 8)
add('260824', 'SAM', 14, 0, '', 'valid', 'Blank redo')

# 260707 1
for i, c in zip([1,2,3,4,6,8,9,10,11,12], [1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 0, 0, 1e-15]):
    add('2607071', 'SAM', i, c, '')

# 260707 2
add('2607072', 'SAM', 1, 1e-9, '')
add('2607072', 'SAM', 2, 1e-10, '')
add('2607072', 'SAM', 3, 1e-11, '')
add('2607072', 'SAM', 4, 1e-12, '')
add('2607072', 'SAM', 5, 1e-13, '')
add('2607072', 'SAM', 11, 1e-13, 5)
add('2607072', 'SAM', 6, 1e-14, '')
add('2607072', 'SAM', 12, 1e-14, 6)
add('2607072', 'SAM', 7, 1e-15, '', 'missing', 'Peeled off')
add('2607072', 'SAM', 9, 1e-15, '')
add('2607072', 'SAM', 8, 0, '')

# 260706
add('260706', 'SAM', 1, 1e-9, '')
add('260706', 'SAM', 2, 1e-10, '')
add('260706', 'SAM', 3, 1e-11, '')
add('260706', 'SAM', 6, 1e-15, '')
add('260706', 'SAM', 8, 1e-12, '', 'missing', 'Fail')
add('260706', 'SAM', 9, 1e-13, '')
add('260706', 'SAM', 10, 1e-14, '')
add('260706', 'SAM', 11, 1e-12, 8, 'valid', 'Redo of 8')
add('260706', 'SAM', 13, 0, '')

df = pd.DataFrame(data)
df.to_csv(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv", index=False)
print("Updated unified ledger!")
