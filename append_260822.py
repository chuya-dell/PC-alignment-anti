import pandas as pd
import os

df = pd.read_csv(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv")

data = []
def add(date, exp, sid, conc, rep, status='valid', reason=''):
    data.append({
        'Date': date, 'ExperimentType': exp, 'SampleID': sid, 
        'Concentration_M': conc, 'ReplicateOf': rep, 'NumberStatus': status,
        'WellStatus': 'normal', 'DriedOut': 'FALSE', 'AdjacentContamination': 'FALSE',
        'WellPosition': '', 'ImagingDateTime': '', 'SubstrateID': 1, 'Notes': reason
    })

# 260822 DNA
add('260822', 'DNA', 0, 0, '', 'valid', 'Blank')
add('260822', 'DNA', 1, 1e-9, '')
add('260822', 'DNA', 3, 1e-11, '')
add('260822', 'DNA', 4, 1e-12, '')
add('260822', 'DNA', 5, 1e-13, '')
add('260822', 'DNA', 6, 1e-14, '')
add('260822', 'DNA', 7, 1e-15, '')

new_df = pd.DataFrame(data)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer\experiment_ledger.csv", index=False)
print("Appended 260822!")
