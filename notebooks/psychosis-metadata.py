import pandas as pd
import numpy as np


# Extraction date
filepath = 'data/astral/metadata/experimental-metadata.csv'
expt_metadata = pd.read_csv(filepath, index_col=0)
expt_metadata.rename(columns={'Sample Name': 'sn'}, inplace=True)
expt_metadata.sn = expt_metadata.sn.str.replace('_00', '_0')
expt_metadata.sn = expt_metadata.sn.str.replace('_06', '_6')

# State
filepath = 'data/astral/metadata/ZH-states-all.csv'
states = pd.read_csv(filepath, index_col=0)
states['sn'] = states.index
states.sn = states.sn.str.replace('_.*$', '', regex=True)

### LYRIKS metadata ###

filepath = 'data/astral/metadata/LYRIKS/lyriks-baseline_medication.csv'
baseline_med = pd.read_csv(filepath, index_col=0)

# three uncategorised drugs
filepath = 'data/astral/metadata/LYRIKS/metadata_392_57.csv'
metadata57 = pd.read_csv(filepath, index_col=0)
metadata57.shape

filepath = 'data/astral/metadata/LYRIKS/metadata_2277_73.csv'
metadata73 = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata10-lyriks.csv'
metadata10 = pd.read_csv(filepath, index_col=0)

# antidepressant and anxiolytics use
filepath = 'data/astral/metadata/LYRIKS/metadata-antidepressant_anxiolytics-JY.csv'
lyriks_jy = pd.read_csv(filepath, index_col=0)

### CSA metadata ###

filepath = 'data/astral/metadata/metadata-csa_199_34.csv'
csa_metadata = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata-csa_199_37.csv'
csa37 = pd.read_csv(filepath, index_col=0)
csa37.columns

### Change L0673_18 to L0673S_24 ###
states.index = states.index.str.replace('L0673S_18', 'L0673S_24')
metadata10.index = metadata10.index.str.replace('L0673S_18', 'L0673S_24')
expt_metadata.sn = expt_metadata.sn.str.replace('L0673S_18', 'L0673S_24')

### Integrate metadata ###
expt_batch = expt_metadata[['sn', 'Extraction Date']].copy()
expt_batch.columns = ['sn', 'extraction_date']
expt_batch.set_index('sn', inplace=True)
expt_batch['study'] = 'LYRIKS'
expt_batch.loc[expt_batch.index.str.startswith('A'), 'study'] = 'BP'
expt_batch.loc[expt_batch.index.str.startswith('CA'), 'study'] = 'CSA'
# TODO: run datetime

# CSA
# add sn and period
csa = csa_metadata[['group', 'age', 'bmi', 'gender', 'ethnicity']].copy()
csa.insert(0, 'timepoint', 0) 
csa.insert(0, 'sn', csa.index)

# LYRIKS
lyriks = metadata73[['sn', 'Period', 'age', 'bmi', 'gend', 'eth']].join(
    metadata10[['final_label']], how='inner'
)
lyriks = lyriks[['sn', 'Period', 'final_label', 'age', 'bmi', 'gend', 'eth']]
print(lyriks.columns)
print(csa.columns)
lyriks.columns = csa.columns
lyriks.gender = np.where(lyriks.gender == 2, 'Male', 'Female')
lyriks.ethnicity = lyriks.ethnicity.str.capitalize()

# Integrate
psy_class = pd.concat([lyriks, csa])
psy_almost = psy_class.join(expt_batch, how='inner')
psy_almost.shape
filepath = 'data/astral/metadata/psy-metadata_599_9.csv'
psy_almost.to_csv(filepath, index=True)

# TODO: Assign state
filepath = 'data/astral/metadata/psy-metadata_599_9.csv'
psy_almost = pd.read_csv(filepath, index_col=0)

psy_all = psy_almost.join(states[['label_mapped']], how='left')
psy_all.rename(columns={'label_mapped': 'state'}, inplace=True)
# print(psy_all[psy_all.sn == 'L0365S'])
filepath = 'data/astral/metadata/psy-metadata_599_10.csv'
psy_all.to_csv(filepath, index=True)

# TODO: medication
csa_metadata

scid = lyriks_jy.iloc[:, 2:58:].copy()
scid.replace({-9999: 0}, inplace=True)
(scid == 1).any(axis=1).sum()

# TODO: comorbidities
# csa_metadata has data on comorbidities
# where is the LYRIKS one?

### Check possible mislabelling of L0673S_18 in proteomics data ###

mask = (metadata73['month_of_conversion'].notna()) # & (metadata73['Period'] == 24)
cvt_meta = metadata73.loc[mask, metadata73.columns[:7]]
cvt_meta_size = cvt_meta.groupby('sn').size().to_frame()

states_cvt = states[(states.stage_label == 'convert')]
cvt_size = states_cvt.groupby('sn').size().to_frame()

cvt_size.join(cvt_meta_size, how='outer', lsuffix='_states', rsuffix='_metadata')

# Patients (convert) that do not have proteomics data
missing_cvt = ['L0333S', 'L0336S', 'L0435S', 'L0635S', 'L0651S']

# Patients (convert) with lesser timepoint data than expected
lesser_cvt = ['L0073S', 'L0141S', 'L0561S', 'L0609S', 'L0673S']
# Only L0673 has sample 18 (which is not recorded in EMR) but lacks 24
# L0673 converted at M19 (M18 'should be' labelled as 'M24')

metadata73.loc[metadata73.sn.isin(lesser_cvt), metadata73.columns[:7]]
states[states.sn.isin(lesser_cvt)]
states.cohort.value_counts()
