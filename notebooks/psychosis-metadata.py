import numpy as np
import pandas as pd


# Extraction date
filepath = 'data/astral/metadata/experimental-metadata.csv'
expt_metadata = pd.read_csv(filepath, index_col=0)
expt_metadata['Sample Name'] = expt_batch.sn.str.replace('_00', '_0')

# State
filepath = 'data/astral/metadata/ZH-states-all.csv'
states = pd.read_csv(filepath, index_col=0)

### LYRIKS metadata ###

filepath = 'data/astral/metadata/LYRIKS/lyriks-baseline_medication.csv'
baseline_med = pd.read_csv(filepath, index_col=0)

# antidepressant and anxiolytics use
filepath = 'data/astral/metadata/LYRIKS/metadata-antidepressnat_axiolytics-JY.csv'

# three uncategorised drugs
filepath = 'data/astral/metadata/LYRIKS/metadata_392_57.csv'
metadata57 = pd.read_csv(filepath, index_col=0)
metadata57.shape

# antidepressants use
filepath = 'data/astral/metadata/LYRIKS/metadata_2277_73.csv'
metadata73 = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata10-lyriks.csv'
metadata10 = pd.read_csv(filepath, index_col=0)
metadata73.loc['L0673S_18']

### CSA metadata ###

filepath = 'data/astral/metadata/metadata-CSA-full.csv'
csa_metadata = pd.read_csv(filepath, index_col=0)

### Integrate metadata ###

expt_batch = expt_metadata[['Sample Name', 'Extraction Date']].copy()
expt_batch.columns = ['sn', 'extraction_date']
expt_batch.sn = expt_batch.sn.str.replace('_00', '_0')
expt_batch.sn = expt_batch.sn.str.replace('_06', '_6')
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

psy_class = pd.concat([lyriks, csa])
psy_almost = psy_class.join(expt_batch, how='inner')

psy_all = psy_almost.join(states[['label_mapped']], how='inner')
psy_all.rename(columns={'label_mapped': 'state'}, inplace=True)
psy_all.head()
psy_all.shape

lyriks.shape
states.index.difference(psy_almost.index)

psy_class.shape
psy_class.index.difference(expt_batch.index)

expt_metadata.shape
expt_batch[expt_batch.study == 'LYRIKS']
expt_batch[expt_batch.study == 'CSA']

filepath = 'data/astral/metadata/psy-metadata_582_10.csv'
psy_all.to_csv(filepath, index=True)

states.shape
lyriks_idx = states[states.cohort == 'LYRIKS'].index
lyriks_idx.shape
states[states.cohort == 'CSA'].shape

states.index

lyriks.shape # lacking
csa.shape # more

lyriks_idx.difference(metadata57.index)

metadata57.shape
metadata74.shape

states.columns
psychosis.shape

expt_batch.shape

# comorbidities
# medication

# TODO: check astral data that L0673S_18 has been mapped properly. no such metadata
