import pandas as pd
import numpy as np


# Extraction date
filepath = 'data/astral/metadata/metadata_experimental-all_645_13.csv'
expt_metadata = pd.read_csv(filepath, index_col=0)
expt_metadata.rename(columns={'Sample.Name': 'sn'}, inplace=True)
expt_metadata.drop(columns='sn', inplace=True)
# expt_metadata.sn = expt_metadata.sn.str.replace('_00', '_0')
# expt_metadata.sn = expt_metadata.sn.str.replace('_06', '_6')

# State
filepath = 'data/astral/metadata/ZH-states-all.csv'
states = pd.read_csv(filepath, index_col=0)
states['sn'] = states.index
states.sn = states.sn.str.replace('_.*$', '', regex=True)

### LYRIKS metadata ###

filepath = 'data/astral/processed/lyriks_605_402_01-knn5.csv'
lyriks_data = pd.read_csv(filepath, index_col=0)

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
filepath = 'data/astral/metadata/LYRIKS/metadata_65_60-antidepressant_anxiolytics-JY.csv'
lyriks_jy = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata_blood_collection-lyriks.csv'
lyriks_collection = pd.read_csv(filepath)
# lyriks_collection.columns
lyriks_collection.index = lyriks_collection.sn + \
    lyriks_collection.is_control + '_' + \
    lyriks_collection.timepoint.astype(str)
print(lyriks_collection.head())
# metadata10.index[~metadata10.index.isin(lyriks_collection.index)]
# lyriks_data.columns[lyriks_data.columns.str.startswith('L0567')]
# lyriks_collection.iloc[200:250]

### CSA metadata ###

filepath = 'data/astral/metadata/metadata-csa_200_37.csv'
csa_metadata = pd.read_csv(filepath, index_col=0)
csa_metadata.head()

### Change L0673_18 to L0673S_24 ###
states.index = states.index.str.replace('L0673S_18', 'L0673S_24')
metadata10.index = metadata10.index.str.replace('L0673S_18', 'L0673S_24')
expt_metadata.index = expt_metadata.index.str.replace('L0673S_18', 'L0673S_24')

### Integrate experimental metadata ###
metadata_expt = expt_metadata[[
    'Concentration.ng.ul.', 'Volume.ul.', 'Total.amount.ug.',
    'Study', 'Extraction.Date', 'Run.DateTime'
]].copy()
metadata_expt.columns = [
    'concentration', 'volume', 'total_amount',
    'study', 'extraction_date', 'run_datetime'
]
metadata_expt = metadata_expt[~metadata_expt.index.str.startswith('QC')]
metadata_expt.extraction_date.value_counts()
# Integrate collection datetime (missing for bipolar cohort)
collection_datetime = pd.to_datetime(pd.concat([
    lyriks_collection.date,
    csa_metadata.collection_datetime
]), format='mixed')
collection_datetime = collection_datetime.rename('collection_datetime')
metadata_expt = metadata_expt.join(collection_datetime, how='left')
print(metadata_expt.tail())

# CSA
csa = csa_metadata[['group', 'age', 'bmi', 'gender', 'ethnicity', 'smoking']].copy()
csa.smoking.value_counts()
csa.smoking.replace({'0': False, '1': True, ' ': pd.NA}, inplace=True)
csa.insert(0, 'timepoint', 0) 
csa.insert(0, 'sn', csa.index)
csa.smoking.tolist()
csa.index[csa.smoking.isna()] # CA114, CA155

# LYRIKS
lyriks = metadata73[[
    'sn', 'Period', 'age', 'bmi', 'gend', 'eth', 'smoke_stat'
]].join(metadata10[['final_label']], how='inner')
lyriks.gend = np.where(lyriks.gend == 2, 'Male', 'Female')
lyriks.eth = lyriks.eth.str.capitalize()
lyriks.smoke_stat = lyriks.smoke_stat.map({
    'non_smoker': False, 'quitted': False,
    'light': True, 'moderate': True, 'heavy': True
})
# Reorder columns
lyriks = lyriks[[
    'sn', 'Period', 'final_label',
    'age', 'bmi', 'gend', 'eth', 'smoke_stat'
]]
lyriks.final_label.replace({
    'ctrl': 'Healthy control',
    'rmt': 'Remit',
    'mnt': 'Maintain',
    'cvt': 'Convert',
}, inplace=True)

print(lyriks.head())
print(csa.head())
lyriks.columns = csa.columns

# Integrate
psy_demo = pd.concat([lyriks, csa])
psy_almost = psy_demo.join(metadata_expt, how='left')
print(psy_demo.shape)
print(psy_almost.shape)
print(psy_almost.head())
print(psy_almost.columns)

filepath = 'data/astral/metadata/metadata-psy_602_15.csv'
psy_almost.to_csv(filepath, index=True)

# TODO: Assign state
# filepath = 'data/astral/metadata/psy-metadata_602_14.csv'
# psy_almost = pd.read_csv(filepath, index_col=0)

psy_all = psy_almost.join(states[['label_mapped']], how='left')
psy_all.rename(columns={'label_mapped': 'state'}, inplace=True)
print(psy_all.shape)
# print(psy_all[psy_all.sn == 'L0365S'])
filepath = 'data/astral/metadata/metadata-psy_602_16.csv'
psy_all.to_csv(filepath, index=True)

# Check with Astral dataset


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
