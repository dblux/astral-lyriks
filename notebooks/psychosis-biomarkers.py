import umap
import pandas as pd
import numpy as np
import statsmodels.api as sm
# from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import biopy.utils as bp


filepath = 'data/astral/metadata/psy-metadata_599_10.csv'
metadata = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata-csa_200_37.csv'
csa_metadata = pd.read_csv(filepath, index_col=0)
csa_metadata.comorbidities.fillna('No', inplace=True)
csa_metadata.collection_datetime = pd.to_datetime(
    csa_metadata.collection_datetime,
    format='mixed'
)
csa_metadata.collection_datetime

filepath = 'data/astral/processed/knn5_csa.csv'
csa = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/processed/knn5_lyriks-605_402-01.csv'
lyriks = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/processed/combat_knn5_lyriks-605_402-01.csv'
combat_lyriks = pd.read_csv(filepath, index_col=0)

psyc = lyriks.join(csa, how='inner')
combat_psyc = combat_lyriks.join(csa, how='inner')


##### Biomarker identification #####

### Prognostic signature ###
# Without explicit batch correction
# Using entire dataset

metadata.columns
metadata.state.value_counts()
metadata.group.value_counts()

# Subset baseline samples from cvt v.s. non-cvt
# TODO: Should i use only LYRIKS dataset instead (it has more features)? Depends on whether the
# analysis is to compare against across studies
baseline_uhr = bp.subset(
    psyc, metadata,
    'timepoint == 0 and group in ["cvt", "mnt", "rmt"]'
)
covariates = ['group', 'age', 'gender', 'extraction_date']
data = pd.concat([
    baseline_uhr.transpose(),
    metadata.loc[baseline_uhr.columns, covariates],
], axis=1)
data.group.replace({'cvt': 1, 'mnt': 0, 'rmt': 0}, inplace=True)
data.extraction_date.value_counts()
# metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8

# ANCOVA
pvalues = []
coefs = []
n_cov = len(covariates)
cov_string = ' + '.join(covariates)
for prot in data.columns[:-n_cov]:
    expression = f'{prot} ~ ' + cov_string
    print(expression)
    model = ols(expression, data=data).fit()
    pvalues.append(model.pvalues['group'])
    coefs.append(model.params['group'])
    # table = sm.stats.anova_lm(model, typ=2)

_, qvalues, _, _ = multipletests(
    pvalues, alpha=0.05, method='fdr_bh'
)
stats = pd.DataFrame(
    {'p': pvalues, 'q': qvalues},
    index=baseline_uhr.index
)
print(stats)

# prots = statvalues.index[statvalues.p < 0.01] # p-value
prots = stats.index[stats.q < 0.05] # q-value
prots.size

### Short conversion signature ###

# Linear mixed models
# TODO: Exact t2c metadata should be used


### Psychosis conversion biomarkers (UHR v.s. Schizo) ###

# Perfect confounding between state and study
# Batch correct first before biomarker identification
metadata.columns
# metadata.query("study == 'CSA'")
cond = "extraction_date == '5/9/24' and state == 'Control'"
ctrl_metadata = metadata.query(cond)
ctrl_metadata.extraction_date.value_counts()
print(ctrl_metadata.study.value_counts())

ctrl = bp.subset(psyc, metadata, cond)
ctrl.head()
ctrl.shape

covariates = ['study', 'age', 'gender']
data = pd.concat([
    ctrl.transpose(),
    metadata.loc[ctrl.columns, covariates],
], axis=1)
data.study.replace({'CSA': 1, 'LYRIKS': 0}, inplace=True)
data.head()
# metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8

# ANCOVA
pvalues = []
coefs = []
n_cov = len(covariates)
cov_string = ' + '.join(covariates)
for prot in data.columns[:-n_cov]:
    expression = f'{prot} ~ ' + cov_string
    print(expression)
    model = ols(expression, data=data).fit()
    pvalues.append(model.pvalues['study'])
    coefs.append(model.params['study'])
    # table = sm.stats.anova_lm(model, typ=2)

batch_effects = pd.Series(coefs, index=ctrl.index)

# Correct CSA batch effects
corr_psyc = combat_psyc.copy()
corr_psyc.loc[:,corr_psyc.columns.str.startswith('CA')] = corr_psyc.loc[
    :,corr_psyc.columns.str.startswith('CA')].subtract(batch_effects, axis=0)
# corr_psyc.equals(combat_psyc)

### Investigate CSA ###

csa_metadata.columns

# Comorbidities
comorb_features = [
    'group', 'comorbidities', 'comorbidities_specify', 'scid_2'
]
csa_metadata[comorb_features].head(30)

ax = bp.plot_pca(
    csa, metadata,
    hue='age', style='group', alpha=0.6,
    figsize=(8,5)
)
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
filepath = 'tmp/astral/fig/pca-csa.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
ax = bp.plot_umap(
    csa, csa_metadata,
    hue='collection_datetime', style='group', alpha=0.6,
    figsize=(8,5)
)
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
filepath = 'tmp/astral/fig/umap-csa.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')


### Investigate dataset after correcting CSA ###

metadata.columns
ax = plot_umap(
    combat_psyc, metadata,
    hue='study', style='group', alpha=0.6
)
ax.figure.set_size_inches(8, 5)
filepath = 'tmp/astral/fig/umap-combat-psyc.pdf'
ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()


ax = plot_umap(corr_psyc, metadata, hue='study', alpha=0.6, style='group')
ax.figure.set_size_inches(8, 5)
filepath = 'tmp/astral/fig/umap-corr-psyc.pdf'
ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

# Perfect confounding between state and medication


# TODO: Hierarchical clustering plots (CSA) and LYRIKS batch ()
