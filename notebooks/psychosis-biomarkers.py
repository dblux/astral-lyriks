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


def subset(df, metadata, condition):
    """Subset dataframe based on metadata condition."""
    metadata = metadata.loc[df.columns]
    assert metadata.index.equals(df.columns)
    idx = metadata.query(condition).index
    return df[idx]


filepath = 'data/astral/metadata/psy-metadata_599_10.csv'
metadata = pd.read_csv(filepath, index_col=0)

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
baseline_uhr = subset(
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

ctrl = subset(psyc, metadata, cond)
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


# TODO: Plot PCA, UMAP and clustering?

def plot_umap(x, metadata, hue=None, alpha=1, ax=None, **kwargs):
    reducer = umap.UMAP()
    z = reducer.fit_transform(x.transpose())
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['UMAP1', 'UMAP2']
    )
    z = z.join(metadata)
    ax = sns.scatterplot(
        data=z,
        x='UMAP1',
        y='UMAP2',
        edgecolor=None,
        hue=hue,
        alpha=alpha,
        ax=ax,
        **kwargs
    )
    return ax


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
