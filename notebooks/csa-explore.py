import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


### Load data

filepath = 'data/astral/raw/reprocessed-data.csv'
astral = pd.read_csv(filepath, index_col=0)
raw = astral.iloc[:, astral.columns.str.startswith('CA')]
raw.replace(0, np.nan, inplace=True)
csa = np.log2(raw)
filepath = 'data/csa/processed/csa-log.csv'
csa.to_csv(filepath)

filepath = 'data/astral/metadata/metadata-CSA-full.csv'
# filepath = 'data/csa/metadata197.csv'
metadata = pd.read_csv(filepath, index_col=0)
# replace na values with 'unknown'
metadata.scid_1.fillna('Control', inplace=True)
metadata.scid_2.value_counts()

filepath = 'data/astral/metadata/experimental-metadata.csv'
exp_metadata = pd.read_csv(filepath, index_col=1)

filepath = 'data/astral/metadata/raw_files-time.txt'
runtime = pd.read_table(filepath, sep='\s+')
# only extract substring between last '/' and '.raw'
runtime['ID'] = runtime['Name'].str.extract(r'/Astral-1_.+_([^/]+)\.raw$')
runtime.set_index('ID', inplace=True)
runtime['DateTime'] = pd.to_datetime(runtime['Date'] + ' ' + runtime['Time'])
# join on other index
exp_metadata1 = exp_metadata.join(
    runtime[['DateTime']], on='Polypeptide Novogene ID', how='left'
)
full_metadata = metadata.join(exp_metadata1, how='left')
full_metadata.columns
full_metadata.iloc[:5,33:]


# Plot: PCA
csa_full = csa.T.dropna(axis=1, how='any')
csa_full.shape
csa_full.columns

pca = PCA(n_components=2)

csa_pca = pca.fit_transform(csa_full)
Z_csa = pd.DataFrame(csa_pca, index=csa_full.index, columns=['PC1', 'PC2'])
Z_csa = Z_csa.join(full_metadata)
Z_csa.comorbidities.fillna('Not applicable', inplace=True)

full_metadata.DateTime.dtype
Z_csa.columns


fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=Z_csa, x='PC1', y='PC2',
    hue='group', style='comorbidities',
    edgecolor=None, alpha=0.7, ax=ax
)
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0
)
plt.tight_layout()
filepath = 'tmp/astral/csa/fig/pca-csa_249.pdf'
fig.savefig(filepath, dpi=300)


# Filtering out
csa_full.shape # 249 proteins
N = csa.shape[0] # 1757 proteins
M = csa.shape[1] # 1757 proteins
pct_na = csa.isna().sum() / N

# plot histogram
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(pct_na, bins=30, kde=False, ax=ax)
ax.set_xlabel('Percentage of missing values')
filepath = 'tmp/astral/csa/fig/hist-pct_na.pdf'
fig.savefig(filepath, dpi=300)

(csa.isna().sum() / N > 0.5).sum()
csa_fltr = csa.loc[:, pct_na <= 0.5]

filepath = 'data/csa/processed/csa-filtered.csv'
csa_fltr.to_csv(filepath)


# Imputation (k-NN)
