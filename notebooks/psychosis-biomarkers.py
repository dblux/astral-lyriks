import sys
import os
import umap
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# sys.path.append(os.getcwd())
import biopy.utils as bp


filepath = 'data/astral/metadata/metadata-psy_602_16.csv'
metadata = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/metadata/metadata-csa_200_37.csv'
metadata_csa = pd.read_csv(filepath, index_col=0)
metadata_csa.comorbidities.fillna('No', inplace=True)
metadata_csa.collection_datetime = pd.to_datetime(
    metadata_csa.collection_datetime,
    format='mixed'
)

filepath = 'data/astral/etc/olink-csa.csv'
olink_stats = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/etc/annotation-olink_75.csv'
annot_olink = pd.read_csv(filepath, index_col=0)
olink_uniprot_map = {
    k: v for k, v in zip(annot_olink.index, annot_olink.uniprot)
}

# Data
filepath = 'data/astral/processed/csa-knn5.csv'
csa_knn = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/processed/lyriks_605_402_01-knn5.csv'
lyriks_knn = pd.read_csv(filepath, index_col=0)

psy_ids = csa_knn.columns.tolist()
psy_ids.extend(lyriks_knn.columns.tolist())

# with open('tmp/astral/psy-ids.txt', 'w') as f:
#     f.write('\n'.join(psy_ids))

filepath = 'data/astral/processed/lyriks_605_402_01-combat_knn5.csv'
lyriks_combat = pd.read_csv(filepath, index_col=0)

psy_knn = lyriks_knn.join(csa_knn, how='inner')
psy_combat = lyriks_combat.join(csa_knn, how='inner')

filepath = 'data/astral/processed/reprocessed-data.csv'
data = pd.read_csv(filepath, index_col=0)
csa_raw = data.iloc[:, data.columns.str.startswith('CA')].copy()
csa_raw.replace(0, np.nan, inplace=True)
csa = np.log2(csa_raw)

prot_fullcsa = ~csa.isna().any(axis=1)
csa_full = csa[prot_fullcsa]
csa_574 = csa.loc[psy_knn.index]
csa_zero_574 = csa_574.fillna(0)
csa_knn_574 = csa_knn.loc[psy_knn.index]

uniprot_gene_map = {
    k: v for k, v in zip(data.index, data.Gene)
}

##### Check metadata coverage #####

psy_ids = csa_knn.columns.append(lyriks_knn.columns)
psy_ids.shape
psy_ids.isin(metadata.index).sum()

# CA114 missing metadata
# smoking missing for CA155 
print(metadata.info())
for feature in metadata.columns:
    print(feature)
    print(metadata.index[metadata[feature].isna()].tolist())

# L0073S_24, L0417S_24
# L0567 has missing blood collection info
lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0073')]
lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0417')]
lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0365')]

##### Transform datetime data #####

# TODO: Transform for plotting and modelling
metadata.run_datetime = pd.to_datetime(
    metadata.run_datetime,
    format='mixed'
)
metadata.collection_datetime = pd.to_datetime(
    metadata.collection_datetime,
    format='mixed'
)

# TODO: Transform according to specific plot (earliest day is different for different subsets)
metadata['run_days'] = (
    metadata.run_datetime - metadata.run_datetime.min()
) / np.timedelta64(1, "D")
metadata['collection_days'] = (
    metadata.collection_datetime - metadata.collection_datetime.min()
) / np.timedelta64(1, "D")

metadata_csa_197['collection_date_sec'] = mdates.date2num(
    metadata_csa_197.collection_datetime
)

##### CSA #####

# CA086, CA114, CA115 not present in Astral CSA dataset
missing_samples = metadata_csa.index[~metadata_csa.index.isin(csa_full.columns)]
metadata_csa_197 = metadata_csa.loc[~metadata_csa.index.isin(missing_samples)].copy()
# TODO: Remove imputation of smoking
metadata_csa_197.loc['CA155', 'smoking'] = '0'

metadata_csa_197['collection_days'] = (
    metadata_csa_197.collection_datetime - metadata_csa_197.collection_datetime.min()
) / np.timedelta64(1, "D")
metadata_csa_197['collection_date_sec'] = mdates.date2num(
    metadata_csa_197.collection_datetime
)

lyriks_knn.columns[lyriks_knn.columns.str.startswith('L0365')]

### Correlation: Expression with collection days ###

# To decide between linear or splines modelling of collection days

# collate stats in a dataframe
rows = []
for prot in csa_full.index:
    corr = spearmanr(
        csa_full.loc[prot],
        metadata_csa_197.loc[csa_full.columns, 'collection_days']
    )
    row = {
        'protein': prot,
        'spearman_r': corr.correlation,
        'p_value': corr.pvalue
    }
    rows.append(row)
    
corr_stats = pd.DataFrame(rows).set_index('protein')
corr_stats.sort_values('spearman_r', ascending=False, key=abs, inplace=True)
prots_highcorr = corr_stats.index[:20]

for prot in prots_highcorr:
    # plot expression against collection days
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=metadata_csa_197.loc[csa_full.columns, 'collection_days'],
        y=csa_full.loc[prot],
        hue=metadata_csa_197.group,
        ax=ax
    )
    ax.set_title(f'{prot} (r = {corr_stats.loc[prot, "spearman_r"]:.2f})')
    filepath = f'tmp/astral/fig/days_expr-{prot}.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

plt.close()


### Linear model with splines ###
# Model days using splines

metadata_csa_197['collection_days_centered'] = metadata_csa_197.collection_days - metadata_csa_197.collection_days.mean()

covariates = ['group', 'bmi', 'age', 'gender', 'smoking', 'collection_days_centered']
data = pd.concat([
    csa_full.transpose(),
    metadata_csa_197.loc[csa_full.columns, covariates],
], axis=1)
data.group.replace({
    'Healthy control': 0,
    'Antipsychotic responsive': 1,
    'Clozapine responsive': 1,
    'Clozapine resistant': 1
}, inplace=True)

# Fit linear model with cubic spline 
pvalues = []
coefs = []
n_cov = len(covariates)
cov_string =  'cr(collection_days_centered, df=6) + bmi + age + C(gender) + C(smoking) + C(group)'
for prot in data.columns[:-n_cov]:
    expression = f'{prot} ~ ' + cov_string
    model = ols(expression, data=data).fit()
    print(expression)
    pvalues.append(model.pvalues['C(group)[T.1]'])
    coefs.append(model.params['C(group)[T.1]'])
    # table = sm.stats.anova_lm(model, typ=2)

_, qvalues, _, _ = multipletests(
    pvalues, alpha=0.05, method='fdr_bh'
)

stats = pd.DataFrame(
    {'p': pvalues, 'q': qvalues},
    index=csa_full.index
)
schizo_sig = stats[stats.q < 0.05].copy()
schizo_genes = schizo_sig.index.map(uniprot_gene_map).tolist()

filepath = 'tmp/astral/schizo-sig.csv'
with open(filepath, 'w') as f:
    f.writelines('\n'.join(schizo_genes))
    f.close()


### PCA and UMAP ###

def plot_pca(ax, x, metadata, colourbar=False, **kwargs):
    '''PCA plot for visualisation of batch effects.'''
    pca = PCA(n_components=2)
    z = pca.fit_transform(x.transpose())
    var_ratio = pca.explained_variance_ratio_
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['PC1', 'PC2']
    )
    z = z.join(metadata)
    ax = sns.scatterplot(
        data=z,
        x='PC1',
        y='PC2',
        edgecolor=None,
        ax=ax,
        **kwargs
    )
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')

    if colourbar:
        import matplotlib.dates as mdates
        import matplotlib as mpl
        # Metadata
        hue = kwargs.pop('hue', None)
        palette = kwargs.pop('palette', 'rocket')
        # Create normalization for colourbar
        norm = mpl.colors.Normalize(
            vmin=metadata[hue].min(),
            vmax=metadata[hue].max()
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        sm.set_array([])
        fig = ax.get_figure()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(hue)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    return ax

def plot_pca(ax, x, metadata, colourbar=False, **kwargs):
    '''PCA plot for visualisation of batch effects.'''
    pca = PCA(n_components=2)
    z = pca.fit_transform(x.transpose())
    var_ratio = pca.explained_variance_ratio_
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['PC1', 'PC2']
    )
    z = z.join(metadata)
    ax = sns.scatterplot(
        data=z,
        x='PC1',
        y='PC2',
        edgecolor=None,
        ax=ax,
        **kwargs
    )
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')

    if colourbar:
        import matplotlib.dates as mdates
        import matplotlib as mpl
        # Metadata
        hue = kwargs.pop('hue', None)
        palette = kwargs.pop('palette', 'rocket')
        # Create normalization for colourbar
        norm = mpl.colors.Normalize(
            vmin=metadata[hue].min(),
            vmax=metadata[hue].max()
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        sm.set_array([])
        fig = ax.get_figure()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(hue)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    return ax

metadata_csa_plot = metadata.loc[csa_full.columns]
metadata_csa_plot['collection_days'] = (
    metadata_csa_plot.collection_datetime - metadata_csa_plot.collection_datetime.min()
) / np.timedelta64(1, "D")
metadata_csa_plot.collection_days

fig, ax = plt.subplots(figsize=(10, 8))
ax = plot_pca(
    ax,
    csa_full,
    metadata_csa_plot,
    colourbar=True,
    hue='collection_days',
    alpha=0.6,
    palette='rocket',
    legend=False
)
# style_levels = metadata['group'].unique()
# markers = ["o", "s", "^", "D", "X", "v", "*", "P"]
# marker_map = {
#     level: marker for level, marker in zip(style_levels, markers)
# }
# handles = [
#     Line2D(
#         [0], [0],
#         marker=marker_map[level],
#         linestyle="",
#         color="black",
#         label=level
#     )
#     for level in style_levels
# ]
# ax.legend(
#     handles=handles,
#     title='group',
#     loc="lower right",
#     # bbox_to_anchor=(1.05, 0.5)
# )
plt.show()

filepath = 'tmp/astral/fig/pca_run_days-csa_full.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 8))
ax = bp.plot_umap(
    ax,
    csa_zero_574,
    metadata_csa_197,
    colourbar=True,
    hue='collection_datenum',
    style='group',
    alpha=0.6,
    palette='rocket',
    legend=False
)
style_levels = metadata_csa_197['group'].unique()
markers = ["o", "s", "^", "D", "X", "v"]
marker_map = {
    level: marker for level, marker in zip(style_levels, markers)
}
handles = [
    Line2D(
        [0], [0],
        marker=marker_map[level],
        linestyle="",
        color="black",
        label=level
    )
    for level in style_levels
]
ax.legend(
    handles=handles,
    title='group',
    loc="lower right",
    # bbox_to_anchor=(1.05, 0.5)
)
filepath = 'tmp/astral/fig/umap-csa_zero_574.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()


### OLINKS ###

csa_full_gene = csa_full.rename(index=protein_gene_map)
olink_stats_uniprot = olink_stats.rename(index=olink_uniprot_map)

olink_stats_uniprot.index.isin(csa_full.index).sum() # 0/75 proteins in LINKS are in csa (249)
olink_stats_uniprot.index.isin(csa.index).sum() # 1/75 proteins in LINKS are in csa (1757)
csa_full.index.sort_values().tolist()
olink_stats_uniprot.index.sort_values().tolist()

# ##### Biomarker identification #####
# 
# ### Prognostic signature ###
# # Without explicit batch correction
# # Using entire dataset
# 
# metadata.columns
# metadata.state.value_counts()
# metadata.group.value_counts()
# 
# # Subset baseline samples from cvt v.s. non-cvt
# # TODO: Should i use only LYRIKS dataset instead (it has more features)? Depends on whether the
# # analysis is to compare against across studies
# baseline_uhr = bp.subset(
#     psyc_knn, metadata,
#     'timepoint == 0 and group in ["cvt", "mnt", "rmt"]'
# )
# covariates = ['group', 'age', 'gender', 'extraction_date']
# data = pd.concat([
#     baseline_uhr.transpose(),
#     metadata.loc[baseline_uhr.columns, covariates],
# ], axis=1)
# data.group.replace({'cvt': 1, 'mnt': 0, 'rmt': 0}, inplace=True)
# data.extraction_date.value_counts()
# # metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8
# 
# # ANCOVA
# pvalues = []
# coefs = []
# n_cov = len(covariates)
# cov_string = ' + '.join(covariates)
# for prot in data.columns[:-n_cov]:
#     expression = f'{prot} ~ ' + cov_string
#     print(expression)
#     model = ols(expression, data=data).fit()
#     pvalues.append(model.pvalues['group'])
#     coefs.append(model.params['group'])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# _, qvalues, _, _ = multipletests(
#     pvalues, alpha=0.05, method='fdr_bh'
# )
# stats = pd.DataFrame(
#     {'p': pvalues, 'q': qvalues},
#     index=baseline_uhr.index
# )
# print(stats)
# 
# # prots = statvalues.index[statvalues.p < 0.01] # p-value
# prots = stats.index[stats.q < 0.05] # q-value
# prots.size
# 
# ### Short conversion signature ###
# 
# # Linear mixed models
# # TODO: Exact t2c metadata should be used
# 
# 
# ### Psychosis conversion biomarkers (UHR v.s. Schizo) ###
# 
# # Perfect confounding between state and study
# # Batch correct first before biomarker identification
# metadata.columns
# # metadata.query("study == 'CSA'")
# cond = "extraction_date == '5/9/24' and state == 'Control'"
# ctrl_metadata = metadata.query(cond)
# ctrl_metadata.extraction_date.value_counts()
# print(ctrl_metadata.study.value_counts())
# 
# ctrl = bp.subset(psyc_knn, metadata, cond)
# ctrl.head()
# ctrl.shape
# 
# covariates = ['study', 'age', 'gender']
# data = pd.concat([
#     ctrl.transpose(),
#     metadata.loc[ctrl.columns, covariates],
# ], axis=1)
# data.study.replace({'CSA': 1, 'LYRIKS': 0}, inplace=True)
# data.head()
# # metadata[metadata.extraction_date == '28/8/24'] # only baseline controls in 28/8
# 
# # ANCOVA
# pvalues = []
# coefs = []
# n_cov = len(covariates)
# cov_string = ' + '.join(covariates)
# for prot in data.columns[:-n_cov]:
#     expression = f'{prot} ~ ' + cov_string
#     print(expression)
#     model = ols(expression, data=data).fit()
#     pvalues.append(model.pvalues['study'])
#     coefs.append(model.params['study'])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# batch_effects = pd.Series(coefs, index=ctrl.index)
# 
# # Correct CSA batch effects
# corr_psyc = psyc_combat.copy()
# corr_psyc.loc[:,corr_psyc.columns.str.startswith('CA')] = corr_psyc.loc[
#     :,corr_psyc.columns.str.startswith('CA')].subtract(batch_effects, axis=0)
# # corr_psyc.equals(psyc_combat)
# 
# ### Investigate CSA ###
# 
# metadata_csa.columns
# 
# # Comorbidities
# # metadata_csa.comorbidities
# comorb_features = [
#     'group', 'comorbidities', 'comorbidities_specify', 'scid_2'
# ]
# metadata_csa[comorb_features].head(30)
# metadata_csa.columns
# metadata_csa.head()
# 
# # Investigate collection datetime
# 
# sns.histplot(
#     metadata_csa.collection_datetime,
#     bins=50
# )
# filepath = 'tmp/astral/fig/hist-collection-datetime.pdf'
# plt.savefig(filepath)
# 
# import matplotlib.dates as mdates
# metadata_csa['collection_datenum'] = mdates.date2num(metadata_csa.collection_datetime)
# 
# fig, ax = plt.subplots(figsize=(8, 8))
# ax = bp.plot_pca(
#     ax,
#     csa_full,
#     metadata_csa,
#     colourbar=True,
#     hue='collection_datenum',
#     style='group',
#     alpha=0.6,
#     palette='rocket',
#     legend=False
# )
# 
# # ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
# ax2 = bp.plot_pca(
#     ax2,
#     csa_full,
#     metadata_csa,
#     hue='group',
#     alpha=0.6
# )
# # ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
# 
# filepath = 'tmp/astral/fig/pca-csa.pdf'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()
#     
# ax = bp.plot_umap(
#     csa, metadata_csa,
#     hue='collection_datetime', style='group', alpha=0.6,
#     figsize=(8,5)
# )
# ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
# filepath = 'tmp/astral/fig/umap-csa.pdf'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# 
# 
# ### Investigate dataset after correcting CSA ###
# 
# metadata.columns
# ax = plot_umap(
#     psyc_combat, metadata,
#     hue='study', style='group', alpha=0.6
# )
# ax.figure.set_size_inches(8, 5)
# filepath = 'tmp/astral/fig/umap-combat-psyc.pdf'
# ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()
# 
# 
# ax = plot_umap(corr_psyc, metadata, hue='study', alpha=0.6, style='group')
# ax.figure.set_size_inches(8, 5)
# filepath = 'tmp/astral/fig/umap-corr-psyc.pdf'
# ax.figure.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()
# 
# # Perfect confounding between state and medication
# 
# 
# # TODO: Hierarchical clustering plots (CSA) and LYRIKS batch ()
