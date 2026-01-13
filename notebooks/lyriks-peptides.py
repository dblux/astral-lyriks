import asyncio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from notebooks.getters import get_isoforms, get_isoforms_async, get_exon_coordinates


##### Load data #####

### Biomarkers

file = "data/tmp/biomarkers/biomarkers-ancova.csv"
stats_ancova = pd.read_csv(file, index_col=0)
bm_ancova = stats_ancova.index
stats_ancova.head()

file = "data/tmp/biomarkers/biomarkers-elasticnet.csv"
stats_enet = pd.read_csv(file, index_col=0)
bm_enet = stats_enet.index
stats_enet.head()

file = "data/astral/etc/mongan-etable5-annot.csv"
mongan = pd.read_csv(file, index_col=0)
bm_mongan = mongan.index[mongan.q < 0.05]

file = "data/astral/etc/perkins.csv"
perkins = pd.read_csv(file, index_col=0)
bm_perkins = perkins.loc[
    perkins["UniProt/CAS"].str.startswith(("P","Q")),
    "UniProt/CAS"
]

file = "data/astral/etc/vawter.csv"
vawter = pd.read_csv(file)
bm_vawter = pd.Series(vawter.Gene.unique()) # only 4 out of 18 genes are present in pg_matrix

### Peptide-level analysis data 

file = "data/tmp/zhihao/dea_results.csv"
dea = pd.read_csv(file, index_col=[1,3,5]) # 3994 peptides, 445 proteins, 443 genes
dea.head()

### Protein-level analysis data

file = "data/tmp/zhihao/prots605-q_logfc.csv"
stats_prot605 = pd.read_csv(file, index_col=0)
stats_prot605.head()

### Metadata

file = "data/astral/processed/metadata-lyriks407.csv"
md = pd.read_csv(file, index_col=0, header=0)
md = md[md.label != "QC"]
md["period"] = md["period"].astype(int)
peptide_novogene_id_map = dict(zip(md["Polypeptide.Novogene.ID"], md.index))

# filepath = "data/astral/metadata/metadata-all.csv"
# metadata = pd.read_csv(filepath, index_col=0)
# metadata.Study.value_counts()

### Data

file = "data/astral/processed/combat_knn5_lyriks-605_402.csv"
lyriks = pd.read_csv(file, index_col=0, header=0).T

file = "data/tmp/zhihao/combat_global_knn.csv"
pepdata = pd.read_csv(file, index_col=[0,2,4]).iloc[:, 2:] # discard some columns
pepdata.rename(columns=peptide_novogene_id_map, inplace=True)

### Tmp files

# filepath = "data/tmp/spliceoforms/venn-69.txt"
# with open(filepath, "r") as f:
#     venn69 = [line.strip() for line in f.readlines()] 
# 
# filepath = "data/tmp/spliceoforms/venn-10.txt"
# with open(filepath, "r") as f:
#     venn10 = [line.strip() for line in f.readlines()] 
# 
# lyriks_symbols = lyriks.columns.map(uniprot_symbol_map)
# in_lyriks = [symbol in lyriks_symbols for symbol in venn69]
# print(in_lyriks)


### Peptide-level analysis

# Log fold change
id_cvt0 = md[
    (md.final_label == "cvt") &
    (md.period == 0)
].index
id_noncvt0 = md[
    (md.final_label.isin(["rmt", "mnt"])) &
    (md.period == 0)
].index
# All non-cvt (TP0) are from 4/9 batch and cvt (TP0) are from 5/9 batch
print(md.loc[id_noncvt0, "Extraction.Date"].value_counts())
logfc = pepdata[id_cvt0].mean(axis=1) - pepdata[id_noncvt0].mean(axis=1)
avg_mean = (pepdata[id_cvt0].mean(axis=1) + pepdata[id_noncvt0].mean(axis=1)) / 2
dea["logfc"] = logfc[dea.index]
dea["avg_mean"] = avg_mean[dea.index]

# Standard deviation among control samples
id_ctrl = md[md.final_label == "ctrl"].index
id_ctrl = id_ctrl[id_ctrl.isin(pepdata.columns)]
noise = pepdata[id_ctrl].std(axis=1)
dea["noise"] = noise[dea.index]

# TODO: Alternative noise measure (4/9 v.s. 5/9)

# Peptide lengths
dea["peptide_length"] = dea.index.get_level_values(1).str.len()

# x = "peptide_length"
# y = "qval"
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.scatterplot(
#     data=dea,
#     x=x,
#     y=y,
#     edgecolor=None,
#     alpha=0.7,
#     ax=ax,
# )
# # ax.set(ylabel="Min. q-value")
# filepath = f"tmp/astral/peptides/fig/scatter-{x}_{y}.png"
# print(filepath)
# fig.savefig(filepath, dpi=300, bbox_inches="tight")

# q-values
# Only unambiguous peptides present in DEA results
symbol_uniprot_map = {
    s: u for u, s in zip(dea.index.get_level_values(0), dea.Gene)
}
uniprot_symbol_map = {v: k for k, v in symbol_uniprot_map.items()}
assert len(symbol_uniprot_map.keys()) == len(set(symbol_uniprot_map.keys()))

uid_vawter = bm_vawter.map(symbol_uniprot_map)
uid_vawter.dropna(inplace=True)
bm_vawter[bm_vawter.isin(dea.Gene)]

# Split into LYRIKS, Mongan, Vawter, Rest
# If it appears in LYRIKS, assign to LYRIKS
bm_lyriks = bm_ancova.union(bm_enet)
bm_perkins.isin(dea.index.get_level_values(0)) # 3 out of 13 Perkins proteins present in DEA results
bm_perkins.isin(bm_mongan) # no overlap between Perkins and Mongan

### Aggregate by protein group

def calc_binary_entropy(x, predicate=lambda x: x < 0):
    """
    Compute the binary Shannon entropy of a vector after mapping its values
    to 1 and 0 using the predicate function.
    """
    enc = np.array([1 if predicate(i) else 0 for i in x])
    p = np.mean(enc)
    q = 1 - p
    if p == 0 or q == 0:
        return 0.0
    entropy = -(p * np.log2(p) + q * np.log2(q))
    return entropy


agg_dea = dea.groupby(dea.index.get_level_values(0)).agg({
    "qval": [
        "mean",
        "min",
        # ("mean_top2", lambda x: x.nsmallest(2).mean()),
    ],
    "coef": [
        "mean",
        "std",
        "size",
        ("mean_abs", lambda x: np.mean(np.abs(x))),
    ],
    "significant": "sum",
    "logfc": calc_binary_entropy,
    "noise": "mean",
})
agg_dea.columns = ["_".join(names) for names in agg_dea.columns.values]
agg_dea.rename(columns={
    "logfc_calc_binary_entropy": "logfc_entropy"
}, inplace=True)


# results = get_isoforms(agg_dea.index) # 483.3s
results = asyncio.run(get_isoforms_async(agg_dea.index)) # 131.2 s
protein_lengths = [result["canonical"]["length"] for result in results]
has_isoforms = [True if result["isoforms"] else False for result in results]
uid_isoforms = [result["accession"] for result in results if result["isoforms"]]

# Add protein-level information
# agg_dea["has_isoform"] = has_isoforms
# agg_dea["protein_length"] = protein_lengths
agg_dea["signature"] = "Rest"
agg_dea.loc[agg_dea.index.isin(uid_vawter), "signature"] = "Vawter"
agg_dea.loc[agg_dea.index.isin(bm_perkins), "signature"] = "Perkins"
agg_dea.loc[agg_dea.index.isin(bm_mongan), "signature"] = "Mongan"
agg_dea.loc[agg_dea.index.isin(bm_lyriks), "signature"] = "LYRIKS"
agg_dea["in_mongan"] = agg_dea.index.isin(bm_mongan)
agg_dea["in_lyriks"] = agg_dea.index.isin(bm_lyriks)
agg_dea["geq_2DE"] = agg_dea.significant_sum >= 2
agg_dea["geq_1DE"] = agg_dea.significant_sum >= 1
assert agg_dea.index.isin(stats_prot605.index).all()
agg_dea["protein_qval"] = stats_prot605.loc[agg_dea.index, "q"]

# Filter out proteins with only 1 peptide
fltr_agg_dea = agg_dea.query("coef_size > 1") # 132 out of 445 proteins have only 1 peptide

### Investigate protein sets of interest
# TODO: Investigate entropy of logfc
# TODO: Look at protein q-val
dea_geq2 = fltr_agg_dea[fltr_agg_dea.geq_2DE]
lowent_geq2de = dea_geq2.loc[dea_geq2.logfc_entropy == 0, ] 
bm_geq2 = dea_geq2.index
bm_lowent_geq2de = lowent_geq2de.index

dea_lowent_geq2de = dea[dea.index.get_level_values(0).isin(bm_lowent_geq2de)].copy()
lowent_geq2de_idx_uid = dea_lowent_geq2de.index.get_level_values(0)
dea_lowent_geq2de["in_mongan"] = lowent_geq2de_idx_uid.isin(bm_mongan)
dea_lowent_geq2de["in_lyriks"] = lowent_geq2de_idx_uid.isin(bm_lyriks)
dea_lowent_geq2de["has_isoform"] = lowent_geq2de_idx_uid.isin(uid_isoforms)
bm_lowent_geq2de[bm_lowent_geq2de.isin(bm_lyriks)]

# TODO: Low-entropy proteins
# Majority have small p-values
# lowent = fltr_agg_dea[fltr_agg_dea.logfc_entropy == 0]
# lowent_geq5 = lowent.loc[lowent.coef_size >= 5, ]
# dea_lowent_geq5 = dea[dea.index.get_level_values(0).isin(lowent_geq5.index)]

plt.figure(figsize=(6, 3))
lowent_geq2de.logfc_entropy.plot.hist(bins=20)
plt.xlabel("Protein q-value")
filepath = "tmp/astral/peptides/fig/hist_lowent_geq2de-logfc_ent.pdf"
plt.savefig(filepath, dpi=300, bbox_inches="tight")

# pd.crosstab(fltr_agg_dea.in_mongan, fltr_agg_dea.logfc_entropy == 0)


fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
axes = axes.flatten()
for i, uid in enumerate(bm_lowent_geq2de):
    dea_uid = dea[dea.index.get_level_values(0) == uid]
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=dea_uid,
        x="logfc",
        y="avg_mean",
        edgecolor=None,
        alpha=0.8,
        ax=axes[i]
    )
    q = stats_prot605.loc[uid, "q"]
    title = f"{uid} (q = {q:.3f})"
    # if uid in bm_lyriks:
    #     title = title + " *"
    axes[i].set_title(title)

filepath = f"tmp/astral/peptides/fig/proteins/scatter-logfc_avg_mean.pdf"
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# plot noise v.s. noise?


x = "Protein.Group"
y = "logfc"
z = "has_isoform"
plt.figure(figsize=(16, 5))
sns.stripplot(
    data=dea_lowent_geq2de,
    x=x,
    y=y,
    hue=z,
    jitter=True,
    dodge=False,
    palette={True: "red", False: "gray"},
)
plt.xticks(rotation=90)
plt.axhline(
    y=0.00,
    linestyle="--",
    linewidth=1,
    color="black"
)
plt.tight_layout()
filepath = f"tmp/astral/peptides/fig/jitter_lowent_geq2de-{z}-{x}_{y}.pdf"
plt.savefig(filepath, dpi=300)


# # Histogram of protein-level logfc entropy
# plt.figure(figsize=(6, 3))
# fltr_agg_dea.logfc_entropy.plot.hist(bins=30)
# plt.xlabel("Protein-level logfc entropy")
# filepath = "tmp/astral/peptides/fig/hist-logfc_entropy.png"
# plt.savefig(filepath, dpi=300, bbox_inches="tight")

x = "logfc_entropy"
y = "protein_qval"
z = "has_isoform"
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=geq2_dea,
    x=x,
    y=y,
    hue=z,
    palette={True: "red", False: "gray"},
)
plt.tight_layout()
filepath = f"tmp/astral/peptides/fig/scatter_{z}-{x}_{y}.png"
plt.savefig(filepath, dpi=300)


### Investigate Mongan proteins
bm_mongan.isin(agg_dea.index) # 33 out of 35 Mongan proteins are present in DEA results
bm_mongan.isin(fltr_agg_dea.index).sum() # 31 out of 35 Mongan proteins are present in DEA results

mongan20_uid = fltr_agg_dea.index[
    fltr_agg_dea.geq_2DE & fltr_agg_dea.in_mongan]
mongan20_sid = mongan20_uid.to_series().replace(uniprot_symbol_map)
print(mongan20_uid.intersection(bm_lyriks))
print(mongan20_sid)

# Plot logfc of all Mongan proteins
dea_mongan = dea[dea.index.get_level_values(0).isin(bm_mongan)].copy()
dea_mongan["label"] = "leq1"
dea_mongan.loc[
    dea_mongan.index.get_level_values(0).isin(bm_geq2), "label"] = "geq2"
dea_mongan.loc[
    dea_mongan.index.get_level_values(0).isin(bm_lyriks),
    "label"
] = "LYRIKS" # assumption: all LYRIKS proteins are in geq2 
dea_mongan["has_isoform"] = dea_mongan.index.get_level_values(0).isin(uid_isoforms)
dea_mongan.sort_values("label", inplace=True)
print(dea_mongan.groupby("Protein.Group").size())
dea_mongan.columns


x = "Protein.Group"
y = "logfc"
z = "label"
plt.figure(figsize=(16, 5))
sns.stripplot(
    data=dea_mongan,
    x=x,
    y=y,
    hue=z,
    jitter=True,
    dodge=False,
    # palette={True: "red", False: "gray"},
)
plt.xticks(rotation=90)
plt.axhline(
    y=0.05,
    linestyle="--",
    linewidth=1,
    color="black"
)
plt.tight_layout()
filepath = f"tmp/astral/peptides/fig/jitter_mongan_{z}-{x}_{y}.png"
plt.savefig(filepath, dpi=300)

# Investigate proteins with >= 2 DE peptides and not in Mongan
dea_geq2 = dea[dea.index.get_level_values(0).isin(bm_geq2)].copy()
dea_geq2["label"] = "geq2"
dea_geq2.loc[
    dea_geq2.index.get_level_values(0).isin(bm_lyriks), "label"] = "LYRIKS"
dea_geq2["has_isoform"] = dea_geq2.index.get_level_values(0).isin(uid_isoforms)
dea_geq2.sort_values("label", inplace=True)

x = "Protein.Group"
y = "logfc"
z = "has_isoform"
plt.figure(figsize=(16, 5))
sns.stripplot(
    data=dea_geq2[dea_geq2.index.get_level_values(0).isin(bm_low_entropy)],
    x=x,
    y=y,
    hue=z,
    jitter=True,
    dodge=False,
)
plt.xticks(rotation=90)
plt.axhline(
    y=0.05,
    linestyle="--",
    linewidth=1,
    color="black"
)
plt.tight_layout()
filepath = f"tmp/astral/peptides/fig/jitter_geq2_{z}-{x}_{y}.png"
plt.savefig(filepath, dpi=300)

fltr_agg_dea.in_mongan.sum()
fltr_agg_dea.iloc[fltr_agg_dea.in_mongan.to_numpy(), [1,4,5,6,7,8]]

# Investigate proteins with >= 2 DE peptides
fltr_agg_dea.groupby(["in_mongan", "geq_2DE"]).size().unstack(fill_value=0)
fltr_agg_dea.groupby(["in_mongan", "geq_1DE"]).size().unstack(fill_value=0)

# Perform a chi-squared test
ctab = pd.crosstab(fltr_agg_dea.in_mongan, fltr_agg_dea.geq_1DE)
chi2, p, dof, expected = chi2_contingency(ctab)
print(p)

(fltr_agg_dea.geq_2DE & fltr_agg_dea.in_mongan).sum()
(fltr_agg_dea.geq_2DE & ~fltr_agg_dea.in_mongan & fltr_agg_dea.in_lyriks).sum()
fltr_agg_dea[fltr_agg_dea.in_lyriks]


# 445 proteins total
x = "in_mongan"
y = "logfc_mean"
z = "geq_2DE"
fig, ax = plt.subplots(figsize=(4, 4))
sns.stripplot(
    data=fltr_agg_dea,
    x=x,
    y=y,
    hue=z,
    dodge=True,
    edgecolor=None,
    alpha=0.8,
    ax=ax,
)
# ax.set(ylabel="No. of peptides")
filepath = f"tmp/astral/peptides/fig/jitter-{x}_{y}.png"
fig.savefig(filepath, dpi=300, bbox_inches="tight")


# TODO: Plot Histogram instead
x = "geq_2DE"
y = "coef_size"
fig, ax = plt.subplots(figsize=(4, 4))
sns.histplot(
    data=fltr_agg_dea[fltr_agg_dea.coef_size <= 20],
    x=y,
    hue=x,
    multiple="dodge",
    edgecolor=None,
    alpha=0.8,
    ax=ax,
)
ax.set(xlabel="No. of peptides")
filepath = f"tmp/astral/peptides/fig/hist-{x}_{y}.png"
fig.savefig(filepath, dpi=300, bbox_inches="tight")


# TODO: Investigate noise as well as no. of imputed values?

x = "protein_length"
y = "coef_size"
hue = "geq_2DE"
style = "in_mongan"
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=fltr_agg_dea[(fltr_agg_dea.coef_size <= 20) & (fltr_agg_dea.protein_length <= 2000)],
    x=x,
    y=y,
    hue=hue,
    style=style,
    edgecolor=None,
    alpha=0.7,
    ax=ax,
)
ax.set(ylabel="No. of peptides")
filepath = f"tmp/astral/peptides/fig/scatter_{hue}_{style}-{x}_{y}.png"
print(filepath)
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# TODO: Explore coefficients and other measures to differentiate peptides
# TODO: Invetigate why entropy = 0 proteins were not discovered

# # Deltas against protein level
# 
# prot445 = stats_prot605.index.intersection(agg_dea.index)
# deltas = stats_prot605.loc[prot445, "q"] - agg_dea.loc[prot445, "qval_min"]
# assert deltas.index.isin(agg_dea.index).all()
# agg_dea["delta_q"] = deltas[agg_dea.index]
# fltr_agg_dea = agg_dea.query("coef_size > 1")

fig, ax = plt.subplots(figsize=(4, 4))
sns.stripplot(
    data=fltr_agg_dea,
    x="signature",
    y="delta_q",
    hue="has_isoform",
    dodge=True,
    edgecolor=None,
    alpha=0.8,
    ax=ax,
)
ax.set(ylabel="Protein q - min. peptide q")
filepath = "tmp/astral/peptides/fig/jitter-delta_q.png"
fig.savefig(filepath, dpi=300, bbox_inches="tight")

x = "coef_size"
y = "qval_min"
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=fltr_agg_dea[fltr_agg_dea.coef_size <= 20],
    x=x,
    y=y,
    hue="signature",
    # style="has_isoform",
    edgecolor=None,
    alpha=0.7,
    ax=ax,
)
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0
)
plt.tight_layout()
ax.set(xlabel="No. of peptides")
filepath = f"tmp/astral/peptides/fig/scatter-{x}_{y}.png"
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# TODO: Explore AutoExon outputs

file = "data/tmp/zhihao/gene-entropy_isoform.tsv"
genes = pd.read_table(file, index_col=0)
genes.head()

# Entropy
genes["in_ancova"] = genes.index.isin(bm_ancova.index)
genes["in_enet"] = genes.index.isin(bm_enet.index)

file = "tmp/astral/peptides/bm-ancova.csv"
genes.loc[genes.in_ancova].drop(["in_ancova", "in_enet"], axis=1).to_csv(file)

file = "tmp/astral/peptides/bm-enet.csv"
genes.loc[genes.in_enet].drop(["in_ancova", "in_enet"], axis=1).to_csv(file)

# Retaining proteins with at least n peptides (for entropy)
n_total = 6
genes_supp = genes[genes.Total >= n_total]
genes_supp.has_isoform.value_counts() # T: 98, F: 118

# Plot no. of peptides mapping to proteins
isoforms_20 = genes.loc[genes.has_isoform].sort_values("DE_count", ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=isoforms_20,
    x=isoforms_20.index,
    y="DE_count",
    hue="has_isoform",
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xlabel("")
ax.set_ylabel("Number of DE peptides")
filepath = f"tmp/astral/peptides/fig/barplot-isoforms20.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")

sns.histplot(
    data=genes_supp,
    x="Entropy",
    hue="has_isoform",
    # stat="density",
    # common_norm=True,
    bins=15,
    multiple="dodge",
)
plt.title(f"Proteins (>= {n_total} peptides)")
filepath = f"tmp/astral/peptides/fig/hist-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")

# Comparing proteins in LYRIKS (ANCOVA) v.s. rest
# bm_ancova.index.isin(genes.index).sum()  # 15
genes_supp["in_ancova"] = genes_supp.index.isin(bm_ancova.index)
ctab = pd.crosstab(genes_supp.has_isoform, genes_supp.in_ancova)
print(ctab)
genes_supp.loc[genes_supp.in_ancova, ["Entropy", "has_isoform"]]
genes_supp.loc[genes_supp.in_ancova]

plt.figure(figsize=(6, 3))
ax = sns.histplot(
    data=genes_supp,
    x="Entropy",
    hue="in_ancova",
    multiple="dodge",
)
ax.get_legend().set_title("In ANCOVA")
plt.title(f"Proteins (>= {n_total} peptides)")
# save fig
filepath = "tmp/astral/peptides/fig/hist_lyriks-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 3))
ax = sns.kdeplot(
    data=genes_supp,
    x="Entropy",
    hue="in_ancova",
    common_norm=False,
    clip=(0, 1),
    palette={True: "red", False: "gray"},
)
ax.get_legend().set_title("In ANCOVA")
plt.title(f"Proteins (>= {n_total} peptides)")
# save fig
filepath = "tmp/astral/peptides/fig/kde_lyriks-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()


# ### Protein-level analysis
# 
# # Model 1A: cvt (M0) v.s. non-cvt (M0)
# # Prognostic markers
# # Only M0 and exclude ctrl samples
# md_1a = md[(md.final_label != "ctrl") & (md.period == 0)]
# md_1a.final_label.replace({"rmt": 0, "mnt": 0, "cvt": 1}, inplace=True)
# lyriks_1a = lyriks.loc[md_1a.index]
# print(md_1a.final_label.value_counts()) # imbalanced
# 
# # Psychosis prognostic (ANCOVA, BH)
# # Dataframe with proteomic and clinical features
# # relapse is labelled as mnt
# data = pd.concat([lyriks_1a, md_1a[["final_label", "age", "gender"]]], axis=1)
# print(data.final_label.value_counts()) # imbalanced
# data.head()
# 
# pvalues = []
# coeffs = []
# for prot in data.columns[:-3]:
#     print(prot)
#     model = ols(
#         f"{prot} ~ final_label + age + gender",
#         data=data
#     ).fit()
#     pvalues.append(model.pvalues["final_label"])
#     coeffs.append(model.params["final_label"])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")
# stats = pd.DataFrame(
#     {"Coefficient": coeffs, "p": pvalues, "q": qvalues},
#     index=data.columns[:-3]
# )
# stats.head()
# 
# # logfc
# means_noncvt = lyriks_1a[md_1a.final_label == 0].mean()
# means_cvt = lyriks_1a[md_1a.final_label == 1].mean()
# logfc = means_cvt - means_noncvt
# stats["logfc"] = logfc[stats.index]
# filepath = "tmp/astral/peptides/prots605-q_logfc.csv"
# stats.to_csv(filepath)
# 
# 
# # Divide into two groups
# stats.loc[venn69, :].to_csv("tmp/astral/peptides/stats-venn69.csv")
# stats.loc[venn10, :].to_csv("tmp/astral/peptides/stats-venn10.csv")
# 
# ### Compare 69 v.s. 10
# filepath = "data/tmp/spliceoforms/comparison-69_10.csv"
# comparison = pd.read_csv(filepath, index_col=0)
# comparison.columns
# comparison.head()
# 
# comparison1 = comparison.merge(
#     stats[["q"]], left_index=True, right_index=True
# )
# comparison1.columns
# comparison1.avg_qval < 0.05 & comparison1.q > 0.05 
# 
# sns.stripplot(
#     data=comparison,
#     x="group",
#     y="entropy",
#     hue="has_isoforms",
# )
# plt.show()
# 
# sns.scatterplot(
#     data=comparison,
#     x="avg_qval",
#     y="entropy",
#     hue="has_isoforms",
#     style="group",
#     palette={True: "red", False: "grey"},
#     alpha=0.7,
#     edgecolor=None,
# )
# filepath = "tmp/astral/peptides/fig/scatter-avg_qval_entropy.png" 
# plt.savefig(filepath, dpi=300, bbox_inches="tight")
# plt.close()
# 
# sns.scatterplot(
#     data=comparison1,
#     x="nonDE_pct",
#     y="q",
#     hue="has_isoforms",
#     style="group",
#     palette={True: "red", False: "grey"},
#     alpha=0.7,
#     edgecolor=None,
# )
# filepath = "tmp/astral/peptides/fig/scatter-protein_peptide.png" 
# plt.savefig(filepath, dpi=300, bbox_inches="tight")
# plt.close()
# 
# plt.show()
# 
# itih1_1_uid = "P19827-1"
# itih2_1_uid = "P19823"
# results = get_isoforms(itih1_1_uid)
# results
# 
# kng1_2_uid = "P01042-2"
# exons = get_exon_coordinates(kng1_2_uid)
# print(exons)
