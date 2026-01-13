import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Load dataset
file = "data/tmp/spliceoforms/sepepquant/sepep_mapping_table.txt"

file = "data/tmp/spliceoforms/sepepquant/sepep_quant_matrix.txt"
raw = pd.read_table(file, index_col=0)

file = "data/astral/processed/metadata-lyriks407.csv"
metadata = pd.read_csv(file, index_col=0)
nid = pd.concat([
    metadata["Polypeptide.Novogene.ID"][:-5].str[:-3],
    metadata["Polypeptide.Novogene.ID"][-5:]
])
peptide_novogene_id_map = dict(zip(nid, metadata.index))

### Pre-processing

idx = pd.MultiIndex.from_tuples(
    raw.index.str.split("_").map(tuple),
    names=["Symbol", "Order", "Class"],
)
raw.index = idx
# Remove C5 SEPEPs
raw = raw[~(raw.index.get_level_values("Class") == "C5")]
# TODO: Remove all NAN rows
# TODO: Remove all genes with only one SEPEP
raw.isna().sum(axis=1)

### Intensities matrix
def map_colnames(x, dict):
    x.columns = x.columns.str[11:24]
    x.columns = x.columns[:-5].append(x.columns[-5:].str[-3:])
    x.columns = x.columns.map(dict)
    return x.iloc[:, :402] # keep only LYRIKS samples


intensities = raw.iloc[:, raw.columns.str.endswith("Intensity")]
intensities = map_colnames(intensities, peptide_novogene_id_map)
intensities.head()

(~(intensities.isna() | intensities.eq(0))).sum()
intst_pct_na = intensities.isna().sum(axis=1) / intensities.shape[1]
intst_pct_na

### Counts matrix
counts = raw.iloc[:, raw.columns.str.endswith("Count")]
counts = map_colnames(counts, peptide_novogene_id_map)

# Normalize counts by library size and log2 transform!
lib_sizes = counts.sum()
norm_counts = counts.div(lib_sizes, axis=1) * np.median(lib_sizes)
lognorm_counts = np.log2(norm_counts)
counts.to_csv("data/tmp/spliceoforms/sepepquant-counts.csv")
lognorm_counts.to_csv("data/tmp/spliceoforms/sepepquant-lognorm_counts.csv")

# Missing values: Only has NaN
counts.isna().sum(axis=1).plot.hist(bins=50)
plt.show()

counts_288 = counts[counts.isna().sum(axis=1) < 201]
counts_13 = counts[counts.isna().sum(axis=1) == 0]
counts_288.fillna(0, inplace=True)

lognorm_counts.shape
lognorm_counts_288 = lognorm_counts[lognorm_counts.isna().sum(axis=1) < 201]
lognorm_counts_13 = lognorm_counts[lognorm_counts.isna().sum(axis=1) == 0]
lognorm_counts_288.fillna(0, inplace=True)

# Plot PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(lognorm_counts_288.T)
Z = pd.DataFrame(
    Z,
    index=lognorm_counts_288.columns,
    columns=["PC1", "PC2"]
)
Z["Batch"] = metadata.loc[Z.index, "Extraction.Date"]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=Z,
    x="PC1",
    y="PC2",
    hue="Batch",
    s=50,
    alpha=0.8,
    edgecolor=None,
)
file = "tmp/astral/peptides/fig/pca_sepq-lognorm_counts_288.pdf"
plt.savefig(file, dpi=300)


# TODO: Missing values
# TODO: MVI
# TODO: Batch correction
