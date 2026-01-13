import pandas as pd
import matplotlib.pyplot as plt


file = 'data/tmp/zhihao/unique_exon-intensities.csv'
intensities = pd.read_csv(file, index_col=[0,1,2,3,4])

intensities = intensities[~(intensities == 0).all(axis=1)]

file = 'data/tmp/zhihao/unique_exon-psm.csv'
psm = pd.read_csv(file, index_col=[0,1,2,3,4])
psm = psm[~(psm == 0).all(axis=1)]

pct_missing_psm = (psm == 0).sum(axis=1) / psm.shape[1]
# plt.figure()
# pct_missing_psm.plot.hist(bins=50, title='Percent missing (PSM)')
# filename = 'tmp/astral/peptides/fig/pct_missing-psm.pdf'
# plt.savefig(filename)
# plt.close()

# Aggregate to spliceoform level by summing
psm_spl = psm.groupby(level=[0,1,2]).sum()
psm_spl.shape
nspliceoform_psm = psm_spl.groupby(level=[0,1]).size()
id_thrb = psm_prot_nspl.index[nspliceoform_psm > 1]

# Same as PSM
intensities_spl = intensities.groupby(level=[0,1,2]).sum()
nspliceoform_intensities = intensities_spl.groupby(level=[0,1]).size()

intensities_spl = intensities.groupby(level=[0,1,2]).sum()
intensities_spl.groupby(level=[0,1]).size() > 1

# Plot PCA to check batch effects

# Plot individual protein
psm_multi_spl = psm_spl[psm_spl.index.droplevel(-1).isin(id_thrb)]
intensities_multi_spl = intensities_spl[
    intensities_spl.index.droplevel(-1).isin(id_thrb)]
psm_multi_spl.to_numpy().T
intensities_multi_spl.to_numpy().T
intensities_multi_spl

# TODO: Investigate why the two spliceoforms have exactly the same values

# TODO: Error in unique exons code. Exons identified are not unique.
intensities[intensities.index.droplevel([2,3,4]).isin(id_thrb)]
intensities[intensities.index.droplevel([2,3,4]).isin(id_thrb)].to_numpy().T
intensities[intensities.index.get_level_values(0) == "ACAN"]
intensities.index
