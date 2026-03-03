import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.decomposition import PCA


def subset(df, metadata, condition):
    """Subset dataframe based on metadata condition."""
    metadata = metadata.loc[df.columns]
    assert metadata.index.equals(df.columns)
    idx = metadata.query(condition).index
    return df[idx]


def plot_umap(ax, x, metadata, colourbar=False, **kwargs):
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
        ax=ax,
        **kwargs
    )

    if colourbar:
        import matplotlib.dates as mdates
        import matplotlib as mpl
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
