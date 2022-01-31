"""
Todo:

    import sys
    sys.path.append('../ANALYTICS_LIB')
    from viz_first import visualize
    visualize(df,subset = columns,
            violinplot_columns = df.columns[4:11])

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from sklearn.decomposition import PCA


class Feature_Transformers:
    """
    以下の分析コンポーネントのためのヘルパークラス。
    Feature_Transformers
    - PCA
    """

    def __init__(self):
        plt.style.use('ggplot')
        # self.name = name
        # self.action = action
        # self.description = description

    ######################################################
    # print('\n\nNUM OF NULL & DTYPES: \n')
    def run_pca(self, df, pca_columns):
        """
        pca
        - 実行
        - 累積寄与率
        - 第二主成分のプロット×カテゴリ
        """
        # pcaを実行
        pca = PCA()
        pca.fit(df[pca_columns])

        # dataframeを主成分空間へ写像する
        feature = pca.transform(df[pca_columns])
        # binary_col='Legendary'

        def cumulative_contribution_rate():
            # Cumulative contribution rate
            plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
            plt.xlabel("Number of principal components")
            plt.ylabel("Cumulative contribution ratio")
            plt.grid()
            plt.show()

        cumulative_contribution_rate()

        def two_label_plot(binary_col):
            # メモ：主成分をカテゴリに分けてプロットし、境界線を見つける
            # 第二主成分までを使用
            plt.figure(figsize=(8, 8))
            for binary in [True, False]:
                plt.scatter(feature[df[f'{binary_col}'] == binary, 0],
                            feature[df[f'{binary_col}'] == binary, 1],
                            alpha=0.6, label=binary)

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.7))
            plt.grid()
            plt.show()
        # two_label_plot(binary_col)
        return pca

# if sample_size:
#     print(f'\n\nSince sample_size={sample_size} is set, dataframe is sampled.')
#     df = df.sample(n=sample_size,random_state=0,replace=False)
#     df = df.sort_index(ascending=True)