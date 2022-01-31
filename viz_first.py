"""
Todo:

    import sys
    sys.path.append('../ANALYTICS_LIB')
    from viz_first import visualize
    visualize(df,subset = columns,
            violinplot_columns = df.columns[4:11])

"""

from msilib.schema import Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys


def visualize(df,subset,violinplot_columns,print_unique=False,sample_size=False):
    """
    以下の可視化を行う。

    - NULLの行数(normal, any, all)とdf.dtypes
    - 50行のデータフレーム
    - 統計量
    - 散布図行列
    - 相関行列
    - violinplot_columns

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        df (pandas.DataFrame): pandas.DataFrame.
        subset (list): Column used for display(df.head(50).style.bar(subset = subset)).
        violinplot_columns (list): Column used for violinplot.

    Returns:
        None


    """
    plt.style.use('ggplot')

    print('\n')
    print('以下の情報を表示します')
    print('  - NUM OF NULL(normal, any, all) & DTYPES')
    print('  - FIRST 50 RECORD')
    print('  - DESCRIBE')
    print('  - SCATTER PLOT MATRIX')
    print('  - CORRELATION MATRIX')
    print('  - VIOLINPLOT')

    if sample_size:
        print(f'\n\nSince sample_size={sample_size} is set, dataframe is sampled.')
        df = df.sample(n=sample_size,random_state=0,replace=False)
        df = df.sort_index(ascending=True)
        

    ######################################################
    print('\n\nNUM OF NULL & DTYPES: \n')
    tmp = pd.concat([pd.DataFrame(df.isnull().sum(),columns=['num_of_null']),
                pd.DataFrame(df.dtypes,columns=['dtypes'])],axis=1)
    # tmp = pd.concat([tmp,pd.DataFrame(df.isnull().all().sum(),columns=['all_num'])])
    # tmp = pd.concat([tmp,pd.DataFrame(df.isnull().any().sum(),columns=['any_num'])])
    tmp = tmp.T
        
    all_null_record = pd.DataFrame(np.array(df.isnull().all(axis=1).sum()).reshape(-1,1),index=['all_null_record'],columns=[df.columns[0]])
    any_null_record = pd.DataFrame(np.array(df.isnull().any(axis=1).sum()).reshape(-1,1),index=['any_null_record'],columns=[df.columns[0]])
    tmp2 = pd.concat([all_null_record,any_null_record],axis=0)

    tmp = pd.concat([tmp,tmp2],axis=0)
    display(tmp)
    
    ######################################################
    print('\n\nFIRST 50 RECORD: \n')
    display(df.head(50).style.bar(subset=subset))

    ######################################################
    print('\n\nDESCRIBE: \n')
    display(df.describe())

    ######################################################
    # 散布図行列
    print('\n\nSCATTER PLOT MATRIX: \n')
    # from pandas.tools import plotting 
    from pandas import plotting
    plotting.scatter_matrix(df.iloc[:, 4:11], figsize=(8, 8)) 
    plt.show()

    ######################################################
    # 相関行列
    print('\n\nCORRELATION MATRIX: \n')
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10), dpi=400)
    correlation = df.corr()
    corr_columns = correlation.columns
    plt.imshow(correlation, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_columns)), corr_columns, rotation='vertical')
    plt.yticks(range(len(corr_columns)), corr_columns)
    plt.show()
    ######################################################
    print('\n\nVIOLINPLOT: \n')
    # violinplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.violinplot(df[violinplot_columns].values)
    # 横軸のラベルが入る場所を指定
    ax.set_xticks( list(  range(1, len(violinplot_columns)+1) ) )
    ax.set_xticklabels(violinplot_columns, rotation=90)
    plt.grid()
    plt.show()
    ######################################################

    unique_dict = dict()
    for col in df.columns:
        uni = list(set(list(df[f'{col}'])))
        num_of_unique = len(uni)
        if col not in unique_dict:
            unique_dict[col] = uni
            unique_dict[col+"_num"] = num_of_unique

    if print_unique:
        from pprint import pprint
        pprint(unique_dict,width=100,compact=True)
        # return unique_dict