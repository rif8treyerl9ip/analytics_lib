"""
分析に使用するモデルのためのスクリプト。
主に回帰分析、ロジスティック分析
"""

"""
回帰分析のためのスクリプト。忘れがちな標準化や切片、係数をまとめるなど。
"""


############################ 回帰分析 ############################

import copy
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sqlalchemy import column
import pprint

def all_linear_regression(df):
    """
    ある変数をある変数以外のほかのすべての変数を使用して回帰分析。
    int,floatのすべての列に対してターゲットと説明変数のペアを作り、全てのペアに対して線形回帰を行う。

    必要な前処理は
    標準化、カテゴリ変数のエンコーディング。カテゴリ変数が多いと結果が見づらくなり無駄な部分が増えるので注意。

    """

    tmp = df.select_dtypes(
        include=['int8','int16','int32','int64',
            'uint8','uint16','uint32','uint64',
            'float16','float32','float64']
    )

    columns = list(tmp.columns)
    mean_squared_error_tr_list = []; mean_squared_error_te_list = []
    r2_tr_list = []; r2_te_list = []
    linear_model_list = []
    linear_model_coef_dict = {}

    for col in columns:
        tmpcolumns = copy.copy(columns)
        tmpcolumns.remove(f"{col}")
        
        train = tmp[tmpcolumns].values
        train_y = tmp[f'{col}'].values

        X_train, X_test, y_train, y_test = train_test_split(
        train, train_y, test_size=0.3, random_state=0)
        
        linear_model = sklearn.linear_model.LinearRegression()
        linear_model.fit(train, train_y)
        
        linear_model_list.append(linear_model)
        
        linear_model_coef = list(zip(tmpcolumns, linear_model.coef_))
        linear_model_coef.append(('intercept', linear_model.intercept_))
        linear_model_coef_dict[col] = linear_model_coef
    
        y_train_pred = linear_model.predict(X_train)
        y_test_pred = linear_model.predict(X_test)
        
        # create scoredf
        mean_squared_error_tr = mean_squared_error(y_train, y_train_pred)
        mean_squared_error_te = mean_squared_error(y_test, y_test_pred)
        r2_tr = r2_score(y_train, y_train_pred)
        r2_te = r2_score(y_test, y_test_pred)
        mean_squared_error_tr_list.append(mean_squared_error_tr)
        mean_squared_error_te_list.append(mean_squared_error_te)
        r2_tr_list.append(r2_tr)
        r2_te_list.append(r2_te)

    scoredf = pd.DataFrame([mean_squared_error_tr_list,
                mean_squared_error_te_list,
                r2_tr_list,
                r2_te_list,linear_model_list],
                index=[['mean_squared_error_tr','mean_squared_error_te','r2_tr','r2_te','model']],
                columns=columns)
    # pprint.pprint(linear_model_coef_dict)
    # pprint.pprint(linear_model_coef)
    
    return scoredf,linear_model_coef_dict
    

def check_coef(column_names, coef_list, intercept=None,only_print=True):
    """回帰係数と切片の確認"""
    weights = dict(zip(column_names, coef_list))
    if intercept:
      weights['intercept'] = intercept
    df = pd.DataFrame.from_dict(weights, orient='index')
    df.columns = ['coef']
    df.sort_values(by='coef', key=lambda t:abs(t), inplace=True, ascending=False)
    
def check_coef(column_names, coef_list, intercept=None,only_print=True):
    if only_print:
        print(df.head(10))
    else:
        return pd.DataFrame(df)

############################ 回帰分析 ############################

# import sklearn.linear_model
# linear_model = sklearn.linear_model.LinearRegression()
# linear_model.fit(df, label)

# linear_model_coef = list(zip(df.columns, linear_model.coef_))
# linear_model_coef.append(('intercept', linear_model.intercept_))
# linear_model_coef



def check_coef(column_names, coef_list, intercept=None,only_print=True):
    """回帰係数と切片の確認"""
    weights = dict(zip(column_names, coef_list))
    if intercept:
      weights['intercept'] = intercept
    df = pd.DataFrame.from_dict(weights, orient='index')
    df.columns = ['coef']
    df.sort_values(by='coef', key=lambda t:abs(t), inplace=True, ascending=False)
    
def check_coef(column_names, coef_list, intercept=None,only_print=True):
    if only_print:
        print(df.head(10))
    else:
        return pd.DataFrame(df)



############################ OLSによる分析 ############################

def ols_summary_and_screening_by_pvalue(df,labels):
    """
    線形回帰は外れ値に弱いのでstatsmodels.api.OLSでp値と信頼区間を出力しp値が0.05より小さい特徴量を使用する。
    """
    import statsmodels.api
    ols_df = standardization_dropped_df.copy()
    # olsでは定数項を自動予測しないので加える必要がある。
    ols_df['const'] = 1
    ols_model = statsmodels.api.OLS(attrition_label, ols_df)
    fit_ols_results = ols_model.fit()
    fit_ols_summary = fit_ols_results.summary2()

    sorted_ols_coef = fit_ols_summary.tables[1].sort_values(by='Coef.', key=lambda t:abs(t), ascending=False)
    sorted_ols_coef = sorted_ols_coef[sorted_ols_coef['P>|t|'] < 0.05]
    
    # print(fit_ols_summary)
    # print(sorted_ols_coef[:10])

    return fit_ols_summary,sorted_ols_coef





import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

def logistic_to_confusion_matrix_and_rocauc(df):
    """
    以下のタスクを実行します。
    - ロジスティック回帰
    - 混同行列
    - ROCAUC


    ROC
    x: 偽陽性率
    y: 真陽性率
    - Acuuracy:(TP+TN)/(TP+FP+FN+TN)
    - クラスに偏りがある場合、機能しない
    - False Positive Rate(偽陽性率）：FP/(FP+TN)
        - 正解データ負に大して正と予測した割合
    - True Positive Rate(真陽性率）：TP/(TP+FN)
        - 正解データ正に大して負と予測した割合

    """
    train_target = df['Legendary']
    train = df.select_dtypes(
            include=['int8','int16','int32','int64',
                'uint8','uint16','uint32','uint64',
                'float16','float32','float64']
        )


    X_train, X_test, y_train, y_test = train_test_split(
    train, train_target, test_size=0.3, random_state=0)

    # LogisticRegression
    clf = LogisticRegression(solver='lbfgs', max_iter=10000)
    clf.fit(X_train, y_train)

    print("正解率(train): ", clf.score(X_train,y_train))
    print("正解率(test): ", clf.score(X_test,y_test))

    pred_y_test = clf.predict(X_test)
    pred_y_test_prob = clf.predict_proba(X_test)
    pred_y_test_prob_True = pred_y_test_prob[:,1]

    confusion_matrix = pd.DataFrame(confusion_matrix(pred_y_test, y_test ,labels=['False', 'True']),
                index=['predicted 0', 'predicted 1'], columns=['real 0', 'real 1'])

   
    # ROCAUC
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
    # FPR（偽陽性率）,TPR（真陽性率）,閾値
    fpr, tpr, thresholds = roc_curve(y_test, pred_y_test_prob_True)
    auc_score = roc_auc_score(y_test, pred_y_test_prob_True)

    plt.plot(fpr, tpr, label='AUC = %.3f' % (auc_score),marker='o',markersize=3)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)






def standardization(df):
    """
    線形回帰、ロジスティック回帰の場合
    スケールが小さいカラムの係数が過剰に大きく出るので回帰分析の前にスケーリングを忘れずに
    """
    import sklearn.preprocessing 
    scaler = sklearn.preprocessing.MinMaxScaler()
    standardization_df = pd.DataFrame(scaler.fit_transform(df),
                                      index=df.index,
                                      columns=df.columns)
    return standardization_df







################################ memo ################################
# """多重共線性の時はonehotのうち片方を落とす"""
# pd.get_dummies(categorical_df, drop_first=True)