"""
分析に使用するモデルのためのスクリプト。
決定木による可視化
分析結果がどうなっているかはわかりやすいが、一度の多くの変数を見ることができない

"""


"""
決定木による分析

多重共線性をそこまで気にしなくてよいのでone hot encodingでdrop=Trueにしなくてよい
特徴量の正規化が必要ない、むしろ正規化すると人の目で洞察するのが難しい。
人の目でみるならmax_depth=4,16ノードくらいが限界？

下記を順番にやっていき、箇条書き的にメモを書き留めておくのが良いか？
やるタスク
- sklearn.tree.plot_tree()による分析
- dtreeviz.trees import dtreevizによる分析

やらなくてもよさそうなタスク
- sklearn.ensemble.RandomForestClassifierのfeature importanceによる分析
理由：
多重共線性に弱く変数重要度が分散する
１つ１つの木をみて分析できない
以上の理由から、どうせ個々の木を見ることができないなら、GBDTに対してSHAPした方が示唆がある？

"""


############################ 決定木による分析 ############################

############################ sklearn.tree.DecisionTreeClassifier ############################

def DecisionTreeClassifier_plot(df):
    import sklearn.tree
    dt_model = sklearn.tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(df, attrition_label)

    plt.figure(figsize=(50, 10))
    sklearn.tree.plot_tree(dt_model, feature_names=df.columns ,filled=True)


"""
memo
dtreevizは更に見やすい
"""

def DecisionTreeClassifier_dtreeviz_plot(df,labels):
    import sklearn.tree
    dt_model = sklearn.tree.DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(df, attrition_label)
    from dtreeviz.trees import dtreeviz
    viz = dtreeviz(dt_model,
                df,
                attrition_label,
                target_name='Attrition',
                feature_names=df.columns,
                class_names=['No', 'Yes'])
    return viz      
DecisionTreeClassifier_dtreeviz_plot(df,attrition_label)


############################ sklearn.ensemble.RandomForestClassifier ############################

"""
estimators_属性を利用してn本目の木の意思決定プロセスを可視化できる。
rf_model.estimators_[0]
"""

rf_model.fit(df, attrition_label)
plt.figure(figsize=(10, 10), dpi=150)
sklearn.tree.plot_tree(rf_model.estimators_[0], feature_names=df.columns ,filled=True)
plt.show()


"""
shap値の可視化。説明変数に対するshap値を出してみてもよいかも
"""
import shap
rf_model = sklearn.ensemble.RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(converted_df, attrition_label)

shap.initjs()
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(converted_df)

plt.figure()
shap.summary_plot(shap_values, converted_df, show=False)
