#ランダムフォレストだけ
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import confusion_matrix
import random, re
from sklearn.grid_search import GridSearchCV


ava = 0
count = 0

# データの読み込み --- (※1)
csv = pd.read_csv("Seeds.csv")
csv_data = csv[["area","perimeter","compactness","length","width","asymmetry","length groove"]]
csv_data = csv[["length","width","length groove"]]
csv_label = csv["name"]


for x in range(10):
    data_train, data_test, label_train, label_test = \
    cross_validation.train_test_split(csv_data, csv_label)

    #パラメータ設定
    #params = [
    #{"C": [1,10,100,1000], "kernel":["linear"]},
    #{"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
#]
    # データの学習・グリッドサーチ --- (※4)
    clf = RandomForestClassifier()
    #clf = GridSearchCV( RandomForestClassifier(), params, n_jobs=-1 )
    clf.fit(data_train, label_train)
    #print("学習器=", clf.best_estimator_)

    # データを予測 --- (※5)
    predict = clf.predict(data_test)

    # 合っているか結果を確認 --- (※6)
    scores = cross_validation.cross_val_score(clf, csv_data, csv_label, cv=5)
    #print("各正解率=", scores)
    #print("クロス正解率=", scores.mean())
    ac_score = metrics.accuracy_score(label_test, predict)
    cl_report = metrics.classification_report(label_test, predict)
    co_matrix = confusion_matrix(label_test,  predict)
    ava += ac_score
    count += 1
    #print("（ランダムf）正解率=", ac_score)
    print("レポート=\n", cl_report)
    print("マトリックス=\n", co_matrix)
#
print("平均：", ava / count)
print("実行回数=",count)
