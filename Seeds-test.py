import pandas as pd
from sklearn import svm, metrics, cross_validation
#from sklearn.model_selection import train_test_split
#from sklearn import model_selection
score = 0
count = 0


csv = pd.read_csv('Seeds.csv')

#csv_data = csv[["area","perimeter","compactness","length","width","asymmetry","length groove"]]
#csv_data = csv[["area","perimeter","length","width","asymmetry","length groove"]]
csv_data = csv[["compactness","length groove"]]
csv_label = csv["name"]


for x in range(100):
    train_data, test_data, train_label, test_label = \
    cross_validation.train_test_split(csv_data, csv_label)

    clf = svm.SVC()
    clf.fit(train_data, train_label)
    pre = clf.predict(test_data)

    ac_score = metrics.accuracy_score(test_label, pre)
    score += ac_score
    count += 1
    print("正解率=", ac_score)

#実行数
print("平均：", score / count)
print("実行回数=",count)
