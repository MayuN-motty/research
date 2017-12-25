import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
#import my_func as mf # メソッドを定義した自作スクリプトをインポート

#+ テストデータでの識別結果を0.3に設定すると+ テストデータでの識別結果:0.952380952381
test_rate = 0.05 # 識別率計算用に、訓練データのうちこの割合をテストデータに。
n_cv = 5 # クロスバリデーションの数


## ここからメインの処理開始 ##

# データの読み込み
csv = pd.read_csv("Seeds.csv")
#csv_data = csv[["area","perimeter","compactness","length","width","asymmetry","length groove"]]
#csv_data = csv[["area","perimeter","length","width","length groove"]]
csv_data = csv[["area","perimeter","length","length groove"]]

csv_label = csv["name"]

#data = pd.read_csv(data_file, header=None)
#label = pd.read_csv(label_data, header=None)




# 特徴量として使用する列（前処理後のデータを使用）のみを抽出
#data = processed_train
#used = data.columns
#data = data.loc[:,used]
#print(data.head(5))
#print(data.columns)

## 学習用とテスト用データに分ける ##
data_train, data_test, label_train, label_test = \
    cross_validation.train_test_split(csv_data, csv_label,test_size=test_rate)

print(len(data_train))
print(len(data_test))

## データを学習 ##
#mod = svm.SVC()
mod = RandomForestClassifier()

## グリッドサーチ（ハイパーパラメータ探索）##
# n_estimators:決定木の数。[5,10,100,300,400,500]でグリッドサーチの結果、決定木の数は300に設定
# max_depth:決定木の深さの最大値.。過学習を避けるためにはこれを調節するのが重要．[5,10,30,100,200,300]の結果、10から結果が変わらなくなったので、10に設定
# random_state:乱数のタネの指定．何かしらの整数を指定

parameters = {
	'n_estimators' : [300],
	'max_depth'    : [10],
    'random_state' : [0]
}

#SVM用グリッドサーチ
#C=100
#parameters = [
#    {"C": [1,10,100,1000], "kernel":["linear"]},
#   	{"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
#]


# グリッドサーチを実行
clf = GridSearchCV(mod, parameters,cv=n_cv,n_jobs=-1)



print(clf)
clf.fit(data_train, label_train)

# 結果を表示
print("\n+ ベストパラメータ（グリッドサーチで見つけた最適値）:\n")
print(clf.best_estimator_) # クロスバリデーションの結果、最も精度の高かったパラメータの値

print("\n+ トレーニングデータでCVした時の平均スコア:\n")
for params, mean_score, all_scores in clf.grid_scores_:
    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

# データを予測
# Call predict on the estimator with the best found parameters.
predict = clf.predict(data_test)

# 予測モデルをシリアライズ
joblib.dump(clf, 'clf2-randomf.pkl')

# 正解率を求める
scores = cross_validation.cross_val_score(clf, csv_data, csv_label, cv=5)
print("\n+ クロスバリデーション結果:\n")
print("各正解率=", scores)
print("平均正解率=", scores.mean())
ac_score = metrics.accuracy_score(label_test, predict)
print("\n+ テストデータでの識別結果:\n")
print(ac_score)

## コンフュージョンマトリックスを計算 ##
print("\n+ コンフュージョンマトリックス:\n")
class_names =  ["kama","rosa","canadian"]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
		
	print(cm)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
#	plt.tight_layout()
#	plt.ylabel('True label')
#	plt.xlabel('Predicted label')

#
cnf_matrix = confusion_matrix(label_test, predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
	
# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix,  classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

#plt.show()

# Feature_importances
print("\n+ 各特徴量の重要度:\n")
fti = clf.best_estimator_.feature_importances_
for i, feat in enumerate(csv_data.columns):
	print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
