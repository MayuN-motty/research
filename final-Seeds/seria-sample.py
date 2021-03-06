#シリアライズしたモデル（pklファイル）を復元し、全データで予測する
from sklearn import datasets
from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics, cross_validation

import itertools
import numpy as np
import matplotlib.pyplot as plt

# データセットを再読み込み
csv = pd.read_csv("Seeds.csv")
#csv_data = csv[["area","perimeter","compactness","length","width","asymmetry","length groove"]]
csv_data = csv[["area","perimeter","length","length groove"]]
csv_label = csv["name"]

#test_rate = 0.05
#data_train, data_test, label_train, label_test = \
#    cross_validation.train_test_split(csv_data, csv_label,test_size=test_rate)



# 予測モデルを復元
clf = joblib.load('clf-randomf.pkl') 
#clf = joblib.load('clf-svm.pkl') 

# 予測結果を出力
print(clf)
#clf.fit(data_train, label_train)
clf.fit(csv_data,csv_label)

print("\n+ ベストパラメータ（グリッドサーチで見つけた最適値）:\n")
print(clf.best_estimator_)

# データを予測
#predict = clf.predict(data_test)
#print("\n+ PREDICT:\n")
#print(predict)

#正解率
scores = cross_validation.cross_val_score(clf, csv_data, csv_label, cv=10)
print("\n+ クロスバリデーション結果:\n")
print("各正解率=", scores)
print("平均正解率=", scores.mean())

#ac_score = metrics.accuracy_score(label_test, predict)
#print("\n+ テストデータでの識別結果:\n")
#print(ac_score)

##全データで予測##
print("\n+ 全データ予測結果:\n")
predict = clf.predict(csv_data)
print(predict)
ac_score = metrics.accuracy_score(csv_label, predict)
print(ac_score)

#コンフュージョンマトリックスseria用
print("\n+ コンフュージョンマトリックス:\n")
class_names =  ["kama","rosa","canadian"]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
#	plt.imshow(cm, interpolation='nearest', cmap=cmap)
#	plt.title(title)
#	plt.colorbar()
	tick_marks = np.arange(len(classes))
#	plt.xticks(tick_marks, classes, rotation=45)
#	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
#	else:
#		print('Confusion matrix, without normalization')
		
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
cnf_matrix = confusion_matrix(csv_label, predict)
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

