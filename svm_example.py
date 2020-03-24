import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

n_samples = len(digits.images)

# 資料攤平:1797 x 8 x 8 -> 1797 x 64
# 這裏的-1代表自動計算，相當於 (n_samples, 64)
data = digits.images.reshape((n_samples, -1))

# 產生SVC分類器
classifier = svm.SVC(gamma=0.002)

# 用前半部份的資料來訓練
classifier.fit(data[:n_samples // 4], digits.target[:n_samples // 4])

expected = digits.target[n_samples // 4:]

#利用後半部份的資料來測試分類器，共 899筆資料
predicted = classifier.predict(data[n_samples // 4:])

print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(expected, predicted))