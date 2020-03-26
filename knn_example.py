from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

n_samples = len(digits.images)

# 資料攤平:1797 x 8 x 8 -> 1797 x 64
# 這裏的-1代表自動計算，相當於 (n_samples, 64)
data = digits.images.reshape((n_samples, -1))

knn = KNeighborsClassifier()

knn.fit(data[:n_samples-100], digits.target[:n_samples-100])

print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(knn.predict(data[n_samples-100:]), digits.target[n_samples-100:]))