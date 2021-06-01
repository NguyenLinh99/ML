import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

class KNN:
    _metrics = {'euclidean': '_l2_distance', 'manhattan': '_l1_distance', 'cosine': '_cosine_similarity'}

    def __init__(self, K, X, y, metric="euclidean"):
        self.K = K
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.metric = metric
    
    def _l2_distance(self, X_new):
        """
        _l2_distance - euclidean
        l2 = sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
        """
        return cdist(X_new, self.X, 'euclidean')
    
    def _l1_distance(self, X_new):
        """
        _l1_distance - manhattan
        l1 = abs(x_1 - x_2) + abs(y_1 - y_2)
        """
        return cdist(X_new, self.X, 'cityblock')
    
    def _cosine_similarity(self, X_new):
        """
        _cosine_similarity - cosine
        similarity = cos(alpha) = dot(A, B) / (len(A)*len(B))
        """
        return cdist(X_new, self.X, 'cosine')

    def predict(self, X_new):
        if self.metric not in self._metrics.keys():
            self.metric = "euclidean"
        func = getattr(self, self._metrics[self.metric]) # goi ham tinh khoang cach
        dist = func(X_new)
        dist = np.argsort(dist, axis=1) # sap xep khoang cach tu nho den lon
        k_nearest = dist[:, :self.K] # lay K khoang cach nho nhat
        labels = self.y[k_nearest]
        res = []
        for label in labels:
            label, count = np.unique(label, return_counts=True)
            label = label[np.argmax(count)]
            res.append(label)
        return res

def compare(X_train, y_train, X_test, y_test):
    K = [1, 3, 5, 7, 9]
    metrics = ['euclidean', 'manhattan', 'cosine']
    for metric in metrics:
        for k in K:
            knn = KNN(k, X_train, y_train, metric=metric)
            y_pred = knn.predict(X_test)
            acc = len(y_test[y_test==y_pred])/len(y_test)
            print("KNN with K = {} and metric = {} | Accuracy: {}".format(k, metric, acc))
        print("-"*50)


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    # print(len(iris_x), iris_y)
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)

    # so sanh knn giua cac metric va k
    is_compare = True
    if is_compare:
        compare(X_train, y_train, X_test, y_test)

    # su dung knn voi k=3, metric="euclidean"
    k = 3
    knn = KNN(k, X_train, y_train)
    pred = knn.predict(X_test)
    print("My KNN accuracy:", len(y_test[pred == y_test]) / len(y_test))
    # khi su dung thu vien sklearn
    sk_knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    sk_knn.fit(X_train, y_train)
    sk_pred = sk_knn.predict(X_test)
    print("Sklearn KNN accuracy:", len(y_test[sk_pred == y_test]) / len(y_test))