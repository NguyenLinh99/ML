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
        if self.metric not in _metrics.keys():
            self.metric = "euclidean"
        func = getattr(self, self._metrics[self.metric]) # goi ham tinh khoang cach
        dist = func(X_new)
        dist = np.argsort(dist, axis=1) # sap xep khoang cach
        k_nearest = dist[:, :self.K]



if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    # print(len(iris_x), iris_y)
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)
