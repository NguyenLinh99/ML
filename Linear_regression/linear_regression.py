import numpy as np 
from sklearn.model_selection import train_test_split

class LinearRegression:
    
    def __init__(self, epoch, alpha):
        self.epoch = epoch
        self.alpha = alpha
        self.w = None
        self.b = None
    
    def mse_loss(self, X, y_hat, y):
        n = y.shape[0]
        loss = np.sum((y_hat-y)**2)/(2*n)
        return loss
    
    def gradient_w(self, X, y_hat, y):
        n = X.shape[0]
        denta_w = 1/n * np.dot(X.T, y_hat-y)
        return denta_w
    
    def gradient_b(self, y_hat, y):
        n = y.shape[0]
        denta_b = 1/n * np.sum(y_hat-y)
        return denta_b

    def update_param(self, denta_w, denta_b):
        self.w -= self.alpha*denta_w
        self.b -= self.alpha*denta_b
    
    def train(self, X, y):
        self.w = np.random.normal(size=(X.shape[1], 1))
        self.b = np.mean(y)
        for i in range(self.epoch):
            y_hat = np.dot(X,self.w)+self.b
            loss = self.mse_loss(X, y, y_hat)
            print("Loss at epoch {}: {}".format(i, loss))
            #update params of weight
            denta_w = self.gradient_w(X, y_hat, y)
            denta_b = self.gradient_b(y_hat, y)
            self.update_param(denta_w, denta_b)
            if np.linalg.norm(denta_w, 2) < 1e-6:
                break
    
    def predict(self, X_test):
        y_hat = np.dot(X_test,self.w)+self.b
        return y_hat

    def r2_score(self, y_hat, y):
        total_sum_squares = np.sum((y - np.mean(y))**2)
        residual_sum_squares = np.sum((y - y_hat)**2)
        return 1 - residual_sum_squares/total_sum_squares


def standart_value(X, y):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return (X-X_mean)/X_std, (y-y_mean)/y_std

def main():
    X = np.loadtxt('data.txt', skiprows=1)
    # columns = ['lcavol', 'lweight',	'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    y = X[:, -1]
    X = X[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = standart_value(X_train, y_train)
    y_train = y_train.reshape((-1, 1))

    alpha = 0.01
    epochs = 500
    lambda_ = 0
    linear_regression = LinearRegression(epochs, alpha)
    linear_regression.train(X_train, y_train)

    X_test, y_test = standart_value(X_test, y_test)
    pred = linear_regression.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Test score: %f" % linear_regression.r2_score(pred, y_test))

if __name__ == '__main__':
    main()

