import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

## tạo dữ liệu bằng cách lấy các điểm theo phân phối chuẩn có kỳ vọng tại các điểm có tọa độ (2, 2), (8, 3) và (3, 6)
# ma trận hiệp phương sai giống nhau và là ma trận đơn vị
# mỗi cluster có 500 điểm.
means = [[2,2],[8,3],[2,6]]
cov_matrix = [[1,0],[0,1]]
N=500
X0 = np.random.multivariate_normal(means[0], cov_matrix, N)
X1 = np.random.multivariate_normal(means[1], cov_matrix, N)
X2 = np.random.multivariate_normal(means[2], cov_matrix, N)
X = np.concatenate((X0,X1,X2), axis=0)#Gộp 3 ma trân X0, X1, X2 thành X theo chiều ngang
K = 3
original_label = np.asarray([0]*N + [1]*N + [2]*N)
# print(original_label) 

## Hàm vẽ 
def kmeans_visual(X, label, centers):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    for center in centers:
        plt.plot(center[0], center[1], 's', color="yellow", markersize = 10, alpha = .8)

    plt.axis('equal')
    plt.plot()
    # plt.show()
    plt.savefig("visual.png")
# kmeans_visual(X, original_label)

## Khởi tạo các centers ban đầu 
def kmeans_init_center(X, k):
    return X[np.random.choice(X.shape[0], k)]

## Gán nhãn mới cho các điểm khi biết các centers
def kmeans_assign_labels(X, centers):
    distance = cdist(X,centers)
    return np.argmin(distance, axis=1)

## Cập nhật các centers mới dựa trên dữ liệu vừa được gán nhãn
def kmeans_update_centers(X, labels, K):
    centers = np.ones((K, X.shape[1]))
    for k in range(K):
        X_k = X[labels==k, :]
        centers[k, :] = np.mean(X_k, axis=0)
    return centers

## Kiểm tra điều kiện dừng của thuật toán 
def is_converged(centers, new_centers):
    # return True nếu hai tập giống nhau 
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [kmeans_init_center(X, K)]
    print(centers)
    labels = []
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if is_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
    return centers, labels

centers, labels = kmeans(X, K)
print('Centers: ')
print(centers[-1])
kmeans_visual(X, labels[-1], centers[-1])
