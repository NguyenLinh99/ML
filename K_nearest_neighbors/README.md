# K-Nearest Neighbors

## Một số khái niệm cơ bản liên quan đến K-Nearest Neighbors.
- Thuật toán K-Nearest Neighbors (KNN) là một trong những thuật toán cơ bản nhất trong Supervised learning (học có giám sát)
- KNN là thuật toán đi tìm đầu ra của một điểm dữ liệu mới bằng cách chỉ dựa trên thông tin của K điểm dữ liệu trong training set gần nó nhất - lazy learning.
- Tóm tắt thuật toán:
   + B1: Xác định tham số K là số láng giềng gần nhất.
   + B2: Tính khoảng cách các dữ liệu mới và các dữ liệu trong training data. 
   + B3: Chọn K láng giềng gần nhất (K khoảng cách nhỏ nhất).
   + B4: Đếm số điểm dữ liệu của mỗi K láng giềng đã xác định ở B3.
   + B5: Lớp của dữ liệu mới dựa vào phần lớn lớp của K.

## Implement.
- Sử dụng KNN để phân loại hoa với bộ dữ liệu Iris (Iris flower dataset). Bộ dữ liệu này bao gồm thông tin của ba loại hoa Iris khác nhau: Iris setosa, Iris virginica và Iris versicolor. 
- Sử dụng scipy.spatial.distance để tính khoảng cách. 
- Sử dụng 3 thuật toán tính khoảng cách để so sánh: euclidean, manhattan, cosine
 - Công thức tính khoảng cách euclidean (còn được gọi là l2_distance):
	dis = sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
 - Công thức tính khoảng cách manhattan (còn được gọi là l1_distance):
	dis = abs(x_1 - x_2) + abs(y_1 - y_2)
 - Công thức tính khoảng cách cosine (khoảng cách cosine có lợi khi dùng cho các vector có hướng):
	dis = 1- cos(alpha) = 1- dot(A, B) / (len(A)*len(B))
- Thử với các K khác nhau 
- So sánh khi sử dụng knn của thư viện sklearn
- Run file knn.py 
	python3 knn.py

