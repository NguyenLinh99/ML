## Linear Regression

# Một số khái niệm cơ bản liên quan đến Linear Regression.
- Thuật toán Linear Regression giải quyết các bài toán có đầu ra là giá trị thực, ví dụ: dự đoán giá nhà, dự đoán giá cổ phiếu, dự đoán tuổi,...

- Nhìn ở phương diện toán học, bài toán này được viết dưới dạng sau:
	y = x0*w0+x1*w1+...xn*wn+b = X*w+b
  Trong đó, y được hiểu như một hàm số tuyến tính theo X, với các tham số cần tìm là w, b.

- Để hiểu rõ hơn về cách tính hàm loss, gradient, các bạn có thể tham khảo đường link sau: https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

- Một số lưu ý khi làm việc với ma trận:
  - X.T được gọi là ma trận chuyển vị của ma trận X. 
  - Đạo hàm của Ax=[a1x a2x … anx] với A là một vector có dạng: ∇x(Ax)=[a1.T a.T2 … an.T]=AT
  - Tham khảo thêm về cách tính đạo hàm các biến: 
	https://machinelearningcoban.com/math/#luu-y-ve-ky-hieu
	https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf

# Implement.
- Data: Data có dạng file txt, chứa các chỉ số liên quan đến bệnh tuyến tiền liệt (lpsa). Các chỉ số được cho: lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45
- Yêu cầu bài toán: Từ dữ liệu đã cho, đi tìm các tham số W, b để có thể xác định được chỉ số liên quan đến bệnh tuyến tiền liệt.
- Chia tập train, test theo tỉ lệ 7:3
- Run file linear_regression.py
	python3 linear_regression.py
