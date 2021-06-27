# PLA - Perceptron Learning Algorithm

## Một số khái niệm cơ bản liên quan đến PLA.
- Perceptron là một thuật toán phân loại (classification) cho trường hợp đơn giản nhất: chỉ có hai lớp (class).
- Bài toán Perceptron: Cho hai class được gán nhãn, tìm một boundary (đường biên giới- đường thẳng, mặt phẳng, siêu mặt phẳng (hyperplane) sao cho toàn bộ các điểm thuộc class 1 nằm về 1 phía, toàn bộ các điểm thuộc class 2 nằm về phía còn lại của boundary đó. Với giả định rằng tồn tại một boundary như thế.
- Ý tưởng thuật toán: ý tưởng cơ bản của PLA là xuất phát từ một nghiệm dự đoán nào đó, qua mỗi vòng lặp, nghiệm sẽ được cập nhật tới một ví trí tốt hơn. Việc cập nhật này dựa trên việc giảm giá trị của một hàm mất mát nào đó.
- Thuật toán:
  - X = [x1, x2, x3, ..., xn] là ma trận chứa các điểm dữ liệu
  - y = [y1, y2, y3, ..., yn] là ma trận chứa các nhãn của các điểm dữ liệu tương ứng, với yi=1 nếu thuộc class 1, yi=-1 nếu thuộc class 2.
  - Tại một thời điểm, giả sử, boundary cần tìm có phương trình:
                  f = w1x1 + w2x2 + w3x3 + ... + wnxn + w0
    - Ví dụ, trong không gian 2 chiều, boundary cần tìm là một đường thẳng như sau:
    ![alt text](PLA/pla.png)
    => Các điểm nằm về cùng 1 phía so với đường thẳng này sẽ làm cho hàm số f mang cùng dấu. Giả sử các điểm nằm trong nửa mặt phẳng nền xanh mang dấu dương (+), các điểm nằm trong nửa mặt phẳng nền đỏ mang dấu âm (-). Các dấu này cũng tương đương với nhãn y (+1, -1)của mỗi class. 
    - Nếu w là một nghiệm của bài toán Perceptron, với một điểm dữ liệu mới x chưa được gán nhãn, công thức xác định class của nó như sau:
                    label(x) = 1 nếu f=wX > 0, otherwise -1
    hay:
                    label(x) = sgn(f) = sgn(wX), sgn là hàm xác định dấu
  - Hàm mất mát:
    - Đếm số lượng các điểm bị misclassified (phân lớp lỗi):
                    L = ∑_(xi∈M)(-yisgn(wxi))
      Trong đó, M là các điểm bị misclassified. Với mỗi điểm xi∈M, yi và sgn(wxi) là ngược dấu nhau, do đó, -y1sgn(wxi)=1.
    - Hàm số này đạt GTNN nếu không có điểm nào bị misclassified. Tuy nhiên, do hàm số này là hàm rời rạc nên khó để tối ưu. Cần tìm một hàm mất mát khác để dễ tối ưu hơn.
    - Xét hàm sau: 
                    L = ∑_(xi∈M)(-yi(wxi))
      => Đạo hàm tương ứng: ∇wL = -yixi
         Quy tắc cập nhật: w = w + ηyixi (với η là learning rate được chọn bằng 1)
  - Tóm tắt thuật toán:
    - Chọn ngẫu nhiên hệ số w
    - Duyệt ngẫu nhiên qua từng điểm xi:
      B1: Nếu xi được phân loại đúng, không cần xử lý.
      B2: Nếu xi được phân loại sai, cập nhật w theo công thức:
                    w = w + yixi 
      B3: Tính toán số điểm bị phân lớp lỗi, nếu không còn điểm nào, dừng thuật toán. Nếu còn, quay lại B2.
    
## Implement.
- Tạo ngẫu nhiên X theo hàm np.random.multivariate_normal - hàm tạo ngẫu nhiên với giá trị trung bình và hiệp phương sai được định nghĩa trước.
- Tính toán theo các bước đã trình bày ở trên
- Run file pla.py
	                  python3 pla.py
- Visualize kết quả:
  ![alt text](PLA/pla_vis.gif)

## Chú ý.
- Chỉ thực hiện thuật toán PLA khi dữ liệu là linear separable, tức là các điểm của hai lớp không giao nhau. Dữ liệu trên thực tế thường hiếm khi linearly separable là một hạn chế của Perceptron, tuy nhiên Perceptron vẫn là nền tảng cho các thuật toán Neural Network hay Deep Learning sau này.
- PLA có thể cho vô số nghiệm khác nhau vì có vô số boundary phân chia 2 lớp dữ liệu.

## Tham khảo.
- https://machinelearningcoban.com/2017/01/21/perceptron/