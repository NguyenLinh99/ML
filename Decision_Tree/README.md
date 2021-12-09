# Decision Tree - Cây Quyết Định

## Một số khái niệm cơ bản liên quan đến Decision Tree.
- Decision Tree là một thuật toán học máy có thể thực hiện được cả 2 tác vu phân loại (classify) và hồi quy (regression)
- Cây quyết định cho bài toán phân loại hoa Iris theo sklearn
![alt tag](https://github.com/linhtinhlinhta/ML/blob/master/Decision_Tree/iris_tree.png)
- Ý tưởng cây Quyết định phân loại:
    Giả sử ta cần phân loại một bông hoa Iris. Hãy bắt đầu từ nút gốc (ở trên cùng). Nút này kiểm tra liệu chiều dài cánh hoa (pental length) <= 2.45 hay không?
    - Nếu TRUE, đi xuống nút con trái của nút gốc, trong TH này thì nó là một nút lá (không có nút con). Cây quyết định dự đoán nó thuộc lớp sentosa (class = sentosa)
    - Ngược lại, đi xuống nút con phải của nút gốc, kiểm tra thêm điều kiện độ rộng cánh hoa (pental width) <= 1.75 hay không?
        + Nếu nhỏ hơn, dự đoán bông hoa thuộc loại versicolor
        + Ngược lại, dự đoán bông hoa thuộc loại verginica

- Đặc điểm: 
    - Không đòi hỏi phải chuẩn bị dữ liệu nhiều, không cần phải co giãn hay căn giữa các giá trị đặc trưng.
    - Scikit-Learn sử dụng thuật toán CART - chỉ trả về cây nhị phân, tức là các nút không phải nút lá chỉ có 2 nút con (câu trả lời là có hoặc không) 
    - Thuật toán ID3 có thể tạo ra các cây quyết định trả về nhiều nút con hơn. Xem chi tiết thuật toán ID3 tại đây: https://machinelearningcoban.com/2018/01/14/id3/

- Thuật toán huấn luyện CART (Classification and Regression tree):
    - Chia tập huấn luyện thành hai tập con theo đặc trưng k và ngưỡng tk (pental length - k < 2.5 - tk)
    - Cách chọn k và tk?
        + Tìm kiếm cặp (k, tk) tạo ra tập con thuần khiết nhất
        + Tiếp tục chia nhỏ các tập con với cùng logic, rồi đến các tập con nhỏ hơn, cứ đệ quy như vậy.
        + Thuật toán dừng lại khi đạt độ sâu tối đa (max_depth) hoặc không tìm được cách chia để giảm độ pha tạp. 
    - Độ pha tạp Entropy: phép đo độ pha tạp, Entropy của một tập bằng 0 khi tập đó chỉ chứa các mẫu thuộc 1 lớp. 