## K-Nearest Neighbors

# Một số khái niệm cơ bản liên quan đến K-Nearest Neighbors.
- Thuật toán K-Nearest Neighbors là một trong những thuật toán cơ bản nhất trong Supervised learning ()
- Trong thuật toán K-means clustering, chúng ta không biết nhãn (label) của từng điểm dữ liệu. Mục đích là làm thể nào để phân dữ liệu thành các cụm (cluster) khác nhau sao cho dữ liệu trong cùng một cụm có tính chất giống nhau.
- Tóm tắt thuật toán:
 - Đầu vào: Dữ liệu X và số lượng cluster cần tìm K
 - Đầu ra: Các center M, và label vector cho từng điểm dữ liệu Y
   + B1: Chọn K điểm bất kỳ làm các centers ban đầu.
   + B2: Đặt mỗi điểm dữ liệu vào cluster có center gần nó nhất.
   + B3: Nếu việc gán dữ liệu vào từng cluster ở bước 2 không thay đổi so với vòng lặp trước nó thì dừng thuật toán.
   + B4: Cập nhật center cho từng cluster bằng cách lấy trung bình cộng của tất các các điểm dữ liệu đã được gán vào cluster đó sau bước 2.
   + B5: Quay lại bước 2.
