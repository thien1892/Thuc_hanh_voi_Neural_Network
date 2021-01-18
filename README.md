# Thực hành mô hình Neural Network trên một số nguồn dữ liệu thu thập được.

## Dữ liệu 1: Nhận dạng xem bức ảnh là mèo hay không?
Data để train và kiểm tra được lưu trữ ở datacat.   
Tôi đã thử qua các mô hình:
1. Mô hình logistic regression
2. Mô hình NN 2 lớp ẩn
3. Mô hình NN 3 lớp ẩn
4. Mô hình NN 4 lớp ẩn
Độ chính xác ở các tập train, test như sau:
<img src ='https://i.imgur.com/KJxWiqR.png'>

Với mô hình Neural Network, tôi đã cập nhật được tham số W,b qua các vòng lặp để tối ưu hóa hàm mất mát.
Nhưng với các siêu tham số như: số lớp ẩn, số nút ở mỗi lớp, hàm kích hoạt(relu? sigmoid?...), tỷ lệ học tập anpha,... những siêu tham số này nên được điều chỉnh ra sao để độ chính xác ngày cảng tăng lên???

## Dữ liệu 2: Digis_dataset

Dữ liệu chữ số viết tay, lưu trữ ở thư mục datamnist, với tập train = 60.000 mẫu, tập test = 10.000 mẫu:  

<img src ='https://i.imgur.com/zfKnh5I.jpg'>
   
**Đầu tiên tôi thử với thuật toán phân cụm knn = 5**, cho kết quả train_accuracy = 98; test_accuracy = 96.88   
**Tiếp theo, Tôi thử với Deep neural network với 3 lớp ẩn, 2 lớp giữa có số node = 128, 64 , epochs = 5, batch_size = 32**, sau 1 thời gian ngắn, cho kết quả như sau: train_accuracy = 98.7; test_accuracy = 97.53








