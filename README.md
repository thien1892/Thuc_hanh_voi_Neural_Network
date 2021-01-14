# Mô hình NN nhiều lớp (X -> (linear --> relu) *L-1 --> sigmoid --> Y_hat):
- X có X.shape = (n_x, m), m là số mẫu, n_x là 1 vecto đặc trưng của x. Nếu x là 1 bức ảnh thì chuyển ảnh thành vecto qua RGB rồi trải thành 1 chiều, ví dụ ảnh 64x64 thì n_x = 64x64x3
- Y có Y.shape = (1, m)
- Để đào tạo mô hình cập nhật tham số, ta có các hàm sau
    1. Khởi tạo tham số:
        def initialize_parameters(n_x, n_h, n_y):
        ...
        return parameters
    ## Ta sử dụng vòng lặp tối ưu hàm mất mát để cập nhật tham số qua các hàm:
    2. Truyền xuôi:
        def linear_activation_forward(A_prev, W, b, activation):
        ...
        return A, cache

    3. Hàm mất mát:
        def compute_cost(AL, Y):
        ...
        return cost

    4.  Truyền ngược:
        def linear_activation_backward(dA, cache, activation):
        ...
        return dA_prev, dW, db

    5. Update tham số:
        def update_parameters(parameters, grads, learning_rate):
        ...
        return parameters
# Thực hành mô hình Neural Network trên một số nguồn dữ liệu thu thập được.

## Dữ liệu 1: Nhận dạng xem bức ảnh là mèo hay không?
Data để train và kiểm tra được lưu trữ ở datacat
Tôi đã thử qua các mô hình:
1. Mô hình logistic regression
2. Mô hình NN 2 lớp ẩn
3. Mô hình NN 3 lớp ẩn
4. Mô hình NN 4 lớp ẩn
Độ chính xác ở các tập train, test như sau:
<img src ='https://i.imgur.com/KJxWiqR.png'>

Với mô hình Neural Network, tôi đã cập nhật được tham số W,b qua các vòng lặp để tối ưu hóa hàm mất mát.
Nhưng với các siêu tham số như: số lớp ẩn, số nút ở mỗi lớp, hàm kích hoạt(relu? sigmoid?...), tỷ lệ học tập anpha,... những siêu tham số này nên được điều chỉnh ra sao để độ chính xác ngày cảng tăng lên???
:)) các khóa học sau ở deeplaerning sẽ gải đáp các câu hỏi trên!





