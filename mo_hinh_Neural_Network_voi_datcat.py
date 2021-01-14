# Import cac goi du lieu
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



np.random.seed(1)

# Load du lieu
path_train = 'datacat/train_catvnoncat.h5'
path_test = 'datacat/test_catvnoncat.h5'
train_X, train_Y, test_X, test_Y, classes = load_data(path_train, path_test)

# chuyen train_X, test_X sang dang (nx, m = so mau du lieu) va chuan hoa bang cach /255
train_X = train_X.reshape(train_X.shape[0], -1).T /255
test_X = test_X.reshape(test_X.shape[0], -1).T /255


# Mo hinh 2 lop an bao gom X -> (Linear -> Relu) * (L-1) -> linear -> Singmoid -> Y_hat

def mo_hinh_NN(X, Y, lop_an, learning_rate = 0.005, num_iterations = 4000, print_cost = False):
    np.random.seed(1)
    costs = []
    ## Khoi tao tham so
    parameters = initialize_parameters_deep(lop_an)

    # Vong lap toi uu hoa
    for i in range(num_iterations):
        # Truyen xuoi:
        Y_hat, caches = L_model_forward(X, parameters)
        # Tinh ham mat mat:
        cost = compute_cost(Y_hat, Y)
        # Truyen nguoc
        grads = L_model_backward(Y_hat, Y, caches)
        # Cap nhat tham so:
        parameters = update_parameters(parameters, grads, learning_rate)

        # In ham mat mat:
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        
        plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = mo_hinh_NN(train_X, train_Y, [12288,10,1], learning_rate = 0.005, num_iterations = 1000, print_cost = True)
pred_train = predict(train_X, train_Y, parameters)
pred_test = predict(test_X, test_Y, parameters)

