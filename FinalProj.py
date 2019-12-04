import sys
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import hw3


def plot_data(data):
    plt.imshow(data, cmap='gray')
    plt.show()


def load_data_m(dataFile):
    Dhat = []
    with open(dataFile, 'r') as data:
        for row in data.readlines():
            Dhat.append([float(val) for val in row.split()])

    return np.array(Dhat)


def main():

    Dhat = load_data_m('prdata.m')
    mask = load_data_m('mask.m')
    print(Dhat.shape)

    n, m = Dhat.shape

    print(Dhat[:1])
    print(Dhat[:0])
    dhat = Dhat[:0]
    print(dhat)
    dhat = dhat[dhat>0.0]
    print(dhat)
    # view data
    plot_data(Dhat)
    #plot_data(mask)

    # Diffusion Matrix 
    s = 0.45
    B = diags([s, 1-2*s, s], [-1, 0, 1], shape = (n,n)).toarray()

    # Blurring Operator
    blur_op_power = 10
    A = np.linalg.matrix_power(B,blur_op_power)  # diffusion matrix B^k
    print(A.shape)

    # Regularize
    method = 'TV'
    alpha = 0.05
    beta = 0.00001
    Xhat = hw3.regularize(A, Dhat, method, alpha=alpha, beta=beta)
    plot_data(Xhat)

    
     



if __name__ == '__main__':
    sys.exit(main())
