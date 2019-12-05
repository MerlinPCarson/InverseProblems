import sys
import numpy as np
import numpy.ma as ma
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

    # view data
    #plot_data(Dhat)
    #plot_data(mask)

    #print(Dhat[:,0])
    #print(mask[:,0])
    #indices = np.where(Dhat[:,0]>0.0)
    #print(indices)
    #print(Dhat[indices,0].shape)

    
    #return 0

    # Diffusion Matrix 
    s = 0.45
    B = diags([s, 1-2*s, s], [-1, 0, 1], shape = (n,n)).toarray()

    # Blurring Operator
    blur_op_power = 10
    A = np.linalg.matrix_power(B,blur_op_power)  # diffusion matrix B^k
    print(A.shape)

    method = 'TSVD'
    k=85
    Xhat = hw3.regularize(A, Dhat, method, p=k)
    plot_data(Xhat)

    # Regularize
    Lambda = 0.02
    method = 'TK-gen'
    Xhat = hw3.regularize(A, Dhat, method, Lop=0, Lambda=Lambda)
    plot_data(Xhat)

    
     



if __name__ == '__main__':
    sys.exit(main())
