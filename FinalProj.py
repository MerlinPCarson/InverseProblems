import sys
import numpy as np
import numpy.ma as ma
from scipy.sparse import diags
import matplotlib.pyplot as plt
import hw3


def plot_data(data1, data2, title1=None, title2=None):
    plt.figure()
    plt.subplot(2,1,1)
    plt.title(title1)
    plt.imshow(data1, cmap='gray')
    plt.subplot(2,1,2)
    plt.title(title2)
    plt.imshow(data2, cmap='gray')
    plt.tight_layout()
    plt.show()

def plot_reg2(data1, data2, title1=None, title2=None):
    plt.figure()
    plt.title(title1)
    plt.imshow(data1, cmap='gray')
    plt.figure()
    plt.title(title2)
    plt.imshow(data2, cmap='gray')
    plt.show()

def plot_reg(data, title=None):
    plt.title(title)
    plt.imshow(data, cmap='gray')
    plt.tight_layout()
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

    n, m = Dhat.shape

    # view data
    plot_data(Dhat, mask, title1="Data", title2="Mask")

    # Diffusion Matrix 
    s = 0.45
    B = diags([s, 1-2*s, s], [-1, 0, 1], shape = (n,n)).toarray()

    # Blurring Operator
    blur_op_power = 10
    A = np.linalg.matrix_power(B,blur_op_power)  # diffusion matrix B^k

    # Regularize
    method = 'TSVD'
    print(f'Regularizing with {method}')
    k=79
    XhatTSVD = hw3.regularize(A, Dhat, method, p=k, mask=mask)
    plot_reg(XhatTSVD, title=f"TSVD with k={k}")

    # Regularize
    method = 'TK-gen'
    print(f'Regularizing with {method}')
    Lambda = 0.003
    XhatTK0 = hw3.regularize(A, Dhat, method, Lop=0, Lambda=Lambda, mask=mask)
    plot_reg(XhatTK0, title=f"Tikhonov-General $L_0$ with λ={Lambda}")

    # Regularize
    method = 'TK-gen'
    print(f'Regularizing with {method}')
    Lambda = 0.005
    XhatTK1 = hw3.regularize(A, Dhat, method, Lop=1, Lambda=Lambda, mask=mask)
    plot_reg(XhatTK1, title=f"Tikhonov-General $L_1$ with λ={Lambda}")

    # Regularize
    method = 'TK-gen'
    print(f'Regularizing with {method}')
    Lambda = 0.002
    XhatTK2 = hw3.regularize(A, Dhat, method, Lop=2, Lambda=Lambda, mask=mask)
    plot_reg(XhatTK2, title=f"Tikhonov-General $L_2$ with λ={Lambda}")

    #plot_reg2(XhatTK2a, XhatTK2b, title1="Tikhonov-General $L_{1}$", title2="Tikhonov-General $L_{2}$")
    #plot_reg2(XhatTK0, XhatTK1, title1="Tikhonov-General $L_{0}$", title2="Tikhonov-General $L_{1}$")
    #plot_reg2(XhatTSVD, XhatTK, title1="TSVD", title2="Tikhonov-General")
    
    return 0 



if __name__ == '__main__':
    sys.exit(main())
