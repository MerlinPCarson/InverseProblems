import sys
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import hw3


def plot_data(data1, data2, title1=None, title2=None):
    plt.figure()
    plt.subplot(2,1,1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title1)
    plt.imshow(data1, cmap='gray')
    plt.subplot(2,1,2)
    plt.xticks([])
    plt.yticks([])
    plt.title(title2)
    plt.imshow(data2, cmap='gray')
    plt.tight_layout()
    plt.show()


def plot_reg(data, title=None):
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data, cmap='gray')
    plt.tight_layout()
    plt.show()


def main():

    Dhat = hw3.load_data_m('prdata.m')
    mask = hw3.load_data_m('mask.m')

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

    return 0 


if __name__ == '__main__':
    sys.exit(main())

