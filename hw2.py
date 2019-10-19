#!/usr/bin/env python3
# coding: utf-8

###################################################
# Script for de-blurring an image                 #
# uses TSVD and Tikonov regularization            #
#                                                 #
# Author: Merlin Carson                           #
# Date: Oct-16-2019                               #
###################################################

import sys
import cv2
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


# display an image matrix on screen
def show_img(img, title=None):
   
    print(f'Image dimensions are {img.shape}')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.1)


# create diffusion matrix n x m
def diffusion_matrix(L, n, m):

    # make matrix tridiagonal 
    # 1-2*L as diagonal and L above and below diagonal
    return diags([L, 1-2*L, L], [-1, 0, 1], shape = (n,m)).toarray()


def tsvd_index(A, Dhat):
    pass

def tk_lambda(A, Dhat):
    pass


def solve(U, S, V, dhat, k):
    #print(f'S{S.shape[0]}')
    xhat = np.zeros((128))
    #print(f'pre-xhat {xhat.shape}')
    #print(f'xtrunc {xtrunc.shape}')
    #print(f'U{U.shape}, V{V.shape}')
    for i in range(k):
        #print(f'xtrunc {xtrunc.shape}')
        xi = (np.dot(U[:,i].T, dhat)/S[i]) * V[:,i]
        xhat = np.add(xhat, xi)
        #print(xhat)
        #print(f'xtrunc {xtrunc.shape}')

    #print(f'xtrunc {xtrunc.shape}')
    #print(xtrunc)
    #print(f'post-xhat {xhat.shape}')

    return np.expand_dims(xhat, axis=1) 

    
def regularize(A, Dhat, method, pmax):
    
    U, S, Vt = np.linalg.svd(A)
    m = A.shape[0]               # number of rows
    print(f'U{U.shape}, S{S.shape}, V{Vt.shape}')

    for p in range(pmax):
        Xhat = np.empty((m,0))
        for i in range(m):
            dhat = Dhat[:,i]
            xhat = solve(U, S, Vt.T, dhat, p) 

            Xhat = np.hstack((Xhat, xhat)) 
            #print(f'Xhat {Xhat.shape}')

        print(f'Xhat: {Xhat.shape}')
        show_img(Xhat, title=f'k={p}')

    return 0


def de_blur(A, Dhat, method):
    if method is 'TSVD':
        print('Regularizing with Truncated Singular Value Decomposition')
        p = A.shape[0] 
    elif method is 'TK':
        print('Regularizing with Tikhonov Regularization')
        p = tk_lambda(A, Dhat)
    else:
        sys.exit('Unknown method') 

    return regularize(A, Dhat, method, p)


def main():

    # open image
    imgFile = 'hw2blur.jpg'
    Dhat = cv2.imread(imgFile,0)

    # display image
    show_img(Dhat, 'Original Picture')

    n, m = Dhat.shape       # dims of image
    L = 0.45                # diagonal val for blurring matrix
    blur_op_power = 10      # power to raise the blurring operator matrix by
   
    # create blurring matrix 
    B = diffusion_matrix(L, n, n)
    A = np.linalg.matrix_power(B,blur_op_power)  # diffusion matrix B^k
  
    # regularization 
    method = 'TSVD'
    Xhat = de_blur(A, Dhat, method)

    # regularization 
    #method = 'TK'
    #Xhat = de_blur(A, Dhat, method)

    return 0


if __name__ == '__main__':
    sys.exit(main())

