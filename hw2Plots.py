#!/usr/bin/env python3
# coding: utf-8

###################################################
# Script for de-blurring an image                 #
# uses TSVD and Tikhonov regularization            #
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
def show_img(img, title=None, pause=True):
   
    print(f'Image dimensions are {img.shape}')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.imshow(img, cmap='gray')
    if pause:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)


def plot_singular_values(S, power):

    xvals = np.arange(1,len(S)+1)
    xticks = np.arange(0,len(S)+1, 8)
    # Log Plot of conditions
    plt.figure()
    plt.title('Singular Values of A=$B^{'+str(power)+'}$')
    plt.yscale('log', basey=10)
    plt.xlim((1,len(S)+1))
    plt.xticks(xticks)
    plt.xlabel('Index of Singular Value')
    plt.ylabel('Singular Value ($log_{10}$)')
    plt.grid()
    plt.plot(xvals, S)
    plt.show()


def plot_filter_factors(filterCoeffs, Lambda):

    xvals = np.arange(1,len(filterCoeffs)+1)
    xticks = np.arange(0,len(filterCoeffs)+1, 8)
    # Log Plot of conditions
    plt.figure()
    plt.title(f'Tikhonov Regularization Filter Coefficients λ={Lambda}')
    #plt.yscale('log', basey=10)
    plt.xlim((1,len(filterCoeffs)+1))
    plt.xticks(xticks)
    plt.xlabel('Index of Filter Value')
    plt.ylabel('Index of data column')
    plt.grid()
    plt.plot(xvals,filterCoeffs)
    #plt.colorbar()
    plt.show()


# create diffusion matrix n x m
def diffusion_matrix(L, n, m):

    # make matrix tridiagonal 
    # 1-2*L as diagonal and L above and below diagonal
    return diags([L, 1-2*L, L], [-1, 0, 1], shape = (n,m)).toarray()


def tsvd_index(A, Dhat):
    pass

def tk_lambda(A, Dhat):
    pass


def solve(U, S, V, dhat, p, method):
    fltr = 1.0
    filterCoeffs = []

    xhat = np.zeros((128))
    
    if method is 'TK':
        n = S.shape[0]  # use all singular values
    else:
        n = p           # truncate singular values

    for i in range(n):
        # determine filter factor based on method
        if method is 'TK':
            fltr = S[i]**2/(S[i]**2 + p**2) 
            filterCoeffs.append(fltr)

        xi = fltr * (np.dot(U[:,i].T, dhat)/S[i]) * V[:,i]
        xhat = np.add(xhat, xi)

    return np.expand_dims(xhat, axis=1), np.expand_dims(np.array(filterCoeffs), axis=1)

    
def regularize(A, Dhat, method, p):
   
    U, S, Vt = np.linalg.svd(A)
    m = A.shape[0]               # number of rows

    Xhat = np.empty((m,0))
    filterFactors = np.empty((m,0))

    for i in range(m):
        dhat = Dhat[:,i]

        xhat, filterCoeffs = solve(U, S, Vt.T, dhat, p, method) 

        Xhat = np.hstack((Xhat, xhat)) 

        if method is 'TK':
            filterFactors = np.hstack((filterFactors, filterCoeffs))

    return Xhat, S, filterFactors


def de_blur(A, Dhat, method):
    n = A.shape[0] 
    if method is 'TSVD':
        print('Regularizing with Truncated Singular Value Decomposition')
        for k in range(75,n):  # try all truncation values to find best fit
            Xhat, S, _ = regularize(A, Dhat, method, k)
            show_img(Xhat, title=f'Truncated Singular Valuse Decomposition with k={k}', pause=False)
    elif method is 'TK':
        print('Regularizing with Tikhonov Regularization')
        for Lambda in reversed(range(n)):  # try all truncation values to find best fit
            Xhat, S, _ = regularize(A, Dhat, method, Lambda)
            show_img(Xhat, title=f'Tikhonov Regularization with λ={Lambda}', pause=False)
    else:
        sys.exit('Unknown method') 

    #return regularize(A, Dhat, method, p)


def load_data_m(dataFile):
    Dhat = []
    with open(dataFile, 'r') as data:
        for row in data.readlines():
            Dhat.append([float(val) for val in row.split()])

    return np.array(Dhat)


def main():

    # open image
    imgFile = 'hw2blur.jpg'
    dataFile = 'hw2data.m'
    #Dhat = cv2.imread(imgFile,0)

    Dhat = load_data_m(dataFile) 

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
    k=100
    Xhat, S, _ = regularize(A, Dhat, method, k)
    show_img(Xhat, title=f'Truncated Singular Value Decomposition with k={k}', pause=True)
#    de_blur(A, Dhat, method)

    # regularization 
    method = 'TK'
    Lambda=0.000005
    Xhat, S, filterFactors = regularize(A, Dhat, method, Lambda)
    show_img(Xhat, title=f'Tikhonov Regularization with λ={Lambda}', pause=True)
    #de_blur(A, Dhat, method)

    print(filterFactors.shape)
    plot_singular_values(S, blur_op_power)
    plot_filter_factors(filterFactors, Lambda)

    return 0


if __name__ == '__main__':
    sys.exit(main())

