#!/usr/bin/env python3
# coding: utf-8

###################################################
# Script for de-blurring an image                 #
# uses TSVD and Tikhonov regularization           #
# using first and second derivative ops           #
# and Total Variance                              #
#                                                 #
# Author: Merlin Carson                           #
# Date: Oct-30-2019                               #
###################################################

import sys
import cv2
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm


def plt_error(error1, error2):

    plt.plot(error1)
    plt.plot(error2)
    plt.show()

def plt_compare(data1, data2, title=None, pause=True):

    plt.plot(data1)
    plt.plot(data2)
    plt.title(title)
    if pause:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)

def plot_l_curve(norms, residuals):

    plt.figure()
    plt.title(f'L-Curve')
    plt.ylabel('$||X_λ||$')
    plt.xlabel('||$A{X_λ}-\hat{D}$||')
    plt.yscale('log', basey=10)
    #plt.xticks(residuals)
    plt.grid()
    plt.plot(residuals, norms)
    #plt.xlim(residuals[-1],residuals[0])
    plt.show()


# create diffusion matrix n x m
def diffusion_matrix(L, n, m):

    # make matrix tridiagonal 
    # 1-2*L as diagonal and L above and below diagonal
    return diags([L, 1-2*L, L], [-1, 0, 1], shape = (n,m)).toarray()


def solve(U, S, V, dhat, p, method):
    fltr = 1.0
    xhat = np.zeros((dhat.shape[0]))
    
    if method is 'TK':
        n = S.shape[0]  # use all singular values
    else:
        n = p           # truncate singular values

    for i in range(n):
        # determine filter factor based on method
        if method is 'TK':
            fltr = S[i]**2/(S[i]**2 + p**2) 

        xi = fltr * (np.dot(U[:,i].T, dhat)/S[i]) * V[:,i]
        xhat = np.add(xhat, xi)

    return np.expand_dims(xhat, axis=1) 

    
def regularize(A, Dhat, method, p):
   
    U, S, Vt = np.linalg.svd(A)
    m, n = Dhat.shape               

    Xhat = np.empty((m,0))

    for i in range(n):
        dhat = Dhat[:,i]

        xhat = solve(U, S, Vt.T, dhat, p, method) 

        Xhat = np.hstack((Xhat, xhat)) 

    return Xhat 


def de_blur(A, X, Dhat, method):
    n = A.shape[0] 
    norms = []
    residuals = []
    if method is 'TSVD':
        print('Regularizing with Truncated Singular Value Decomposition')
        for k in tqdm(range(50,150)):  # try all truncation values to find best fit
            Xhat = regularize(A, Dhat, method, k)
            norm = np.linalg.norm(Xhat)
            norms.append(norm)
            res = np.linalg.norm(np.subtract(np.matmul(A,Xhat),Dhat))
            residuals.append(res)
            plt_compare(Xhat, X, title=f'k={k}', pause=False)
    elif method is 'TK':
        print('Regularizing with Tikhonov Regularization')
        residuals = []
        for Lambda in tqdm(np.arange(0.000001,1.0,0.001)):  # try all truncation values to find best fit
            Xhat = regularize(A, Dhat, method, Lambda)
            norm = np.linalg.norm(Xhat)
            norms.append(norm)
            res = np.linalg.norm(np.subtract(np.matmul(A,Xhat),Dhat))
            residuals.append(res)
            #show_img(Xhat, title=f'λ={Lambda}', pause=False)


    else:
        sys.exit('Unknown method') 

    plot_l_curve(norms, residuals)

def load_data_m(dataFile):
    Dhat = []
    with open(dataFile, 'r') as data:
        for row in data.readlines():
            Dhat.append([float(val) for val in row.split()])

    return np.array(Dhat)


def main():

    # open image
    imgFile = 'hw2blur.jpg'
    dataFile = 'BlurData.m'
    trueData = 'TrueData.m'
    #Dhat = cv2.imread(imgFile,0)

    Dhat = load_data_m(dataFile) 
    X = load_data_m(trueData)

    # display Blurred data and True data 
    plt_compare(Dhat, X)


    n, m = Dhat.shape       # dims of image
    L = 0.45                # diagonal val for blurring matrix
    blur_op_power = 10      # power to raise the blurring operator matrix by
   
    # create blurring matrix 
    B = diffusion_matrix(L, n, n)
    A = np.linalg.matrix_power(B,blur_op_power)  # diffusion matrix B^k
  
    # regularization 
    method = 'TSVD'
    k=124
    Xhat = regularize(A, Dhat, method, k)
    k=125
    Xhat2 = regularize(A, Dhat, method, k)
    plt_compare(X, Xhat2)
    plt_error(Xhat-X, Xhat2-X)
    print(np.sum((Xhat-X)**2), np.sum((Xhat2-X)**2))
    ##show_img(Xhat, title=f'k={k}', pause=True)
    #de_blur(A, X, Dhat, method)

    # regularization 
    method = 'TK'
    Lambda=0.004
    Xhat = regularize(A, Dhat, method, Lambda)
    #Best?
    Lambda=0.0035
    Xhat2 = regularize(A, Dhat, method, Lambda)
    plt_compare(X, Xhat)
    plt_error(Xhat-X, Xhat2-X)
    print(np.sum((Xhat-X)**2), np.sum((Xhat2-X)**2))
    #show_img(Xhat, title=f'λ={Lambda}', pause=True)
    #de_blur(A, Dhat, method)

    return 0


if __name__ == '__main__':
    sys.exit(main())

