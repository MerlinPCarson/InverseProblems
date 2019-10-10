#!/usr/bin/env python3
# coding: utf-8

###################################################
# Script for blurring and reconstructing an image #
#                                                 #
# Author: Merlin Carson                           #
# Date: Oct-9-2019                                #
###################################################

import sys
import cv2
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


# plot condition values
def plot_conditions(cnumbers, cmax):

    # Log Plot of conditions
    plt.figure()
    plt.title(f'Condition of $B^k$ for k=1:{cmax}, log scale')
    plt.yscale('log')
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('condition')
    plt.grid()
    plt.plot(cnumbers)

    # Linear Plot of conditions
    plt.figure()
    plt.title(f'Condition of $B^k$ for k=1:{cmax}, linear scale')
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('condition')
    plt.grid()
    plt.plot(cnumbers)

    plt.show()


# plot absolute and relative errors
def plot_errors(abs_errors, rel_errors, cmax):

    # Log Plot of conditions
    plt.figure()
    plt.title(f'Absolute Errors of $\hat{{X}}$ for k=1:{cmax}, log scale base 10')
    plt.yscale('log', basey=10)
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.grid()
    plt.plot(abs_errors)

    # Linear Plot of conditions
    plt.figure()
    plt.title(f'Relative Errors of $\hat{{X}}$ for k=1:{cmax}, log scale base 10')
    plt.yscale('log', basey=10)
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.grid()
    plt.plot(rel_errors)

    plt.show()


# display an image matrix on screen
def show_img(img, title=None):
   
    print(f'Image dimensions are {img.shape}')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


# create diffusion matrix n x m
def diffusion_matrix(L, n, m):

    # make matrix tridiagonal 
    # 1-2*L as diagonal and L above and below diagonal
    return diags([L, 1-2*L, L], [-1, 0, 1], shape = (n,m)).toarray()


# create matrix of noise n x m   
def noise_matrix(n, m, mean=0, stddev=1):
    return  0.01 * np.random.normal(mean, stddev, size=(n,m))


def blur(X, B, k, enoise):

    A = np.linalg.matrix_power(B,k)         # diffusion matrix B^k
    D = np.matmul(A,X) + enoise             # blurred image
    Xhat = np.matmul(np.linalg.inv(A), D)   # reconstructed image

    return A, D, Xhat 


def blur_image(X, B, enoise, cmax):

    cnumbers = []
    abs_errors = []
    rel_errors = []

    # blur image with B^kX+noise for k=1:cmax
    for k in range(1, cmax+1):
        A, D, Xhat = blur(X, B, k, enoise)
        cnumbers.append(np.linalg.cond(A))
        abs_error = np.linalg.norm(X-Xhat)
        abs_errors.append(abs_error)
        rel_errors.append(abs_error/np.linalg.norm(X))

        # show reconstructions
        if k in [1,5,20]:
            title = f'Reconstruction of image for k={k}' 
            show_img(Xhat, title)

    # plotting functions
    plot_conditions(cnumbers, cmax)
    plot_errors(abs_errors, rel_errors, cmax)
        
        
def main():

    # open image
    imgFile = 'clown1.jpg'
    X = cv2.imread(imgFile,0)

    # display image
    show_img(X, 'Original Picture')

    n, m = X.shape  # dims of image
    L = 0.1         # diagonal val for blurring matrix
   
    # create blurring matrix 
    B = diffusion_matrix(L, n, n)
  
    # create noise matrix 
    enoise = noise_matrix(n, m, stddev=1)

    # blur image
    cmax = 20 
    blur_image(X, B, enoise, cmax)

    return 0


if __name__ == '__main__':
    sys.exit(main())

