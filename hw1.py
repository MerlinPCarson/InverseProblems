#!/usr/bin/env python
# coding: utf-8
import sys
import cv2
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


def show_img(img):
   
    print(f'Image is {img.shape}')
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def diffusion_matrix(L, n, m):

    # make matrix tridiagonal 
    # 1-2*L as diagonal and L above and below diagonal
    return diags([L, 1-2*L, L], [-1, 0, 1], shape = (n,m)).toarray()
   
def noise_matrix(n, m, mean=0, stdev=1):
    return  0.01 * np.random.normal(size=(n,m))


def blur(X, B, k, enoise):

    A = np.linalg.matrix_power(B,k)
    D = np.matmul(A,X) + enoise
    Xhat = np.matmul(np.linalg.inv(A), D)

    return A, D, Xhat 

def plot_conditions(cnumbers, cmax):

    # Log Plot of conditions
    plt.figure()
    plt.title(f'Condition of $A^k$ for k=1:{cmax}, log scale')
    plt.yscale('log')
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('condition')
    plt.grid()
    plt.plot(cnumbers)

    # Linear Plot of conditions
    plt.figure()
    plt.title(f'Condition of $A^k$ for k=1:{cmax}, linear scale')
    plt.xlim((1,cmax))
    plt.xticks(np.arange(1,cmax+1,2))
    plt.xlabel('k')
    plt.ylabel('condition')
    plt.grid()
    plt.plot(cnumbers)

    plt.show()

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

def blur_image(X, B, enoise):

    cmax = 20 
    cnumbers = []
    abs_errors = []
    rel_errors = []

    for k in range(1, cmax+1):
        A, D, Xhat = blur(X, B, k, enoise)
        cnumbers.append(np.linalg.cond(A))
        abs_error = np.linalg.norm(X-Xhat)
        abs_errors.append(abs_error)
        rel_errors.append(abs_error/np.linalg.norm(X))

    #plot_conditions(cnumbers, cmax)
    plot_errors(abs_errors, rel_errors, cmax)
        
        
def main():

    imgFile = 'clown1.jpg'
    X = cv2.imread(imgFile,0)

    #show_img(X)

    n, m = X.shape
    L = 0.1
    
    B = diffusion_matrix(L, n, n)
   
    enoise = noise_matrix(n, m)

    blur_image(X, B, enoise)
    return 0


if __name__ == '__main__':
    sys.exit(main())




#
#X-Xhat = error
#plot(cnumber)
#plt(log10(cnumber))
#
#
## In[ ]:
#
#
##SVD
#A=USVtranspose = s11+U1V1transpose+s22U2V2tranpose+...
#
#S is diagonal
#U and V are orthogonal
#
#A is mxn
#U is mxm
#Vtranspose is nxn
#
#
## In[83]:
#
#
#print(A.shape)
#U, S, V = np.linalg.svd(A)
#print(U.shape, S.shape, V.shape)
#print(S)
#
