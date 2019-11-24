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
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

def plot(estimates):

    print(estimates.shape)
    #for i in range(estimates.shape[0]):
    #    print(estimates[i, 0], estimates[i,1])
    plt.plot(estimates[:, 0], estimates[:, 1])
    plt.show()

def pos(t, x0, v0):
    return v0 * sin(t) + x0 * cos(t)
    
def velocity(t, x0, v0):
    return v0 * cos(t) - x0 * sin(t)

def observation(X, H, obs_error):
    return H@X + obs_error

def model(X, M, model_error):
    return M@X + model_error

def forecast(X, M):
    return M@X 

def cov_forecast(M, P_a, Q):
    return M@P_a@M.T + Q

def gen_observation(H, x, obs_error):
    return H@x + obs_error

def kalman_gain(P_f, H, R):
    return P_f@H.T @ np.linalg.inv(H@P_f@H.T + R) 

def analysis(x_f, K, y, H):
    return x_f + K@(y - H@x_f)

def cov_analysis(K, H, P_f):
    return (np.eye(K.shape[0]) - K@H)@P_f


def main():
    time_steps = 1000

    # setup errors
    prior_error = np.random.normal(loc=0, scale=1, size=(2,1))
    obs_error = np.random.normal(loc=0, scale=0.1, size=(2,1))
    model_error = np.random.normal(loc=0, scale=1, size=(2,1))

    # setup initial vals
    t = 0
    x0 = v0 = 1
    X0 = np.array([[pos(t, x0, v0)],[velocity(t, x0, v0)]])

    H = np.array([[1,0], [0,1]])
    M = np.array([[1,1], [-1,1]])
    P_a = Q = np.eye(M.shape[0])
    R = np.zeros(H.shape)
    

    x = X0 + prior_error

    estimates = []
    for _ in range(time_steps):

        # forecast
        x_f = forecast(x, M)
        P_f = cov_forecast(M, P_a, Q)

        # generate observation
        y = gen_observation(H, x, obs_error)

        # calc Kalman gain
        K = kalman_gain(P_f, H, R)

        # analysis step
        x = analysis(x_f, K, y, H)
        estimates.append(x.flatten())

        P_a = cov_analysis(K, H, P_f)

    plot(np.array(estimates))

    return 0

if __name__ == '__main__':
    sys.exit(main())