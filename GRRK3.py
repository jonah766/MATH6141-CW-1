# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:40:48 2022

@author: jf3g19
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import Options
from scipy.optimize import fsolve

# to solve the system of linear equations to obtain k_1 and k_2
def solve_k(f, t_n, q_n, dt, options):
    # setup the initial guess
    k_1_0 = f(t_n+dt/3.0, q_n, options)
    k_2_0 = f(t_n+dt, q_n, options)
    K_0 = np.array([k_1_0, k_2_0])
    K_shape = K_0.shape

    # the function we want to solve
    def F(K, f, t_n, q_n, dt, options):
        a = f(t_n + (1.0/3.0)*dt, q_n + (dt/12.0)*(5*K[0:1] - K[2:3]), options)
        b = f(t_n + dt, q_n + (dt/4.0)*(3*K[0:1] + K[2:3]), options)
        return K - np.array([a, b]).ravel()

    # solve for K = (k_1,k_2)^T
    K = fsolve(F, K_0.ravel(), args=(f, t_n, q_n, dt, options))
    return K.reshape(K_shape)

# one iteration of the RK3 algorithm
def GRRK3_step(f, t_n, q_n, dt, options):
    k_1, k_2 = solve_k(f, t_n, q_n, dt, options)
    step = (dt/4.0) * (3*k_1 + k_2)
    q_new = q_n + step
    return q_new

# to run many iterations of the RK3 algorithm at once
def run_GRRK3(f, t_0, t_max, dt, options):

    nsteps = int((t_max-t_0)/dt)

    # allocate some memory for the solution
    t = np.arange(t_0, t_max, dt)
    q = np.zeros((nsteps, 2))
    q[0] = np.array([np.sqrt(2), np.sqrt(3)])

    # numerically compute the solution to the diff eq
    for i in range(nsteps-1):
        q[i+1] = GRRK3_step(f, t[i], q[i], dt, options)

    return t, q


def main():
    # define the function modelling the diff eq
    def f(t, q, options):

        # ensure the input is the correct length
        assert len(
            q) == 2, f'ERROR, q should be of length 2, q is of length {len(q)}'

        # perform the computation in steps
        x, y = q
        A = np.array([[options.gamma, options.epsilon], [options.epsilon, -1]])
        B = np.array([(-1 + x**2 - np.cos(t))/(2*x),
                     (-2 + y**2 - np.cos(options.omega*t))/(2*y)])
        C = np.array([np.sin(t)/(2*x), options.omega *
                     np.sin(options.omega * t)/(2*y)])

        return np.matmul(A, B) - C

    # set the simulation parameters
    t_0 = 0.0
    t_max = 1.0
    dt = 0.0001

    # set up the options
    options = Options(gamma=-2, omega=5, epsilon=0.05)

    # perform the computation
    t, q = run_GRRK3(f, t_0, t_max, dt, options)

    # plot the results
    plt.plot(t, q)
    plt.show()


if __name__ == '__main__':
    main()
