# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import Options

# one iteration of the RK3 algorithm
def RK3_step(f, tn, qn, dt, options):
    k_1 = f(tn, qn, options)
    k_2 = f(tn+0.5*dt, qn+0.5*dt*k_1, options)
    k_3 = f(tn+dt, qn+dt*(-k_1 + 2*k_2), options)
    step = (dt/6.0) * (k_1 + 4*k_2 + k_3)
    qnew = qn + step
    return qnew

# to run many iterations of the RK3 algorithm at once
def run_RK3(f, t_0, t_max, dt, options):
    
    nsteps = int((t_max-t_0)/dt)
    
    # allocate some memory for the solution
    t = np.arange(t_0, t_max, dt)
    q = np.zeros((nsteps,2))
    q[0] = np.array([np.sqrt(2), np.sqrt(3)])
    
    # numerically compute the solution to the diff eq
    for i in  range(nsteps-1):
        q[i+1] = RK3_step(f, t[i], q[i], dt, options)
        
    return t, q
      
def main():
    # define the function modelling the diff eq
    def f(t, q, options):
        
        # ensure the input is the correct length
        assert len(
            q) == 2, f'ERROR, q should be of length 2, q is of length {len(q)}'
        
        # perform the computation in steps
        x,y = q
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
    t,q = run_RK3(f, t_0, t_max, dt, options)
    
    # plot the results
    plt.plot(t,q)
    plt.show()

if __name__ == '__main__':
    main()
    
