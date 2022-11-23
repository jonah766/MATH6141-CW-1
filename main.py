# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# algorithms and tooling
from RK3 import run_RK3
from GRRK3 import run_GRRK3
from utilities import Options

# external libraries
import numpy as np
import matplotlib.pyplot as plt

# the RHS function we are investigating
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


def task_1():
    print('see RK3.py')


def task_2():
    print('see GRRK3.py')


def task_3():

    # set the simulation parameters
    t_0 = 0.0
    t_max = 1.0
    dt = 0.05
    
    # set up the options
    options = Options(gamma=-2, omega=5, epsilon=0.05)
    
    # calc exact sol
    nsamples = int(np.abs(t_max - t_0) / dt)
    t_exact = np.linspace(t_0, t_max, nsamples)
    x_exact = np.sqrt(1 + np.cos(t_exact))
    y_exact = np.sqrt(2 + np.cos(options.omega * t_exact))

    # perform the computation
    t1, q1 = run_RK3(f, t_0, t_max, dt, options)
    t2, q2 = run_GRRK3(f, t_0, t_max, dt, options)

    # plot the results
    plt.subplot(2, 2, 1)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 0], color='blue')
    plt.plot(t_exact, x_exact, 'kx', label='exact')
    plt.title('RK3 algorithm solution (X)')
    plt.xlabel(r't')
    plt.ylabel(r'x(t)')
    plt.xlim(0, 1)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 1], color='red')
    plt.plot(t_exact, y_exact, 'kx', label='exact')
    plt.title('RK3 algorithm solution (Y)')
    plt.xlabel(r't')
    plt.ylabel(r'y(t)')
    plt.xlim(0, 1)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)  # 1 x 2 grid, on the 1st panel
    plt.plot(t2, q2[:, 0], color='green')
    plt.plot(t_exact, x_exact, 'kx', label='exact')
    plt.title('GRRK3 algorithm solution (X)')
    plt.xlabel(r't')
    plt.ylabel(r'x(t)')
    plt.xlim(0, 1)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)  # 1 x 2 grid, on the 1st panel
    plt.plot(t2, q2[:, 1], color='orange')
    plt.plot(t_exact, y_exact, 'kx', label='exact')
    plt.title('GRRK3 algorithm solution (Y)')
    plt.xlabel(r't')
    plt.ylabel(r'y(t)')
    plt.xlim(0, 1)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()

    plt.suptitle(f'${options.as_string()}$')
    plt.tight_layout()
    plt.show()


def task_4():
    # set up the options
    options = Options(gamma=-2, omega=5, epsilon=0.05)

    niter = 8
    dt = np.zeros(8)
    one_norm_rk3 = np.zeros(8)
    one_norm_grrk3 = np.zeros(8)
    for j in range(niter):
        # set the simulation parameters
        t_0 = 0.0
        t_max = 1.0
        dt[j] = 0.1 / (2.0**j)

        # generate the exact solution with the correct # samples
        nsamples = int(np.abs(t_max - t_0) / dt[j])
        t_exact = np.linspace(t_0, t_max, nsamples)
        y_exact = np.sqrt(2 + np.cos(options.omega * t_exact))

        # perform the computation
        t1, q1 = run_RK3(f, t_0, t_max, dt[j], options)
        t2, q2 = run_GRRK3(f, t_0, t_max, dt[j], options)

        # compute the one norms
        inner_rk3 = np.abs(q1[:, 1] - y_exact)
        one_norm_rk3[j] = dt[j] * np.sum(inner_rk3)
        
        inner_grrk3 = np.abs(q2[:, 1] - y_exact)
        one_norm_grrk3[j] = dt[j] * np.sum(inner_grrk3)

    # calc rk3 poly fit
    rk3_p = np.polyfit(np.log(dt), np.log(one_norm_rk3), 1)
    
    # plot the results
    plt.subplot(1, 2, 1)  # 1 x 2 grid, on the 1st panel
    plt.loglog(dt, one_norm_rk3, 'kx', label='Numerical Data')
    plt.loglog(dt, np.exp(rk3_p[1])*dt**(rk3_p[0]) , 'b-', label="Line slope {:.4f}".format(rk3_p[0]))
    plt.xlabel(r'dt')
    plt.ylabel(r'$\|$ Error $\|$')
    plt.title('RK3')
    plt.legend()
    plt.grid()
    
    # calc grrk3 poly fit
    grrk3_p = np.polyfit(np.log(dt), np.log(one_norm_grrk3), 1)

    # plot the results
    plt.subplot(1, 2, 2)  # 1 x 2 grid, on the 2nd panel
    plt.loglog(dt, one_norm_grrk3, 'kx', label='Numerical Data')
    plt.loglog(dt, np.exp(grrk3_p[1])*dt**(grrk3_p[0]) , 'b-', label="Line slope {:.4f}".format(grrk3_p[0]))
    plt.xlabel(r'dt')
    plt.ylabel(r'$\|$ Error $\|$')
    plt.title('GRRK3')
    plt.legend()
    plt.grid()
    
    plt.suptitle(f'RK3 and GRRK3 algorithm error plots, ${options.as_string()}$')
    plt.tight_layout()
    plt.show()


def task_5():
    # set the simulation parameters
    t_0 = 0.0
    t_max = 1.0
    dt = 0.001

    # set up the options
    options = Options(gamma=-2e5, omega=20, epsilon=0.5)
    
    # calc exact sol
    nsamples = int(np.abs(t_max - t_0) / (25*dt))
    t_exact = np.linspace(t_0, t_max, nsamples)
    x_exact = np.sqrt(1 + np.cos(t_exact))
    y_exact = np.sqrt(2 + np.cos(options.omega * t_exact))

    # perform the computation
    t1, q1 = run_RK3(f, t_0, t_max, dt, options)

    # plot the results
    plt.subplot(1, 2, 1)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 0], label='numeric solution', color='blue')
    plt.plot(t_exact, x_exact, label='exact', color='green')
    plt.xlabel(r't')
    plt.ylabel(r'x(t)')
    plt.xlim(0, 0.01)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 1], label='numeric solution', color='red')
    plt.plot(t_exact, y_exact, label='exact', color='green')
    plt.xlabel(r't')
    plt.ylabel(r'y(t)')
    plt.xlim(0, 0.01)
    plt.ylim(1.0, 1.8)
    plt.legend()
    plt.grid()
    
    plt.suptitle(f'RK3 algorithm solution, ${options.as_string()}$')
    plt.tight_layout()
    plt.show()


def task_6():
    # set the simulation parameters
    t_0 = 0.0
    t_max = 1.0
    dt = 0.001

    # set up the options
    options = Options(gamma=-2e5, omega=20, epsilon=0.5)
    
    # calc exact sol
    nsamples = int(np.abs(t_max - t_0) / (25 * dt))
    t_exact = np.linspace(t_0, t_max, nsamples)
    x_exact = np.sqrt(1 + np.cos(t_exact))
    y_exact = np.sqrt(2 + np.cos(options.omega * t_exact))

    # perform the computation
    t1, q1 = run_GRRK3(f, t_0, t_max, dt, options)

    # plot the results    
    plt.subplot(1, 2, 1)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 0], color='blue', label='Numeric solution')
    plt.plot(t_exact, x_exact, 'kx', label='exact')
    plt.xlabel(r't')
    plt.ylabel(r'x(t)')
    plt.xlim(0, 1)
    plt.ylim(1.2, 1.5)
    plt.legend(loc='lower left')
    plt.grid()

    plt.subplot(1, 2, 2)  # 1 x 2 grid, on the 1st panel
    plt.plot(t1, q1[:, 1], color='red', label='numeric solution')
    plt.plot(t_exact, y_exact, 'kx', label='exact')
    plt.xlabel(r't')
    plt.ylabel(r'y(t)')
    plt.xlim(0, 1)
    plt.ylim(0.9, 1.8)
    plt.legend(loc='lower right')
    plt.grid()
    
    plt.suptitle(f'GRRK3 algorithm solution, ${options.as_string()}$')
    plt.tight_layout()
    plt.show()


def task_7():
    # set up the options
    options = Options(gamma=-2, omega=5, epsilon=0.05)

    niter = 8
    dt = np.zeros(8)
    one_norm_grrk3 = np.zeros(8)
    for j in range(niter):
        # set the simulation parameters
        t_0 = 0.0
        t_max = 1.0
        dt[j] = 0.1 / (2.0**j)

        # generate the exact solution with the correct # samples
        nsamples = int(np.abs(t_max - t_0) / dt[j])
        t_exact = np.linspace(t_0, t_max, nsamples)
        y_exact = np.sqrt(2 + np.cos(options.omega * t_exact))

        # perform the computation
        t1, q1 = run_RK3(f, t_0, t_max, dt[j], options)
        t2, q2 = run_GRRK3(f, t_0, t_max, dt[j], options)

        # compute the one norms      
        inner_grrk3 = np.abs(q2[:, 1] - y_exact)
        one_norm_grrk3[j] = dt[j] * np.sum(inner_grrk3)

    # calc grrk3 poly fit
    grrk3_p = np.polyfit(np.log(dt), np.log(one_norm_grrk3), 1)

    # plot the results
    plt.subplot(1, 1, 1)  # 1 x 2 grid, on the 2nd panel
    plt.loglog(dt, one_norm_grrk3, 'kx', label='Numerical Data')
    plt.loglog(dt, np.exp(grrk3_p[1])*dt**(grrk3_p[0]) , 'b-', label="Line slope {:.4f}".format(grrk3_p[0]))
    plt.xlabel(r'dt')
    plt.ylabel(r'$\|$ Error $\|$')
    plt.title('GRRK3')
    plt.legend()
    plt.grid()
    
    plt.suptitle(f'GRRK3 algorithm error plot, ${options.as_string()}$')
    plt.tight_layout()
    plt.show()


def main():
    # %% Task (1) - Implement the RK3 algorithm
    task_1()

    # %% Task (2) - Implement the GRRK3 algorithm
    task_2()

    # %% Task (3) - Apply your RK3 and GRRK3 algorithms to the system of equations (2) (unstiff)
    task_3()

    # %% Task (4) - show evidence of convergence of your algorithms
    task_4()

    # %% Task (5) - Apply your RK3 algorithm to the system of equations (2) (stiff)
    task_5()

    # %% Task (6) - Apply your GRRK3 algorithm to the system of equations (2) (stiff)
    task_6()

    # %% Task (7) - Show evidence of the convergence of your GRRK3 algorithm
    task_7()


if __name__ == '__main__':
    main()
