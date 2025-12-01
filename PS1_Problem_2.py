import numpy as np
import matplotlib.pyplot as plt
import shared_funcs as sf

def ODE_derivative(y,t,w):
    """
    See problem sheet for variable definitions
    """
    return -y*w

def ODE_second_derivative(y,t,w):
    """
    Derivative of ODE_derivative w.r.t. y
    """
    return -w

def volcanic_forcing(t):
    return 2 + 2*np.sin(4*np.pi*t)

def no_forcing(t):
    """
    Works for scalar inputs only
    """
    return 0

def forward_euler_step(dt, t, y_n, F_t, f_yt, dfdy_yt, *args):
    """
    Computes y at time step n+1 according to a first-order forward Euler scheme

    Inputs:
        dt: time step
        t: time
        y_n: y(t)
        F_t: function F(t) giving additional external forcing
        f_yt: function f(y,t) giving dy/dt in the non-forced case
        dfdy_yt: partial derivative of f(y,t,*args) wrt y. Not used here, but has to 
            be in arguments so that it can be called interchangeably with backward_euler_step.
            Can replace with a dummy.
        *args: optional arguments for f_yt
    
    Outputs:
        y(t+dt)
    """
    return y_n + dt*(F_t(t) + f_yt(y_n, t, *args))

def backward_euler_step(dt, t, y_n, F_t, f_yt, dfdy_yt, *args):
    """
    Computes y at time step n+1 according to a first-order backward Euler scheme

    Inputs:
        dt: time step
        t: time
        y_n: y(t)
        F_t: function F(t) giving additional external forcing
        f_yt: function f(y,t,*args) giving dy/dt in the non-forced case
        dfdy_yt: partial derivative of f(y,t,*args) wrt y
        *args: optional arguments for f_yt
    
    Outputs:
        y(t+dt)

    Looks for root within tolerance of 1e-6 and returns an error after 100 iterations (inflexible)
    """

    def func_to_root(y_n1):
        """
        Find the root of this function to find y(t+dt)
        """
        return y_n1 - y_n - dt*(F_t(t+dt) + f_yt(y_n1, t+dt, *args))
    
    def derivative_of_func_to_root(y_n1):
        return 1 - dt*dfdy_yt(y_n1, t+dt, *args)
    
    y_n_plus_1 = sf.root_newton(func_to_root, derivative_of_func_to_root, y_n, 1e-9, 100, log=False)
    return y_n_plus_1

def euler_integrate(dt, tend, y0, method, F_t, f_yt, dfdy_yt, *args):
    """
    Uses an Euler method to solve a first-order initial-value from t=0 to t=tend

    Inputs:
        dt: time step 
        tend: time at which to stop
        y0: initial value
        method: function - scheme used to step forward in time
        F_t: function F(t) giving additional external forcing
        f_yt: function f(y,t,*args) giving dy/dt in the non-forced case
        dfdy_yt: partial derivative of f(y,t,*args) wrt y
        *args: optional arguments for f_yt

    Outputs:
        tvec: times
        yvec: solution values
        tvec and yvec are 1d ndarrays of the same length
    """
    tvec = np.arange(0, tend, dt)
    yvec = np.zeros(tvec.shape)

    yvec[0] = y0

    for i, t in enumerate(tvec[:-1]):
        yvec[i+1] = method(dt, t, yvec[i], F_t, f_yt, dfdy_yt, *args)

    return tvec, yvec

def unforced_analytical_solution(y0, w, t):
    return y0 * np.exp(-w*t)

def euler_forward_error(y0, w, dt, tvec):
    """
    Calculates the error in the forward euler solution of the unforced exponential decay problem
    Variables have been defined previously
    """
    return y0*((1-w*dt)**(tvec/dt) - np.exp(-w*tvec))

def euler_backward_error(y0, w, dt, tvec):
    """
    Calculates the error in the backward euler solution of the unforced exponential decay problem
    Variables have been defined previously
    """
    return y0*((1+w*dt)**(-tvec/dt) - np.exp(-w*tvec))

w = 5
y0 = 1
tend = 5

dt_list = [0.01, 0.05, 0.1, 0.2, 0.5]

forward_error = {}
backward_error = {}

# Solve the unforced system for different values of dt
forward_error = {}
backward_error = {}
for j, dt in enumerate(dt_list):

    tvec_forward, yvec_forward = euler_integrate(dt, tend, y0, forward_euler_step, no_forcing, ODE_derivative, ODE_second_derivative, w)
    tvec_backward, yvec_backward = euler_integrate(dt, tend, y0, backward_euler_step, no_forcing, ODE_derivative, ODE_second_derivative, w)

    # Save time series of errors
    forward_error[dt] = np.vstack((tvec_forward, yvec_forward - unforced_analytical_solution(y0, w, tvec_forward)))
    backward_error[dt] = np.vstack((tvec_backward, yvec_backward - unforced_analytical_solution(y0, w, tvec_backward)))

    # Plot analytical and numerical solutions for comparison
    fig, ax = plt.subplots()
    ax.plot(tvec_forward, unforced_analytical_solution(y0, w, tvec_forward), label='Analytical')
    ax.plot(tvec_forward, yvec_forward, label='Forward Euler')
    ax.plot(tvec_backward, yvec_backward, label='Backward Euler')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.set_title(rf'$\Delta t$ = {dt}')
    plt.show()

# Plot time series of errors
fig, axx = plt.subplots(1,2,layout='constrained',figsize=(8,4))
for dt in dt_list:

    fe = forward_error[dt]
    # Plot numerical error in practice
    axx[0].semilogy(fe[0,1:], np.abs(fe[1,1:]), label=dt)
    # Plot analytical expression for numerical error
    axx[0].semilogy(fe[0,1:], np.abs(euler_forward_error(y0, w, dt, fe[0,1:])), 'k--', alpha=0.5)

    fb = backward_error[dt]
    # Plot numerical error 
    axx[1].semilogy(fb[0,1:], np.abs(fb[1,1:]), label=dt)
    # Plot analytical expression for numerical error
    axx[1].semilogy(fb[0,1:], np.abs(euler_backward_error(y0, w, dt, fb[0,1:])), 'k--', alpha=0.5)
for ax in axx:
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\log(|\varepsilon|)$')
axx[0].set_title('Forward Euler')
axx[1].set_title('Backward Euler')
axx[0].legend(title=r'$\Delta t$')
fig.suptitle('Unforced case - Numerical errors')
plt.show()

# Solve the forced system

for dt in [0.01, 0.1]:

    tvec_forward, yvec_forward = euler_integrate(dt, tend, y0, forward_euler_step, volcanic_forcing, ODE_derivative, ODE_second_derivative, w)
    tvec_backward, yvec_backward = euler_integrate(dt, tend, y0, backward_euler_step, volcanic_forcing, ODE_derivative, ODE_second_derivative, w)

    plt.plot(tvec_forward, yvec_forward, label=fr'Forward Euler, $\Delta t$ = {dt}')
    plt.plot(tvec_backward, yvec_backward, label=fr'Backward Euler, $\Delta t$ = {dt}')

plt.legend()
plt.show()