import numpy as np
import matplotlib.pyplot as plt
import shared_funcs as sf

def steady_lorenz(xvec, sigma, r, beta):
    x, y, z = xvec
    return np.array([sigma*(y-x),
                     r*x - y - x*z,
                     x*y - beta*z])

def steady_lorenz_derivative(xvec, sigma, r, beta):
    x, y, z = xvec
    return np.array([[-sigma,sigma,0],
                     [r-z,-1,-x],
                     [y,x,-beta]])

beta0 = 8/3
sigma0 = 10
r0 = 28

### Find roots of steady-state Lorentz equations ###

x0 = np.array([0.01, 0.01, 0.01])
zero_root = sf.root_newton_vector(steady_lorenz, steady_lorenz_derivative, x0, 1e-6, 100, sigma0, r0, beta0, log=True)

x0_nz_plus = np.array([10, 10, 28])
plus_non_zero_root = sf.root_newton_vector(steady_lorenz, steady_lorenz_derivative, x0_nz_plus, 1e-6, 100, sigma0, r0, beta0, log=True)

x0_nz_minus = np.array([-10, -10, 28])
minus_non_zero_root = sf.root_newton_vector(steady_lorenz, steady_lorenz_derivative, x0_nz_minus, 1e-6, 100, sigma0, r0, beta0, log=True)


### Bonus - solve non-steady equations numerically ###

x0 = np.array([1, 1, 1])
tmax = 50
dt = 0.01
tvec = np.arange(0,tmax,dt)
xvec = np.zeros((len(tvec),3))
xvec[0,:] = x0

for i, t in enumerate(tvec[:-1]):
    xvec[i+1,:] = xvec[i,:] + dt*steady_lorenz(xvec[i,:], sigma0, r0, beta0)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xvec[:,0], xvec[:,1], xvec[:,2])
plt.show()