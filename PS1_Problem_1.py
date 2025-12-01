import numpy as np
import matplotlib.pyplot as plt
import shared_funcs as sf

### a.i), b.i) Use bisection + Newton method to find root of quadratic
def example_quadratic(x):
    """
    Roots are at x = 1 and x = -6
    Single global minimum at x = -2.5
    """
    return x**2 + 5*x - 6

def example_quadratic_derivative(x):
    return 2*x + 5

qxmin, qxmax = -1.5, 6
tol = 1e-6

# Find the root at x = 1 with bisection
qroot1 = sf.root_bisection(example_quadratic, qxmin, qxmax, tol, 100)
# Find the root at x = -6 with bisection
qroot2 = sf.root_bisection(example_quadratic, -10, qxmin, tol, 100)
# Find the root at x = 2 with Newton-Rhapson
qroot_nr = sf.root_newton(example_quadratic, example_quadratic_derivative, qxmin, tol, 100)

# Plot
qxvec = np.linspace(qxmin, qxmax, 100)
plt.plot(qxvec, example_quadratic(qxvec))
plt.axhline(0, color='grey', zorder=-1)
plt.axvline(0, color='grey', zorder=-1)
plt.scatter(qroot1, example_quadratic(qroot1), color='r', label='Bisection method')
plt.scatter(qroot_nr, example_quadratic(qroot_nr), color='g', label='Newton method',
            marker='+', s=100)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Example quadratic')
plt.show()




### a.ii), b.ii) Use bisection + Newton method to find root of transcendental equation
def example_transcendental(x):
    return 1 + x/(1+np.exp(x))

def example_transcendental_derivative(x):
    return (1 + (1-x)*np.exp(x))/(1+np.exp(x))**2

txmin, txmax = -2, 6

troot = sf.root_bisection(example_transcendental, txmin, txmax, tol, 100)
troot_nr = sf.root_newton(example_transcendental, example_transcendental_derivative, 
                       txmin, tol, 100)

txvec = np.linspace(txmin, txmax, 100)
plt.plot(txvec, example_transcendental(txvec))
plt.scatter(troot, example_transcendental(troot), color='r', label='Bisection')
plt.scatter(troot_nr, example_transcendental(troot_nr), label='Newton',
            color='g', marker='+', s=100)
plt.axhline(0, color='grey')
plt.axvline(0, color='grey')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Example transcendental')
plt.show()


