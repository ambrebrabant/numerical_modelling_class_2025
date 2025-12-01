import numpy as np

def root_bisection(fn, xmin, xmax, tol, nits, log=True):
    """
    Finds a root of fn(x) = 0 in the range xmin < x < xmax by the bisection method.
    Raises an error if this cannot be found within nits iterations

    Inputs:
        fn: 1 -> 1 function
        xmin: lower bound of range to search
        xmax: upper bound of range to search
        tol: tolerance within which to find root
        nits: maximum number of iterations to try before stopping
        log: if True, plot number of iterations needed to find root

    Outputs:
        m: root of fn fuch that |f(m)| < tol

    TODO currently cannot handle cases where boundary values have the same sign
    (i.e. even number of zero-crossings in the interval)
    """
    a, b = xmin, xmax
    n = 1
    while n <= nits:

        fa, fb = fn(a), fn(b)
        m = (a+b)/2
        fm = fn(m)

        if np.abs(fm)<tol: # If f(m) is within tolerance of zero, ...
            if log:
                print(f'Bisection found root within tolerance after {n} iterations')
            return m # ...return m
        # Otherwise, find in which interval sign change happens 
        elif np.sign(fa)==np.sign(fm):
            a = m
        else: # np.sign(fb) == np.sign(fm)
            b = m

        n +=1 
    raise Exception(f'Did not converge within tolerance after {nits} iterations')

def root_newton(fn, fnderiv, x0, tolf, nits, *args, log=True):
    """
    Finds a root of fn(x) = 0 in the range xmin < x < xmax by the Newton-Rhapson method.
    Raises an error if this cannot be found within nits iterations

    Inputs:
        fn: 1 -> 1 function
        fnderiv: user-supplied derivative of fn
        x0: starting point of the search
        tolf: tolerance within which to find root
        nits: maximum number of iterations to try before stopping
        *args: additional arguments for fn and fnderiv
        log: if True, plot number of iterations needed to find root

    Outputs:
        m: root of fn fuch that |f(m)| < tolf
    """
    xn = x0
    n = 1
    while n <= nits:
        fxn = fn(xn,*args)
        if np.abs(fxn)<tolf: # If f(m) is within tolerance of zero, ...
            if log:
                print(f'Newton found root within tolerance after {n} iterations')
            return xn # ...return m
        else:
            xn = xn - fxn/fnderiv(xn,*args)
        n +=1 
    raise Exception(f'Did not converge within tolerance after {nits} iterations')

def root_newton_vector(fn, fnderiv, x0, tolf, nits, *args, log=True):
    """
    Finds a root of fn(x) = 0 in the range xmin < x < xmax by the Newton-Rhapson method.
    This is a version of root_newton() which can handle vector fn and x
    This only works for fn and x of the same dimension (i.e. fn maps R^N --> R^N)
    Raises an error if this cannot be found within nits iterations

    Inputs:
        fn: N -> N function
        fnderiv: user-supplied function which returns the Jacobian matrix of shape (N,N)
        x0: starting point of the search with shape (N,)
        tolf: tolerance within which to find root
        nits: maximum number of iterations to try before stopping
        *args: additional arguments for fn and fnderiv
        log: if True, plot number of iterations needed to find root

    Outputs:
        m: root of fn fuch that |f(m)| < tolf
    """
    xn = x0
    n = 1
    while n <= nits:
        fxn = fn(xn, *args)
        if np.linalg.norm(fxn)<tolf: # If magnitude of vector f(m) is within tolerance of zero, ...
            if log:
                print(f'Newton found root within tolerance after {n} iterations')
            return xn # ...return m
        else:
            xn = xn - np.linalg.inv(fnderiv(xn, *args))@fxn
        n +=1 
    raise Exception(f'Did not converge within tolerance after {nits} iterations')