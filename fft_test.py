import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import time

import math

class Polynomial:

    def __init__(self, coeficents, degrees=None):
        if not degrees:
            self.degree = list(reversed(range(len(coeficents))))
        else:
            self.degree = degrees
        self.coeficents = coeficents

    def __call__(self, x):
        print(self.coeficents)
        print(self.degree)
        return sum([self.coeficents[i]*x**self.degree[i] for i in range(len(self.coeficents))])

a = Polynomial([1, 2,1], [2,1,0])
b = Polynomial([1,-2,1], [2,1,0])

_PI = np.math.pi
# Number of sample points
N = 1024
# sample spacing
T = 1 / N

def DFT_slow(x, inverse : bool):
    """discrete Fourier Transform of the 1D array x"""
    tmp_PI = _PI if inverse else -_PI
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * tmp_PI * k * n / N)
    return np.dot(M, x)

def FFT_Recursive(x, inverse : bool = False):
    """ FFT with recursive Cooley-Tukey Algorithm"""
    N = x.shape[0]
    
    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")
    # this cutoff should be optimized
    # as recursion on small N does not help much
    # rather do in in one N^2 call.
    elif N <= 16:
        if inverse:
            return DFT_slow(x, True)/N
        else:
            return DFT_slow(x, False)

    tmp_PI = _PI if inverse else -_PI
    w = np.exp(-2j * tmp_PI / N * np.arange(N))
    x_even = FFT_Recursive(x[::2], inverse)
    x_odd = FFT_Recursive(x[1::2], inverse)
    
    X = np.concatenate([x_even + w[: N>>1]*x_odd, x_even + w[ N>>1:]*x_odd])
    return X

def FFT_iter(x, inverse : bool= False):
    """Iter. version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    tmp_PI = _PI if inverse else -_PI
    N = x.shape[0]

    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 16)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * _PI * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    if inverse:
        X/=N
    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1]>>1]
        X_odd = X[:, X.shape[1]>>1:]
        factor = np.exp(-1j * _PI * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])
    return X.ravel()

def timeit(tgt_func, msg = '', rpt = 1):
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg}\n>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')

x = np.linspace(0.0, N*T, N)
y = 0.75*np.sin(5 * 2.0*_PI*x) + 0.5*np.sin(250 * 2.0*_PI*x)# + 0.25*np.sin(400 * 2.0*_PI*x)

xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
yf = FFT_iter(y, False)
rec = FFT_iter(yf, True)

#ToCompare
newyf=fft(y)
newrec=ifft(newyf)

print(np.allclose(yf, newyf))
print(np.allclose(rec, newrec))

fig, ax = plt.subplots(5)

ax[0].plot(x, y)
ax[1].plot(xf, np.abs(newyf[:N//2]))
ax[2].plot(xf, np.abs(yf[:N//2]))
ax[3].plot(x, rec.real)
ax[4].plot(x, newrec.real)
plt.show()

timeit(lambda: FFT_Recursive(y), 'FFT_recu', 10)
timeit(lambda: FFT_iter(y), 'FFT_iter', 10)