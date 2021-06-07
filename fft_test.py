import matplotlib.pyplot as plt
import numpy as np
import time


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

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    N = len(x)
    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")
    # this cutoff should be optimized
    elif N <= 16:
        return DFT_slow(x)

    X_even = FFT(x[::2])
    X_odd = FFT(x[1::2])
    factor = np.exp(-2j*np.pi*np.arange(N)/ N)
    
    X = np.concatenate([X_even+factor[:N>>1]*X_odd, X_even+factor[N>>1:]*X_odd])
    return X

def timeit(tgt_func, msg = '', rpt = 1):
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg}\n>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')

# Number of samplepoints
N = 2048*16
# sample spacing
T = 1 / N

x = np.linspace(0.0, N*T, N)
y = 0.1*np.sin(50 * 2.0*np.pi*x) + 0.7*np.sin(110 * 2.0*np.pi*x) + 0.3*np.sin(470 * 2.0*np.pi*x)
#timeit(lambda: DFT_slow(y), 'DFT', 5)

yf = FFT(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots(2)
ax[0].plot(x, y)
ax[1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

timeit(lambda: FFT(y), 'FFT', 20)
#np.fft.fft(x)