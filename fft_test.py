import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.fftpack import fft,ifft,fft2,ifft2,fftshift
import time
import cv2
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

def DFT_slow(x, inverse : bool = False):
    """discrete Fourier Transform of the 1D array x"""
    tmp_PI = -_PI if inverse else _PI
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * tmp_PI * k * n / N)
    return np.dot(M, x)/N if inverse else np.dot(M, x)

def FFT_recu(x, inverse : bool = False):
    """ FFT with recursive Cooley-Tukey Algorithm"""
    N = x.shape[0]
    
    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")
    # this cutoff should be optimized
    # as recursion on small N does not help much
    # rather do in in one N^2 call.
    elif N <= 16:
        return DFT_slow(x, inverse)

    tmp_PI = -_PI if inverse else _PI
    w = np.exp(-2j * tmp_PI / N * np.arange(N))
    x_even = FFT_recu(x[::2], inverse)
    x_odd = FFT_recu(x[1::2], inverse)
    
    X = np.concatenate([x_even + w[: N>>1]*x_odd, x_even + w[ N>>1:]*x_odd])
    return X/N if inverse else X

def FFT_iter(x, inverse : bool= False):
    """Iter. version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=complex)
    tmp_PI = -_PI if inverse else _PI
    N = x.shape[0]

    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")
    # this cutoff should be optimized
    # as recursion on small N does not help much
    # rather do in in one N^2 call.
    N_min = min(N, 16)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * tmp_PI * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    
    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1]>>1]
        X_odd = X[:, X.shape[1]>>1:]
        factor = np.exp(-1j * tmp_PI * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
    return X.ravel()

def timeit(tgt_func, msg = '', rpt = 1):
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg}\n>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')

def rgb2gray(rgb):
    bw = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return cv2.resize(bw, (1024, 1024), interpolation=cv2.INTER_CUBIC)



def img_transform(data, inverse : bool= False):
    fig, ax = plt.subplots(5)

    tmp = np.array(list(map(lambda row: FFT_iter(row), data)), dtype=complex)
    tmp=tmp.T

    showtmp = fftshift(tmp)
    ax[0].imshow(showtmp.real, cmap=plt.get_cmap('gray'))#, vmin = 0, vmax = 16)
    ax[1].imshow(showtmp.imag, cmap=plt.get_cmap('gray'))#, vmin = 0, vmax = 16)

    tmp = np.array(list(map(lambda col: FFT_iter(col), data)), dtype=complex)
    tmp=tmp.T

    
    showtmp = fftshift(tmp)
    ax[2].imshow(showtmp.real, cmap=plt.get_cmap('gray'))#, vmin = 0, vmax = 16)
    ax[3].imshow(showtmp.imag, cmap=plt.get_cmap('gray'))#, vmin = 0, vmax = 16)
    ax[4].imshow(np.abs(showtmp), cmap=plt.get_cmap('gray'))#, vmin = 0, vmax = 16)
    #print(tmp)
    return data


fig, ax = plt.subplots(4)

img = mpimg.imread('./ripple.png')
#print(img.shape)
img = rgb2gray(img)
#print(img.shape)
test = img_transform(img)
# Number of sample points
N = 1024
# sample spacing
T = 1 / N

x = np.linspace(0.0, 1.0, img.shape[0])
y = img[400]
#x = np.linspace(0.0, 1.0, N)
#y = 0.75*np.sin(5 * 2.0*_PI*x) + 0.5*np.sin(250 * 2.0*_PI*x) + 0.25*np.sin(400 * 2.0*_PI*x)

ax[0].imshow(img, cmap=plt.cm.binary)

imgf = fft2(img)
print(imgf.shape)
ax[1].imshow((fftshift(imgf)).real, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 512)
ax[2].imshow((fftshift(imgf)).imag, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 512)

rec = ifft2(imgf)
ax[3].imshow(np.abs(rec), cmap=plt.cm.binary)
plt.show()

xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
yf = FFT_iter(y, False)
rec = FFT_iter(yf, True)/N

#To Compare
newyf=fft(y)
newrec=ifft(newyf)

print(np.allclose(yf, newyf))
print(np.allclose(rec, newrec))

fig, ax1d = plt.subplots(5)

ax1d[0].plot(x, y)
ax1d[1].plot(xf, np.abs(newyf[:N//2]))
ax1d[2].plot(xf, np.abs(yf[:N//2]))
ax1d[3].plot(x, rec.real)
ax1d[4].plot(x, newrec.real)
plt.show()

# performance benchmark
if N < 2048:
    timeit(lambda: DFT_slow(y), 'DFT_slow', 5)
else: print("DFT_slow\n>avg    >1000 ms per loop")
timeit(lambda: FFT_recu(y), 'FFT_recu', 10)
timeit(lambda: FFT_iter(y), 'FFT_iter', 10)