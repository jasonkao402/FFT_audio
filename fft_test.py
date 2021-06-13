import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
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


a = Polynomial([1, 2, 1], [2, 1, 0])
b = Polynomial([1, -2, 1], [2, 1, 0])

_PI = np.math.pi
IMG_SIZE = (512, 512)
# Number of sample points
N = 1024
# sample spacing
T = 1 / N


def DFT_slow(x, inverse: bool = False):
    """discrete Fourier Transform of the 1D array x"""
    tmp_PI = -_PI if inverse else _PI
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * tmp_PI * k * n / N)
    return np.dot(M, x)/N if inverse else np.dot(M, x)


def FFT_recu(x, inverse: bool = False):
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

    X = np.concatenate([x_even + w[: N >> 1]*x_odd, x_even + w[N >> 1:]*x_odd])
    return X/N if inverse else X


def FFT_iter(x, inverse: bool = False):
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
        X_even = X[:, :X.shape[1] >> 1]
        X_odd = X[:, X.shape[1] >> 1:]
        factor = np.exp(-1j * tmp_PI *
                        np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
    return X.ravel()/N if inverse else X.ravel()


def ImgFFTYuki(img):
    x = np.zeros((img.shape[0], img.shape[1]), dtype=complex)
    for i in range(img.shape[0]):
        x[i] = (FFT_iter(img[i, 0:img.shape[1]]))
    for i in range(x.shape[1]):
        x[0:x.shape[0], i] = (FFT_iter(x[0:x.shape[0], i]))
    return x


def ImgFFTYukiv2(img, inverse: bool = False):
    x = np.zeros((img.shape[0], img.shape[1]), dtype=complex)

    for i, row in enumerate(img):
        x[i] = FFT_iter(row, inverse)

    x = x.T
    for i, col in enumerate(x):
        x[i] = FFT_iter(col, inverse)

    return x.T


def ImgFFTJason(data, inverse: bool = False):
    tmp = np.array(list(map(lambda row: FFT_iter(row), data)), dtype=complex)
    tmp = tmp.T
    tmp = np.array(list(map(lambda col: FFT_iter(col), tmp)), dtype=complex)
    tmp = tmp.T
    return tmp


def timeit(tgt_func, msg='', rpt=1):
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg}\n>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')


def rgb2gray(rgb):
    bw = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    return cv2.resize(bw, IMG_SIZE, interpolation=cv2.INTER_LINEAR)


def FFT_col(data):
    ESP = np.nextafter(np.float32(0), np.float32(1))
    return 20*np.log(ESP + np.abs(fftshift(data)))

def prepare_freq_info(msg:str = '', power:float = 99999):
    info = np.empty(IMG_SIZE, dtype=np.float64)
    cv2.putText(info, str(msg), (100,200), cv2.FONT_HERSHEY_DUPLEX, 2, power, 4)
    info += np.rot90(info, 2)
    return info

fig, ax = plt.subplots(4, 5)

# img = mpimg.imread('C:/Users/yukimura/Documents/Workplace/FFT_audio/squares.png')
# img = cv2.imread('C:/Users/yukimura/Documents/Workplace/FFT_audio/A6gibnM.jpg')
img = cv2.imread('./checker.png')
# img = cv2.imread('./A6gibnM.jpg')

img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
_BGR = ['Blues', 'Greens', 'Reds']


masksize_low = 150
mask_low = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
mask_low[256-masksize_low:256+masksize_low,
         256-masksize_low:256+masksize_low] = 1

mask_lowRGB = np.zeros(
    (img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
mask_lowRGB[256-masksize_low:256+masksize_low,
            256-masksize_low:256+masksize_low] = 1

masksize_high = 10
mask_high = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
mask_high[256-masksize_high:256+masksize_high,
          256-masksize_high:256+masksize_high] = 0

mask_highRGB = np.ones(
    (img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
mask_highRGB[256-masksize_high:256+masksize_high,
             256-masksize_high:256+masksize_high] = 0

imgf = fft2(rgb2gray(img))
imgyuki = ImgFFTYukiv2(rgb2gray(img))

ax[0, 0].imshow(img)
ax[1, 0].imshow(FFT_col(imgf), cmap='gray')
ax[2, 0].imshow(FFT_col(imgyuki), cmap='gray')


(R, G, B) = cv2.split(img)

for (i, col_ary), col_name in zip(enumerate([B, G, R]), _BGR):
    ax[i, 1].imshow(col_ary, cmap=col_name)

colf = [0,0,0]
colif = [0,0,0]
my_info = prepare_freq_info('password')
ixi = ImgFFTYukiv2(my_info, 1)
#cv2.imshow('', np.abs(ixi))

for (i, col_ary), col_name in zip(enumerate([B, G, R]), _BGR):

    colf[i] = ImgFFTYukiv2(col_ary + ixi)
    colif[i] = ImgFFTYukiv2(colf[i], 1)
    ax[i, 2].imshow(FFT_col(colf[i]), cmap=col_name)
    ax[i, 3].imshow(np.abs(colif[i]), cmap=col_name)


bfsci = fft2(B)
gfsci = fft2(G)
rfsci = fft2(R)
bifsci = ifft2(bfsci)
gifsci = ifft2(gfsci)
rifsci = ifft2(rfsci)

merge = np.dstack(colf)
mergesci = np.dstack((bfsci, gfsci, rfsci))
merge_col = FFT_col(merge)
mergesci_col = FFT_col(mergesci)

Rmask = ifftshift(fftshift(colif[2])*mask_high)
Gmask = ifftshift(fftshift(colif[1])*mask_high)
Bmask = ifftshift(fftshift(colif[0])*mask_high)
Rmaskif = ImgFFTYukiv2(Rmask, 1)
Gmaskif = ImgFFTYukiv2(Gmask, 1)
Bmaskif = ImgFFTYukiv2(Bmask, 1)
merge_mask = np.dstack((Rmaskif, Gmaskif, Bmaskif))

merge_if = np.dstack((colif[2], colif[1], colif[0]))
merge_ifsci = np.dstack((rifsci, gifsci, bifsci))
print(merge.shape, merge.dtype,  sep='\n')

# merge = rgb2gray(FFT_col(merge).astype(np.uint8))
# imgfft2 = rgb2gray(imgfft2.astype(np.uint8))
ax[0, 4].imshow(merge_col.astype(np.uint8))
ax[1, 4].imshow(mergesci_col.astype(np.uint8))
ax[2, 4].imshow(np.abs(merge_if).astype(np.uint8))

ax[3, 0].imshow((FFT_col(merge)*mask_highRGB).astype(np.uint8))
ax[3, 1].imshow(np.abs(Rmaskif).astype(np.uint8), cmap='Reds')
ax[3, 2].imshow(np.abs(Gmaskif).astype(np.uint8), cmap='Greens')
ax[3, 3].imshow(np.abs(Bmaskif).astype(np.uint8), cmap='Blues')
ax[3, 4].imshow(np.abs(merge_mask).astype(np.uint8))
plt.show()
'''
out = np.abs(merge_if).astype(np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imwrite("ipass.png", out)
'''

'''
# performance benchmark
timeit(lambda: ImgFFTJason(img), 'ImgFFTJason', 3)
timeit(lambda: ImgFFTYuki(img), 'ImgFFTYuki', 3)
timeit(lambda: ImgFFTYukiv2(img), 'ImgFFTYukiv2', 3)
'''

'''
# 1D FFT, complete
# x = np.linspace(0.0, N*T, N)
# y = 0.75*np.sin(5 * 2.0*_PI*x) + 0.5*np.sin(250 * 2.0*_PI*x) + 0.25*np.sin(400 * 2.0*_PI*x)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
yf = FFT_iter(y, False)
rec = FFT_iter(yf, True)

# To Compare
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

# performance benchmark
if N < 2048:
    timeit(lambda: DFT_slow(y), 'DFT_slow', 5)
else: print("DFT_slow\n>avg    >1000 ms per loop")
timeit(lambda: FFT_recu(y), 'FFT_recu', 10)
timeit(lambda: FFT_iter(y), 'FFT_iter', 10)
'''
