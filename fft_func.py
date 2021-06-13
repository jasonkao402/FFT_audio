import numpy as np

_PI = np.math.pi

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

def getLowMask(img_size:tuple, masksize_low):
    img_center = (img_size[0]>>1, img_size[1]>>1)

    mask_low = np.zeros(img_size, dtype=np.uint8)
    mask_low[img_center[0]-masksize_low : img_center[0]+masksize_low,
             img_center[1]-masksize_low : img_center[1]+masksize_low] = 1
    return mask_low


def getLowRGBMask(img_size:tuple, masksize_low):
    img_center = (img_size[0]>>1, img_size[1]>>1)

    mask_lowRGB = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    mask_lowRGB[img_center[0]-masksize_low : img_center[0]+masksize_low,
                img_center[1]-masksize_low : img_center[1]+masksize_low] = 1
    return mask_lowRGB


def getHighMask(img_size:tuple, masksize_high):
    img_center = (img_size[0]>>1, img_size[1]>>1)

    mask_high = np.zeros(img_size, dtype=np.uint8)
    mask_high.fill(1)
    mask_high[img_center[0]-masksize_high : img_center[0]+masksize_high,
             img_center[1]-masksize_high : img_center[1]+masksize_high] = 0
    return mask_high


def getHighRGBMask(img_size:tuple, masksize_high):
    img_center = (img_size[0]>>1, img_size[1]>>1)

    mask_highRGB = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    mask_high.fill(1)
    mask_highRGB[img_center[0]-masksize_high : img_center[0]+masksize_high,
                img_center[1]-masksize_high : img_center[1]+masksize_high] = 0
    return mask_highRGB


def yuki_shift(data):
    row = data.shape[0]
    col = data.shape[1]
    data[0:int(row/2)] = data[int(row/2):row]
    data[0:int(col/2)] = data[int(col/2):col]
    return data


def yuki_ishift(data):
    row = data.shape[0]
    col = data.shape[1]
    data[0:int(row/2)] = data[int(row/2):row]
    data[0:int(col/2)] = data[int(col/2):col]
    return data
'''
# 1D FFT, complete

# Number of sample points
N = 1024
# sample spacing
T = 1 / N

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