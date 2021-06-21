import numpy as np

# constant
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
    # This cutoff should be optimized
    # as recursion on small N does not help much
    # rather do it in one N^2 call.
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
    # This cutoff should be optimized
    # as recursion on small N does not help much
    # rather do it in one N^2 call.
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


def FFT_iter_v2(x, inverse: bool = False):
    """Iter. version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=complex)
    tmp_PI = -_PI if inverse else _PI
    N = x.shape[0]

    if (N & (N - 1)):
        raise ValueError("size of x must be a power of 2")
    # this cutoff should be optimized
    # as recursion on small N does not help much
    # rather do it in one N^2 call.
    N_min = min(N, 16)
    tmp_2iPI = -2j * tmp_PI / N_min

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(tmp_2iPI * n * k)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] >> 1]
        X_odd = X[:, X.shape[1] >> 1:]
        factor = np.exp(-1j * tmp_PI * np.arange(X.shape[0]) / X.shape[0])[:, None]
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

def ImgFFTYukiv3(img, inverse: bool = False):
    x = np.zeros((img.shape[0], img.shape[1]), dtype=complex)

    for i, row in enumerate(img):
        x[i] = FFT_iter_v2(row, inverse)

    x = x.T
    for i, col in enumerate(x):
        x[i] = FFT_iter_v2(col, inverse)

    return x.T

def ImgFFTJason(data, inverse: bool = False):
    tmp = np.array(list(map(lambda row: FFT_iter(row), data)), dtype=complex)
    tmp = tmp.T
    tmp = np.array(list(map(lambda col: FFT_iter(col), tmp)), dtype=complex)
    tmp = tmp.T
    return tmp


def FFT_col(data):
    '''This function is only used when applying color on the "frequency domain!"'''
    return 20*np.log(1 + np.abs(yuki_shift(data)))


def yuki_shift(data):
    row = data.shape[0]
    col = data.shape[1]
    data = np.hstack((data[0:row, int(col/2):col],
                     data[0:row, 0:int(col/2)]))
    data = np.vstack((data[int(row/2):row], data[0:int(row/2)]))
    return data


def yuki_ishift(data):
    row = data.shape[0]
    col = data.shape[1]
    data = np.hstack((data[0:row, int(col/2):col],
                     data[0:row, 0:int(col/2)]))
    data = np.vstack((data[int(row/2):row], data[0:int(row/2)]))
    return data

