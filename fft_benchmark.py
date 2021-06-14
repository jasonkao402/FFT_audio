import matplotlib.pyplot as plt
from util_func import timeit
from fft_func import *

# 1D FFT, complete
if __name__ == '__main__':
    #constant
    _PI = np.math.pi
    # Number of sample points
    N = 2048*4
    # sample spacing
    T = 1 / N

    x = np.linspace(0, 1, N)
    y = 0.75*np.sin(5 * 2 * _PI * x)# + 0.5*np.sin(250 * 2 * _PI * x) + 0.25*np.sin(400 * 2 * _PI * x)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = FFT_iter_v2(y, False)
    rec = FFT_iter_v2(yf, True)

    # To Compare
    #newyf=fft(y)
    #newrec=ifft(newyf)
    #print(np.allclose(yf, newyf))
    print(np.allclose(rec, y))
    
    # 1D performance benchmark
    '''
    if N < 2048:
        timeit(lambda: DFT_slow(y), 'DFT_slow', 5)
    else: print("DFT_slow\n>avg    >1000 ms per loop")
    if N < 262144:
        timeit(lambda: FFT_recu(y), 'FFT_recu', 8)
    else: print("FFT_recu\n>avg    >1000 ms per loop")
    '''
    '''
    timeit(lambda: FFT_iter(y), 'FFT_iter_v1', 100)
    timeit(lambda: FFT_iter_v2(y), 'FFT_iter_v2', 100)
    timeit(lambda: FFT_iter(yf, 1), 'iFFT_iter_v1', 100)
    timeit(lambda: FFT_iter_v2(yf, 1), 'iFFT_iter_v2', 100)
    '''
    img = np.random.rand(256, 256)
    imgfd = ImgFFTYukiv2(img)
    imgsd = ImgFFTYukiv2(imgfd, 1)
    # 2D performance benchmark
    #timeit(lambda: ImgFFTJason(img), 'ImgFFTJason', 32)
    timeit(lambda: ImgFFTYuki(img), 'ImgFFTYuki', 256)
    timeit(lambda: ImgFFTYukiv2(img), 'ImgFFTYukiv2', 256)
    
    #show plot
    '''
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(x, y)
    ax[1, 0].plot(xf, np.abs(yf[:N//2]))
    ax[2, 0].plot(x, rec.real)
    ax[0, 1].imshow(np.abs(img), cmap = 'gray')
    ax[1, 1].imshow(FFT_col(imgfd), cmap = 'gray')
    ax[2, 1].imshow(np.abs(imgsd), cmap = 'gray')
    plt.show()
    '''
