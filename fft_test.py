import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from fft_func import *
from util_func import *
from scipy.fftpack import fft2, ifft2
import cv2

IMG_SIZE = (256, 256)

fig, ax = plt.subplots(4, 6)
SRC = 'checker.png'
#lol no need to change this anymore
try:
    img = cv2.imread('C:/Users/yukimura/Documents/Workplace/FFT_audio/'+SRC)
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
except:
    img = cv2.imread('./'+SRC)
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

_RGB = ['Reds', 'Greens', 'Blues']


imgf = fft2(rgb2gray(img, IMG_SIZE))
imgyuki = ImgFFTYukiv2(rgb2gray(img, IMG_SIZE))

ax[0, 0].imshow(img)
ax[1, 0].imshow(FFT_col(imgf), cmap='gray')
ax[2, 0].imshow(FFT_col(imgyuki), cmap='gray')


(R, G, B) = cv2.split(img)

my_info = prepare_freq_info(IMG_SIZE, 'password')
ixi = ImgFFTYukiv2(my_info, 1)

colf = [0, 0, 0]
colif = [0, 0, 0]

for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):
    colf[i] = ImgFFTYukiv2(col_ary)
    colif[i] = ImgFFTYukiv2(colf[i], 1)
    ax[i, 1].imshow(col_ary, cmap=col_name)
    ax[i, 2].imshow(FFT_col(colf[i]), cmap=col_name)
    ax[i, 3].imshow(np.abs(colif[i]), cmap=col_name)

merge = np.dstack((colf[0], colf[1], colf[2]))
merge_col = FFT_col(merge)
merge_if = np.dstack((colif[0], colif[1], colif[2]))

scif = [0, 0, 0]
sciif = [0, 0, 0]
for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):

    scif[i] = fft2(col_ary)
    sciif[i] = ifft2(scif[i])

mergesci = np.dstack(scif)
mergesci_col = FFT_col(mergesci)

mask = getMask(IMG_SIZE, 100, 0)
#+ getMask(IMG_SIZE, 80, 0)
maskRGB = getMaskRGB(IMG_SIZE, 100, 0)
#+ getMaskRGB(IMG_SIZE, 80, 0)

print(mask)

Rmask = ifftshift(fftshift(colf[0]) * mask)
Gmask = ifftshift(fftshift(colf[1]) * mask)
Bmask = ifftshift(fftshift(colf[2]) * mask)
Rmaskif = ImgFFTYukiv2(Rmask, 1)
Gmaskif = ImgFFTYukiv2(Gmask, 1)
Bmaskif = ImgFFTYukiv2(Bmask, 1)

ax[0, 5].imshow(FFT_col(Rmask), cmap='Reds')
ax[1, 5].imshow(FFT_col(Gmask), cmap='Greens')
ax[2, 5].imshow(FFT_col(Bmask), cmap='Blues')

merge_mask = np.dstack((Rmaskif, Gmaskif, Bmaskif))


merge_ifsci = np.dstack((sciif[0], sciif[1], sciif[2]))
#print(merge.shape, merge.dtype,  sep='\n')

ax[0, 4].imshow(merge_col.astype(np.uint8))
ax[1, 4].imshow(mergesci_col.astype(np.uint8))
ax[2, 4].imshow(np.abs(merge_if).astype(np.uint8))


ax[3, 0].imshow((FFT_col(merge) * maskRGB).astype(np.uint8))
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
