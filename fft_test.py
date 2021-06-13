import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from fft_func import *
from util_func import *
from scipy.fftpack import fft2, ifft2
import cv2

IMG_SIZE = (256, 256)

fig, ax = plt.subplots(4, 6)
SRC = 'wave.png'
#lol no need to change this anymore
img = cv2.imread('./'+SRC)
if not img.any():
    img = cv2.imread('C:/Users/yukimura/Documents/Workplace/FFT_audio/'+SRC)

img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
_BGR = ['Blues', 'Greens', 'Reds']


imgf = fft2(rgb2gray(img, IMG_SIZE))
imgyuki = ImgFFTYukiv2(rgb2gray(img, IMG_SIZE))

ax[0, 0].imshow(img)
ax[1, 0].imshow(FFT_col(imgf), cmap='gray')
ax[2, 0].imshow(FFT_col(imgyuki), cmap='gray')


(R, G, B) = cv2.split(img)


for (i, col_ary), col_name in zip(enumerate([B, G, R]), _BGR):
    ax[i, 1].imshow(col_ary, cmap=col_name)

my_info = prepare_freq_info(IMG_SIZE, 'password')
ixi = ImgFFTYukiv2(my_info, 1)

colf = [0,0,0]
colif = [0,0,0]

for (i, col_ary), col_name in zip(enumerate([B, G, R]), _BGR):

    colf[i] = ImgFFTYukiv2(col_ary)
    colif[i] = ImgFFTYukiv2(colf[i], 1)
    ax[i, 2].imshow(FFT_col(colf[i]), cmap=col_name)
    ax[i, 3].imshow(np.abs(colif[i]), cmap=col_name)

merge = np.dstack(colf)
merge_col = FFT_col(merge)
merge_if = np.dstack((colif[2], colif[1], colif[0]))

scif = [0,0,0]
sciif = [0,0,0]
for (i, col_ary), col_name in zip(enumerate([B, G, R]), _BGR):

    scif[i] = fft2(col_ary)
    sciif[i] = ifft2(scif[i])

mergesci = np.dstack(scif)
mergesci_col = FFT_col(mergesci)

mask = getMask(IMG_SIZE, 100, 1) + getMask(IMG_SIZE, 80, 0)
maskRGB = getMaskRGB(IMG_SIZE, 100, 1) + getMaskRGB(IMG_SIZE, 80, 0)

Rmask = ifftshift(fftshift(colif[2]) * mask)
Gmask = ifftshift(fftshift(colif[1]) * mask)
Bmask = ifftshift(fftshift(colif[0]) * mask)
Rmaskif = ImgFFTYukiv2(Rmask, 1)
Gmaskif = ImgFFTYukiv2(Gmask, 1)
Bmaskif = ImgFFTYukiv2(Bmask, 1)

merge_mask = np.dstack((Rmaskif, Gmaskif, Bmaskif))


merge_ifsci = np.dstack((sciif[2], sciif[1], sciif[0]))
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
