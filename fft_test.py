import matplotlib.pyplot as plt
import numpy as np
import cv2
# our library
from fft_func import ImgFFTYukiv2, yuki_shift, yuki_ishift, FFT_col
from util_func import *
# check answer
from scipy.fftpack import fft2, ifft2

IMG_SIZE = (256, 256)
MASK_SIZE = 8
# LowPASS:1 HighPass:0 
MASK_MODE = 0
# square: 's', circle: 'c', NoFilter: ''
MASK_SHAPE = 'c'

fig, ax = plt.subplots(4, 6)
SRC = 'images/checker.png'
# lol no need to change this anymore
try:
    img = cv2.imread('C:/Users/yukimura/Documents/Workplace/FFT_image/'+SRC)
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
except:
    img = cv2.imread('./'+SRC)
    img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)

_RGB = ['Reds', 'Greens', 'Blues']


imgf = fft2(rgb2gray(img, IMG_SIZE))
imgyuki = ImgFFTYukiv2(rgb2gray(img, IMG_SIZE))
# grey scale fft
ax[0, 0].imshow(img)


(R, G, B) = cv2.split(img)

#not used
#my_info = prepare_freq_info(IMG_SIZE, 'password')
#ixi = ImgFFTYukiv2(my_info, 1)

mask = getMask(IMG_SIZE, MASK_SIZE, MASK_MODE, MASK_SHAPE)
#+ getMask(IMG_SIZE, 80, 0)
maskRGB = getMaskRGB(IMG_SIZE, MASK_SIZE, MASK_MODE, MASK_SHAPE)
#+ getMaskRGB(IMG_SIZE, 80, 0)
# print(mask)

colf = [0, 0, 0]
colif = [0, 0, 0]

masked_colf = [0, 0, 0]
masked_colif = [0, 0, 0]

for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):

    colf[i] = ImgFFTYukiv2(col_ary)
    masked_colf[i] = yuki_ishift(yuki_shift(colf[i]) * mask)

    colif[i] = ImgFFTYukiv2(colf[i], 1)
    masked_colif[i] = ImgFFTYukiv2(masked_colf[i], 1)

    ax[i, 1].imshow(col_ary, cmap=col_name)
    ax[i, 2].imshow(FFT_col(colf[i]), cmap=col_name, vmin = 0, vmax = 200)
    ax[i, 3].imshow(np.abs(colif[i]), cmap=col_name)
    ax[i, 4].imshow(FFT_col(masked_colf[i]), cmap=col_name, vmin = 0, vmax = 200)
    ax[i, 5].imshow(np.abs(masked_colif[i]), cmap=col_name)

merge = np.dstack(colf)
#merge_col = FFT_col(merge)
merge_if = np.dstack(colif)

merge_maskf = np.dstack(masked_colf)
#merge_mask_col = FFT_col(merge_maskf)
merge_mask_if = np.dstack(masked_colif)

scif = [0, 0, 0]
sciif = [0, 0, 0]

for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):

    scif[i] = fft2(col_ary)
    sciif[i] = ifft2(scif[i])

mergesci = np.dstack(scif)
#mergesci_col = FFT_col(mergesci)
merge_ifsci = np.dstack(sciif)
#print(merge.shape, merge.dtype,  sep='\n')

ax[1, 0].imshow(FFT_col(imgf), cmap='gray')
ax[2, 0].imshow(FFT_col(mergesci).astype(np.uint8))
ax[3, 0].imshow(np.abs(merge_ifsci).astype(np.uint8))

ax[3, 1].imshow(FFT_col(imgyuki), cmap='gray')
ax[3, 2].imshow(FFT_col(merge).astype(np.uint8))
# mask result
ax[3, 3].imshow(np.abs(merge_if).astype(np.uint8))
ax[3, 4].imshow(FFT_col(merge_maskf).astype(np.uint8))
ax[3, 5].imshow(np.abs(merge_mask_if).astype(np.uint8))
plt.show()
'''
out = np.abs(merge_if).astype(np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imwrite("ipass.png", out)
'''