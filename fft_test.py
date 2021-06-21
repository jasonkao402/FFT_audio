import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# our library
from fft_func import ImgFFTYukiv2, yuki_shift, yuki_ishift, FFT_col
from util_func import *
# check answer
from scipy.fftpack import fft2, ifft2

# only power of 2
# higher means better resolution, at cost of slower
IMG_SIZE = (256, 256)
# should not go over half the img size
MASK_SIZE = 6
# LowPASS:1 HighPass:0 
MASK_MODE = 1
# square: 's', circle: 'c', NoFilter: ''
MASK_SHAPE = 'c'

_RGB = ['Reds', 'Greens', 'Blues']

# lol no need to change this anymore
absFilePath = os.path.abspath(__file__)
os.chdir( os.path.dirname(absFilePath))
SRC = './images/checker.png'
RESULT = './result/'

# read start, do not change under this line
img = cv2.imread(SRC)
img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(4, 6)

# grey scale fft
img_sci_bw = fft2(rgb2gray(img, IMG_SIZE))
imgyuki_bw = ImgFFTYukiv2(rgb2gray(img, IMG_SIZE))
ax[0, 0].imshow(img)

(R, G, B) = cv2.split(img)

# not used
#my_info = prepare_freq_info(IMG_SIZE, 'password')
#ixi = ImgFFTYukiv2(my_info, 1)

mask = getMask(IMG_SIZE, MASK_SIZE, MASK_MODE, MASK_SHAPE)
maskRGB = getMaskRGB(IMG_SIZE, MASK_SIZE, MASK_MODE, MASK_SHAPE)
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

    cache = [col_ary, FFT_col(colf[i]), np.abs(colif[i]), FFT_col(masked_colf[i]), np.abs(masked_colif[i])]
    ax[i, 1].imshow(cache[0], cmap=col_name)
    ax[i, 2].imshow(cache[1], cmap=col_name, vmin = 0, vmax = 225)
    ax[i, 3].imshow(cache[2], cmap=col_name)
    ax[i, 4].imshow(cache[3], cmap=col_name, vmin = 0, vmax = 225)
    ax[i, 5].imshow(cache[4], cmap=col_name)

    plt.imsave(f'{RESULT}color_{col_name}.jpg', cache[0], cmap = col_name)
    plt.imsave(f'{RESULT}FFT_{col_name}.jpg', cache[1], cmap=col_name, vmin = 0, vmax = 225)
    plt.imsave(f'{RESULT}mask_FFT_{col_name}.jpg', cache[3], cmap=col_name, vmin = 0, vmax = 225)
    plt.imsave(f'{RESULT}mask_IFFT_{col_name}.jpg', cache[4], cmap=col_name)


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

ax[1, 0].imshow(FFT_col(img_sci_bw), cmap='gray')
ax[2, 0].imshow(FFT_col(mergesci).astype(np.uint8))
ax[3, 0].imshow(np.abs(merge_ifsci).astype(np.uint8))

ax[3, 1].imshow(FFT_col(imgyuki_bw), cmap='gray')
ax[3, 2].imshow(FFT_col(merge).astype(np.uint8))
# mask result
ax[3, 3].imshow(np.abs(merge_if).astype(np.uint8))
ax[3, 4].imshow(FFT_col(merge_maskf).astype(np.uint8))
ax[3, 5].imshow(np.abs(merge_mask_if).astype(np.uint8))
plt.show()

plt.imsave(f'{RESULT}FFT_grey.jpg', FFT_col(imgyuki_bw), cmap='gray')
plt.imsave(f'{RESULT}FFT_merge.jpg', FFT_col(merge).astype(np.uint8))
plt.imsave(f'{RESULT}mask_FFT.jpg', FFT_col(merge_maskf).astype(np.uint8))
plt.imsave(f'{RESULT}mask_IFFT.jpg', np.abs(merge_mask_if).astype(np.uint8))
'''
out = np.abs(merge_if).astype(np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imwrite("ipass.png", out)
'''