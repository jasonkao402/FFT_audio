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
#grey scale fft
ax[0, 0].imshow(img)
ax[1, 0].imshow(FFT_col(imgf), cmap='gray')
ax[2, 0].imshow(FFT_col(imgyuki), cmap='gray')


(R, G, B) = cv2.split(img)

my_info = prepare_freq_info(IMG_SIZE, 'password')
ixi = ImgFFTYukiv2(my_info, 1)

mask = getMask(IMG_SIZE, 20, 0)
#+ getMask(IMG_SIZE, 80, 0)
maskRGB = getMaskRGB(IMG_SIZE, 20, 0)
#+ getMaskRGB(IMG_SIZE, 80, 0)
#print(mask)

colf = [0, 0, 0]
colif = [0, 0, 0]

masked_colf=[0, 0, 0]
masked_colif=[0, 0, 0]

for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):

    colf[i] = ImgFFTYukiv2(col_ary)
    masked_colf[i] = yuki_ishift(yuki_shift(colf[i]) * mask)

    colif[i] = ImgFFTYukiv2(colf[i], 1)
    masked_colif[i] = ImgFFTYukiv2(masked_colf[i], 1)

    ax[i, 1].imshow(col_ary, cmap=col_name)
    ax[i, 2].imshow(FFT_col(colf[i]), cmap=col_name)
    ax[i, 3].imshow(np.abs(colif[i]), cmap=col_name)
    ax[i, 4].imshow(FFT_col(masked_colf[i]).astype(np.uint8), cmap=col_name)
    ax[i, 5].imshow(np.abs(masked_colif[i]), cmap=col_name)

merge = np.dstack(colf)
merge_col = FFT_col(merge)
merge_if = np.dstack(colif)

merge_maskf = np.dstack(masked_colf)
merge_mask_col = FFT_col(merge_maskf)
merge_mask_if = np.dstack(masked_colif)

scif = [0, 0, 0]
sciif = [0, 0, 0]

for (i, col_ary), col_name in zip(enumerate([R, G, B]), _RGB):

    scif[i] = fft2(col_ary)
    sciif[i] = ifft2(scif[i])

mergesci = np.dstack(scif)
mergesci_col = FFT_col(mergesci)
merge_ifsci = np.dstack(sciif)
#print(merge.shape, merge.dtype,  sep='\n')

ax[3, 0].imshow((FFT_col(merge) * maskRGB).astype(np.uint8))
ax[3, 1].imshow(merge_col.astype(np.uint8))
ax[3, 2].imshow(mergesci_col.astype(np.uint8))
#result
ax[3, 3].imshow(np.abs(merge_if).astype(np.uint8))
ax[3, 4].imshow(FFT_col(merge_maskf).astype(np.uint8))
ax[3, 5].imshow(np.abs(merge_mask_if).astype(np.uint8))
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
