import numpy as np
import cv2
import time
#from scipy.fftpack import fftshift, ifftshift


def timeit(tgt_func, msg='', rpt=1):
    '''Timer function, usage : timeit(lambda: FFT_recu(y), 'FFT_recu', 10)'''
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg} * {rpt}...')
    print(f'>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')


def rgb2gray(rgb, IMG_SIZE: tuple = (256, 256)):
    '''convert RGB(3 channels) to greyscale'''
    bw = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    return cv2.resize(bw, IMG_SIZE, interpolation=cv2.INTER_LINEAR)


def prepare_freq_info(IMG_SIZE: tuple = (256, 256), msg: str = '', power: float = 99999):
    '''this was used to hide info in freq. domain, not used.'''
    info = np.empty(IMG_SIZE, dtype=np.float64)
    cv2.putText(info, str(msg), (100, 200),
                cv2.FONT_HERSHEY_DUPLEX, 1, power, 2)
    info += np.rot90(info, 2)
    return info


def getMask(img_size: tuple, masksize, ishigh: bool, m_shape: str):
    '''mask shape(x, y)'''
    img_center = (img_size[0] >> 1, img_size[1] >> 1)
    not_high = not ishigh
    mask = np.ones(img_size, dtype=np.uint8)

    #circle
    if m_shape == 'c':
        Y, X = np.ogrid[:img_size[0], :img_size[1]]
        dist_from_center = (X - img_center[0])**2 + (Y-img_center[1])**2
        sq_r = masksize**2
        mask = dist_from_center <= sq_r if ishigh else dist_from_center > sq_r
        return mask
    #square
    elif m_shape == 's':
        mask.fill(not_high)
        mask[img_center[0]-masksize: img_center[0]+masksize,
            img_center[1]-masksize: img_center[1]+masksize] = ishigh
        return mask
    else: return mask

def getMaskRGB(img_size: tuple, masksize, ishigh: bool, m_shape: str):
    '''mask shape(x, y, 3)'''
    img_center = (img_size[0] >> 1, img_size[1] >> 1)
    not_high = not ishigh
    maskRGB = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8)
    #circle
    if m_shape == 'c':
        Y, X = np.ogrid[:img_size[0], :img_size[1]]
        dist_from_center = (X - img_center[0])**2 + (Y-img_center[1])**2
        sq_r = masksize**2
        maskRGB = dist_from_center <= sq_r if ishigh else dist_from_center > sq_r
        return np.dstack((maskRGB, maskRGB, maskRGB)) 
    #square
    elif m_shape == 's':
        maskRGB.fill(not_high)
        maskRGB[img_center[0]-masksize: img_center[0]+masksize,
                img_center[1]-masksize: img_center[1]+masksize] = ishigh
        return maskRGB
    else: return maskRGB
