import numpy as np
import cv2
import time
from scipy.fftpack import fftshift

def timeit(tgt_func, msg='', rpt=1):
    t = 0
    start = time.time()
    for _ in range(rpt):
        tgt_func()
    stop = time.time()
    print(f'{msg}\n>avg {(stop - start)*1000/rpt : 8.3f} ms per loop')


def rgb2gray(rgb, IMG_SIZE:tuple = (256, 256)):
    bw = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    return cv2.resize(bw, IMG_SIZE, interpolation=cv2.INTER_LINEAR)


def FFT_col(data, shift:bool = True):
    ESP = np.nextafter(np.float32(0), np.float32(1))
    if shift:
        return 20*np.log(ESP + np.abs(fftshift(data)))
    else:
        return 20*np.log(ESP + np.abs(data))

def prepare_freq_info(IMG_SIZE:tuple = (256, 256), msg:str = '', power:float = 99999):
    info = np.empty(IMG_SIZE, dtype=np.float64)
    cv2.putText(info, str(msg), (100,200), cv2.FONT_HERSHEY_DUPLEX, 2, power, 4)
    info += np.rot90(info, 2)
    return info