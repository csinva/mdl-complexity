import requests
from tqdm import tqdm
from os.path import join as oj
import tables, numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
import h5py
from copy import deepcopy
from skimage.filters import gabor_kernel
out_dir = '/scratch/users/vision/data/gallant/vim_2_crcns'

def calc_feats(image, kernel):
    '''power calculation
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    '''
    mag = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                  ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    return np.log(1 + mag)

def all_feats(im, DOWNSAMPLE=2):
    # prepare filter bank kernels
    feat_vec = []
    for frequency in [0.05, 0.25]:
        kernels = []
        for sigma in (1, 3):
            for theta in range(4):
                theta = theta / 4. * np.pi
                kernel = gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma)
                kernels.append(kernel)

        for kernel in kernels:
            feats = calc_feats(im, kernel)
            
            if frequency == 0.05:
                feats = feats[::DOWNSAMPLE * 2, ::DOWNSAMPLE * 2]
            elif frequency == 0.25:
                feats = feats[::DOWNSAMPLE, ::DOWNSAMPLE]
        feat_vec.append(deepcopy(feats.flatten()))
    return np.hstack(feat_vec)