import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def get_kernel(ksize, sigmaX, sigmaY):
    kernelX = cv2.getGaussianKernel(ksize, sigmaX)
    kernelY = cv2.getGaussianKernel(ksize, sigmaY)
    return np.outer(kernelY, kernelX.transpose())


def get_gaussian_blur(img, kernel):
    h, w, c = img.shape
    return cv2.filter2D(img, -1, kernel).reshape(h, w, c)


def downsample(img, kernel, inter, scale=2):
    img = get_gaussian_blur(img, kernel)
    h, w, c = img.shape
    if inter == 'nearest':
        inter_cv = cv2.INTER_NEAREST
    elif inter == 'linear':
        inter_cv = cv2.INTER_LINEAR
    elif inter == 'cubic':
        inter_cv = cv2.INTER_CUBIC
    img = np.array(cv2.resize(img, (w // scale, h // scale), inter_cv))
    if len(img.shape)==3:
        return img
    else:
        return np.expand_dims(img, axis=2)
