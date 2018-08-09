import numpy as np
import cv2


def region_mask(img_channel, percent_region):
    pass


def abs_sobel_thresh(img_channel, orient='x', ksize=3, thresh_min=0, thresh_max=255):
    if orient is 'x':
        direction = [1, 0]
    elif orient is 'y':
        direction = [0, 1]
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, direction[0], direction[1],ksize=ksize)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


def mag_thresh(img_channel, ksize=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img_channel, ksize=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output