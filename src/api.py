import logging
import sys

import cv2
import numpy as np
import os

common_image_file_extensions = ["png", "bmp", "jpeg", "jpg"]

def process_dir(start_dir, extensions=None):
    if extensions is None:
        extensions = []
    for root, dirs, files in os.walk(start_dir):
        for basename in files:
            file_name, ext = os.path.splitext(basename)
            ext = ext[1:].lower()
            if len(extensions) > 0 and ext not in extensions:
                continue
            path = os.path.join(root, basename)
            yield path


def setup_logging(log_level, log_file=None):
    class InfoFilter(logging.Filter):
        def filter(self, rec):
            return rec.levelno in (logging.DEBUG, logging.INFO, logging.WARNING)

    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(logging.DEBUG)
    h1.addFilter(InfoFilter())
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.ERROR)

    handlers = [h1, h2]
    kwargs = {"format": "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
              "datefmt": '%Y-%m-%d:%H:%M:%S', "level": log_level}

    if log_file:
        h1 = logging.FileHandler(filename=log_file)
        h1.setLevel(logging.DEBUG)
        handlers = [h1]

    kwargs["handlers"] = handlers
    logging.basicConfig(**kwargs)

def find_biggest_contour(thresholded_image):
    contours, hierarchy = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        #cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        return c
    return None

def create_mask_from_contour(source_image, contours, binary_value=255):
    mask = np.zeros((source_image.shape[0], source_image.shape[1], 1), np.uint8)
    cv2.drawContours(mask, contours, -1, binary_value, -1)
    return mask

class BackgroundSubstraction:
    threshold_val = 15
    gaussian_blur_size = 7
    opening_kernel_size = 13

    def __init__(self, first_frame, equalize_hist=True):
        self.first_gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.equalize_hist = equalize_hist
        if self.equalize_hist:
            cv2.equalizeHist(self.first_gray_frame, self.first_gray_frame)

    def apply(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize_hist:
            cv2.equalizeHist(gray_frame, gray_frame)
        if self.gaussian_blur_size:
            gray_frame = cv2.GaussianBlur(gray_frame, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        difference = cv2.absdiff(self.first_gray_frame, gray_frame)
        _, difference = cv2.threshold(difference, self.threshold_val, 255, cv2.THRESH_BINARY)

        if self.opening_kernel_size:
            kernel = np.ones((self.opening_kernel_size, self.opening_kernel_size), np.uint8)

            difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

        return difference
