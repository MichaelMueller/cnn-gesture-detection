import cv2
import numpy as np


class BackgroundSubstraction:
    threshold_val = 35
    gaussian_blur_size = 3
    opening_kernel_size = 3

    def __init__(self, first_frame, equalize_hist=True):
        self.first_gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.equalize_hist = equalize_hist
        if self.equalize_hist:
            cv2.equalizeHist(self.first_gray_frame, self.first_gray_frame)

    def apply(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize_hist:
            cv2.equalizeHist(gray_frame,gray_frame)
        if self.gaussian_blur_size:
            gray_frame = cv2.GaussianBlur(gray_frame, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        difference = cv2.absdiff(self.first_gray_frame, gray_frame)
        _, difference = cv2.threshold(difference, self.threshold_val, 255, cv2.THRESH_BINARY)

        if self.opening_kernel_size:
            kernel = np.ones((self.opening_kernel_size, self.opening_kernel_size), np.uint8)

            difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

        return difference