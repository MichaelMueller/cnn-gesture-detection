import json
import logging
import sys
import tempfile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import cv2
import numpy as np
import os
import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

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
    _, contours, hierarchy = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        # cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        return c
    return None


def create_mask_from_contour(source_image, contours, binary_value=255):
    mask = np.zeros((source_image.shape[0], source_image.shape[1], 1), np.uint8)
    cv2.drawContours(mask, contours, -1, binary_value, -1)
    return mask


def mask_image(frame, mask):
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result


def draw_text(img, text, pos=(20, 20), bg_color=(255, 255, 255)):
    font_face = cv2.QT_FONT_NORMAL
    scale = 0.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 10

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


class HsvSegmentation:
    col1 = (0, 0, 0)
    col2 = (179, 255, 255)

    def __init__(self):
        self.cache_file = os.path.join(tempfile.gettempdir(), "HsvSegmentationValues.json")

    def save(self):
        data = [self.col1, self.col2]
        json_data = json.dumps(data, indent=4)
        with open(self.cache_file, 'w') as out_file:
            out_file.write(json_data)

    def load(self):
        if os.path.exists(self.cache_file) is False:
            return
        with open(self.cache_file) as in_file:
            json_data = json.load(in_file)
            self.col1 = json_data[0]
            self.col2 = json_data[1]

    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h = self.col1[0]
        s = self.col1[1]
        v = self.col1[2]
        h2 = self.col2[0]
        s2 = self.col2[1]
        v2 = self.col2[2]

        if h2 < h:
            lower = np.array([h, s, v])
            upper = np.array([179, s2, v2])
            mask1 = cv2.inRange(hsv, lower, upper)
            lower = np.array([0, s, v])
            upper = np.array([h2, s2, v2])
            mask2 = cv2.inRange(hsv, lower, upper)
            mask = cv2.add(mask1, mask2)
        else:
            lower = np.array([h, s, v])
            upper = np.array([h2, s2, v2])
            mask = cv2.inRange(hsv, lower, upper)
        return mask

def create_model(image_size, nClasses):

    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    optimizer1 = optimizers.Adam()

    base_model = vgg_base  # Topless
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='fc4')(x)
    predictions = Dense(nClasses, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train top layers only
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def resize_image_if_necessary(img, image_size):
    if img.shape[1] != image_size or img.shape[0] != image_size:
        # print("resizing image to %sx%s" % (expected_input_size[0], expected_input_size[1]))
        return cv2.resize(img, (image_size, image_size))
    else:
        return img.copy()

def load_images_masks(dpath, use_masks=False, image_size=128, divisor=255.0):
    images =[]
    labels = []
    for fpath in process_dir(dpath, common_image_file_extensions):
        class_name, is_mask = get_img_info(fpath)
        if use_masks and is_mask is False:
            print("Skipping {} because it is no mask".format(fpath))
            continue
        elif use_masks is False and is_mask is True:
            print("Skipping {} because it is a mask".format(fpath))
            continue

        img = cv2.imread(fpath)

        if img.shape[1] != image_size or img.shape[0] != image_size:
            #print("resizing image to %sx%s" % (expected_input_size[0], expected_input_size[1]))
            img = cv2.resize(img, (image_size, image_size))

        images.append(img)
        labels.append(int(class_name))

    images = np.array(images)
    labels = np.array(labels)
    if divisor:
        images = images / divisor
    return images, to_categorical(labels)

def get_img_info(fpath):
    parts = fpath.split('.')
    class_name = None
    is_mask = False
    parts_len = len(parts)
    if parts_len > 2:
        if parts[parts_len - 2] == "mask":
            is_mask = True
            if parts_len > 3:
                class_name = parts[parts_len - 3]
        else:
            class_name = parts[parts_len - 2]
    return class_name, is_mask


def blur_and_open_mask(mask, gaussian_blur_size=3, opening_kernel_size=3):
    mask_blurred = cv2.GaussianBlur(mask, (gaussian_blur_size, gaussian_blur_size), 0)
    kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)

    mask_openend = cv2.morphologyEx(mask_blurred, cv2.MORPH_OPEN, kernel)
    return mask_openend


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
