import cv2
import numpy

import api
import argparse
import os

# args
parser = argparse.ArgumentParser(
    description='tester for training data.')
parser.add_argument('weights', type=str, help='weights')
parser.add_argument('num_classes', type=int, default=None, help='num_classes')
parser.add_argument('image_size', type=int, default=128, help='image_size')
parser.add_argument('--cam', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--width', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--height', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--exposure', type=int, default=-7, help='the exposure level')
args = parser.parse_args()

# cam settings
cap = cv2.VideoCapture(args.cam)
if args.exposure:
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
if args.width and args.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

# loop vars
hsv_segmentation = api.HsvSegmentation()
hsv_segmentation.load()
i = 0
curr_class_idx = 0
cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)


def nothing(x):
    pass


# Creating track bar
cv2.createTrackbar('h', 'Controls', hsv_segmentation.col1[0], 179, nothing)
cv2.createTrackbar('s', 'Controls', hsv_segmentation.col1[1], 255, nothing)
cv2.createTrackbar('v', 'Controls', hsv_segmentation.col1[2], 255, nothing)
cv2.createTrackbar('h2', 'Controls', hsv_segmentation.col2[0], 179, nothing)
cv2.createTrackbar('s2', 'Controls', hsv_segmentation.col2[1], 255, nothing)
cv2.createTrackbar('v2', 'Controls', hsv_segmentation.col2[2], 255, nothing)

model = api.create_model(args.image_size, args.num_classes)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
if os.path.exists(args.weights):
    print("loading %s"%args.weights)
    model.load_weights(args.weights)

while True:
    try:
        # read
        _, frame = cap.read()
        frame_orig = frame.copy()

        # process
        h = cv2.getTrackbarPos('h', 'Controls')
        s = cv2.getTrackbarPos('s', 'Controls')
        v = cv2.getTrackbarPos('v', 'Controls')
        h2 = cv2.getTrackbarPos('h2', 'Controls')
        s2 = cv2.getTrackbarPos('s2', 'Controls')
        v2 = cv2.getTrackbarPos('v2', 'Controls')
        hsv_segmentation.col1 = (h, s, v)
        hsv_segmentation.col2 = (h2, s2, v2)
        mask = hsv_segmentation.apply(frame)
        biggest_contour = api.find_biggest_contour(mask)
        if biggest_contour is not None:
            mask_biggest_contour = api.create_mask_from_contour(mask, [biggest_contour])
            mask = api.blur_and_open_mask(mask_biggest_contour)
            # cv2.imshow("mask_with_contour", mask_with_contour)
        else:
            mask = None

        # write status
        if mask is not None:
            mask_resized = api.resize_image_if_necessary(mask, args.image_size)
            mask_resized_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            predictions = model.predict(numpy.array([mask_resized_color]))
            predicted_class_idx = int(numpy.argmax(predictions[0]))
            frame = api.mask_image(frame, mask)
            api.draw_text(frame, "predicted class: {}".format(predicted_class_idx))

        # show images
        cv2.imshow("Frame", frame)
        # cv2.imshow("mask", mask)
        # cv2.imshow("mask_biggest_contour", mask_biggest_contour)

        # process keys
        key = cv2.waitKeyEx(30)
        if key == 27:
            break
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise e

cap.release()
cv2.destroyAllWindows()
