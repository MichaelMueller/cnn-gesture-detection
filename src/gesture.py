import cv2
import numpy

import api
import argparse
import os
import hands_mask_rcnn
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# args
parser = argparse.ArgumentParser(
    description='Recorder for training data.')
parser.add_argument('mask_rcnn_model_file', type=str, help='the exposure level')
parser.add_argument('--mask_rcnn_confidence', type=float, default=0.7, help='the exposure level')
parser.add_argument('--cam', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--width', type=int, default=None, help='the number of the camera to use')
parser.add_argument('--height', type=int, default=None, help='the number of the camera to use')
parser.add_argument('--exposure', type=int, default=-7, help='the exposure level')
args = parser.parse_args()

# cam settings
cap = cv2.VideoCapture(args.cam)
if args.exposure:
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
if args.width and args.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

# model loading
print("Loading rcnn")


class InferenceConfig(hands_mask_rcnn.HandsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.DETECTION_MIN_CONFIDENCE = args.mask_rcnn_confidence
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=".")

print("Loading weights ", args.mask_rcnn_model_file)
model.load_weights(args.mask_rcnn_model_file, by_name=True)

while True:
    try:
        # read
        _, frame = cap.read()
        frame_orig = frame.copy()

        # detection
        print("running detection")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = model.detect([frame_rgb], verbose=1)[0]
        mask = r['masks']
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (numpy.sum(mask, -1, keepdims=True) >= 1).astype(numpy.uint8)
            mask[mask > 0] = 1
            frame = api.mask_image(frame, mask)

        cv2.imshow("Frame", frame)

        key = cv2.waitKeyEx(30)
        if key == 27:
            break
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise e

cap.release()
cv2.destroyAllWindows()
