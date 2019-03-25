import cv2
import cnn_gesture_detection_api
import argparse

# args
parser = argparse.ArgumentParser(
    description='Recorder for training data.')
parser.add_argument('classes', nargs='+', help='the set of classes to learn for the network (max number of three)')
parser.add_argument('--output_dir', default="var", type=str, help='the directory containing the recorded files')
parser.add_argument('--num_testimages', type=int, default=0, help='how many test images per class')
parser.add_argument('--cam', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--exposure', type=int, default=-7, help='the exposure level')
parser.add_argument('--timeout', type=int, default=500, help='timeout between frames in milliseconds')
parser.add_argument('--clean', action='store_true', default=False,
                    help='whether to remove train and test dir before running')
args = parser.parse_args()

cap = cv2.VideoCapture(args.cam)
if args.exposure:
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
subtractor = None

while True:
    _, frame = cap.read()
    cv2.imshow("Frame", frame)

    if subtractor:
        diff = subtractor.apply(frame)
        cv2.imshow("difference", diff)

    key = cv2.waitKeyEx(30)
    if key == 32:
        print("setting first frame")
        cv2.imshow("First frame", frame)
        subtractor = cnn_gesture_detection_api.BackgroundSubstraction(frame)
    if key == 27:
        break

cap.release()
