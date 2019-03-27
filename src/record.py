import cv2
import api
import argparse
import os

# args
parser = argparse.ArgumentParser(
    description='Recorder for training data.')
parser.add_argument('classes', nargs='+', help='the set of classes to learn for the network (max number of three)')
parser.add_argument('output_dir', type=str, help='the directory containing the recorded files')
parser.add_argument('--num_testimages', type=int, default=0, help='how many test images per class')
parser.add_argument('--cam', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--width', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--height', type=int, default=0, help='the number of the camera to use')
parser.add_argument('--exposure', type=int, default=-7, help='the exposure level')
parser.add_argument('--timeout', type=int, default=500, help='timeout between frames in milliseconds')
parser.add_argument('--clean', action='store_true', default=False,
                    help='whether to remove train and test dir before running')
args = parser.parse_args()

cap = cv2.VideoCapture(args.cam)
if args.exposure:
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
if args.width and args.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

subtractor = None

i = 0
while True:
    _, frame = cap.read()
    cv2.imshow("Frame", frame)

    mask = None
    if subtractor:
        mask = subtractor.apply(frame)
        biggest_contour = api.find_biggest_contour(mask)
        if biggest_contour is not None:
            mask = api.create_mask_from_contour(mask, [biggest_contour])
            # cv2.imshow("mask_with_contour", mask_with_contour)
            cv2.imshow("mask", mask)
        else:
            mask = None

    key = cv2.waitKeyEx(30)
    if key == 102:
        print("setting first frame")
        #cv2.imshow("First frame", frame)
        subtractor = api.BackgroundSubstraction(frame)
    elif key == 32 and mask is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        i = i + 1
        img_path = os.path.join(args.output_dir, str(i)+".png")
        mask_path = os.path.join(args.output_dir, str(i) + ".mask.png")
        while os.path.exists(img_path) or os.path.exists(mask_path):
            i = i + 1
            img_path = os.path.join(args.output_dir, str(i)+".png")
            mask_path = os.path.join(args.output_dir, str(i) + ".mask.png")
        cv2.imwrite(img_path, frame)
        cv2.imwrite(mask_path, mask)
    if key == 27:
        break

cap.release()
