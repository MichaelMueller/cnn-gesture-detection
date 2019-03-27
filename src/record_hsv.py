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
parser.add_argument('--timeout', type=int, default=500, help='timeout between Frames in milliseconds')
parser.add_argument('--clean', action='store_true', default=False,
                    help='whether to remove train and test dir before running')
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
            frame = api.mask_image(frame, mask)
        api.draw_text(frame, "current class: {}".format(args.classes[curr_class_idx]))

        # show images
        cv2.imshow("Frame", frame)
        # cv2.imshow("mask", mask)
        # cv2.imshow("mask_biggest_contour", mask_biggest_contour)

        # process keys
        key = cv2.waitKeyEx(30)
        if key == 99:  # c
            curr_class_idx = curr_class_idx + 1
            if curr_class_idx >= len(args.classes):
                curr_class_idx = 0
        elif key == 32 and mask is not None:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            curr_class = args.classes[curr_class_idx]
            i = i + 1
            img_path = os.path.join(args.output_dir, str(i) + "." + curr_class + ".png")
            mask_path = os.path.join(args.output_dir, str(i) + "." + curr_class + ".mask.png")
            while os.path.exists(img_path) or os.path.exists(mask_path):
                i = i + 1
                img_path = os.path.join(args.output_dir, str(i) + curr_class + ".png")
                mask_path = os.path.join(args.output_dir, str(i) + curr_class + ".mask.png")
            cv2.imwrite(img_path, frame_orig)
            cv2.imwrite(mask_path, mask)
            hsv_segmentation.save()
        if key == 27:
            break
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise e

cap.release()
cv2.destroyAllWindows()
