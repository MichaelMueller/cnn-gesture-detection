import cv2
import numpy

import api
import argparse
import os
import hands_mask_rcnn
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import keyboard
import subprocess
from keras.utils import to_categorical
# args
parser = argparse.ArgumentParser(description='gesture control.')
parser.add_argument('weights', type=str, help='weights')
parser.add_argument('num_classes', type=int, default=None, help='num_classes')
parser.add_argument('--run_training', default=False, action="store_true", help='run_training')
parser.add_argument('--image_size', type=int, default=128, help='image_size')
parser.add_argument('--num_epochs', type=int, default=30, help='num_epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
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

# loop vars
#cmd = '"C:\\Program Files\\mRayClient\\bin\\mRayClient.exe" --hotkey'
#p = subprocess.Popen(cmd)
#pid = p.pid

training = args.run_training
curr_class_idx = 0
x_train = []
y_train = []
model = api.create_model(args.image_size, args.num_classes)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
if os.path.exists(args.weights):
    print("loading %s"%args.weights)
    model.load_weights(args.weights)
#input("Press Enter to continue...")
while True:
    try:
        # read
        _, frame = cap.read()

        # training
        if training:
            api.draw_text(frame, "current class: {}".format(curr_class_idx))
            cv2.imshow("Frame", frame)
            key = cv2.waitKeyEx(30)

            if key == 99:  # c
                curr_class_idx = curr_class_idx + 1
                if curr_class_idx >= args.num_classes:
                    curr_class_idx = 0
            elif key == 32:
                print("collecting frame for class %s" % curr_class_idx)
                img = frame.copy()
                img = api.resize_image_if_necessary(img, args.image_size)
                x_train.append(img)
                y_train.append(curr_class_idx)
            elif key == 116:
                print("running training")
                x_train = numpy.array(x_train)
                y_train = numpy.array(y_train)
                x_train = x_train / 255.0
                y_train = to_categorical(y_train)
                model.fit(x_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size)
                model.save_weights(args.weights)
                x_train = []
                y_train = []
                training = False
            elif key == 27:
                training = False

        # classification
        else:
            img = frame.copy()
            img = api.resize_image_if_necessary(img, args.image_size)
            predictions = model.predict(numpy.array([img]))
            predicted_class_idx = int(numpy.argmax(predictions[0]))
            api.draw_text(frame, "predicted class: {}".format(predicted_class_idx))

            if predicted_class_idx == 6:
                keyboard.press_and_release("e")
            elif predicted_class_idx == 5:
                keyboard.press_and_release("q")
            elif predicted_class_idx == 1:
                keyboard.press_and_release("w")
            elif predicted_class_idx == 2:
                keyboard.press_and_release("a")
            elif predicted_class_idx == 3:
                keyboard.press_and_release("s")
            elif predicted_class_idx == 4:
                keyboard.press_and_release("d")

            cv2.imshow("Frame", frame)
            key = cv2.waitKeyEx(30)
            if key == 116:
                training = True
            elif key == 27:
                break
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise e

cap.release()
cv2.destroyAllWindows()
