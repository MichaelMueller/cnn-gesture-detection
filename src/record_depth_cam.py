# First import the library
import argparse
import os
import shutil
import time

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# args
parser = argparse.ArgumentParser(description='A recorder for hands and mask creation using a realsense camera')
parser.add_argument('output_dir', help='the directory for writing images')
parser.add_argument('classes', nargs='+', help='the set of classes to learn for the network (max number of three)')
parser.add_argument('-m', '--masks_output_dir', help='the directory for writing the corresponding masks')
parser.add_argument('-o', '--timeout', type=int, default=500, help='timeout between frames in milliseconds')
parser.add_argument('-c', '--clipping_range', type=float, default=0.7, help='range for doing clipping')
parser.add_argument('-k', '--opening_kernel_size', type=int, default=None, help='the kernel size for doing closing')
parser.add_argument('-cl', '--cleanup', action='store_true', default=False, help='whether to remove train and test dir before running')
args = parser.parse_args()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth)
#config.enable_stream(rs.stream.color)

#config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = args.clipping_range  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

### cleanup before
if args.cleanup:
  if os.path.isdir(args.traindir):
    shutil.rmtree(args.traindir)
  if os.path.isdir(args.testdir):
    shutil.rmtree(args.testdir)

class_recorded = -1
timePoint = None

# Streaming loop
classesText = ""
for index, class_name in enumerate(args.classes):
  classesText = classesText + class_name + " (" + str(index + 1) + "), "
classesText = "got classes %spress the according key to record" % classesText
print(classesText)
try:
  while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
      continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 0
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    if args.masks_output_dir:
      gray_image_bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
      gray_image_bg_removed[gray_image_bg_removed > 0] = 255
      if args.opening_kernel_size:
            kernel = np.ones((args.opening_kernel_size, args.opening_kernel_size), np.uint8)

            difference = cv2.morphologyEx(gray_image_bg_removed, cv2.MORPH_OPEN, kernel)
      # Copy the thresholded image.
      im_floodfill = gray_image_bg_removed.copy()

      # Mask used to flood filling.
      # Notice the size needs to be 2 pixels than the image.
      h, w = gray_image_bg_removed.shape[:2]
      mask = np.zeros((h + 2, w + 2), np.uint8)

      # Floodfill from point (0, 0)
      cv2.floodFill(im_floodfill, mask, (0, 0), 255);

      # Invert floodfilled image
      im_floodfill_inv = cv2.bitwise_not(im_floodfill)

      # Combine the two images to get the foreground.
      gray_image_bg_removed = gray_image_bg_removed | im_floodfill_inv
      mask = cv2.cvtColor(gray_image_bg_removed, cv2.COLOR_GRAY2BGR)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))
    cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Align Example', images)

    # write a frame using the current class
    if class_recorded >= 0:
      if timePoint == None:
        timePoint = time.time()
      # only write image if enough time is elapsed
      timediff = (time.time() - timePoint) * 1000
      # print("timediff "+str(timediff))
      if timediff > args.timeout:
        timePoint = None
        className = args.classes[class_recorded]

        dir = os.path.join(args.output_dir, className)
        masks_dir = os.path.join(args.masks_output_dir, className) if args.masks_output_dir else None
        if not os.path.isdir(dir):
          os.makedirs(dir)
        if masks_dir and not os.path.isdir(masks_dir):
          os.makedirs(masks_dir)

        i = 0
        while True:
          i = i + 1
          file_name = os.path.join(dir, str(i).zfill(14) + ".png")
          if not os.path.exists(file_name):
            print("saving image " + file_name)
            cv2.imwrite(file_name, color_image)
            if masks_dir:
              file_name = os.path.join(masks_dir, str(i).zfill(14) + ".png")
              print("saving mask image " + file_name)
              cv2.imwrite(file_name, mask)
            break

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
      cv2.destroyAllWindows()
      break
    elif key == ord(' '):
      class_recorded = -1
      timePoint = None
      print(classesText)
    elif key == ord('1'):
      class_recorded = 0
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('2'):
      class_recorded = 1
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('3'):
      class_recorded = 2
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('4'):
      class_recorded = 3
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('5'):
      class_recorded = 4
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('6'):
      class_recorded = 5
      print("recording class " + args.classes[class_recorded] + ". press space to return")
    elif key == ord('7'):
      class_recorded = 6
      print("recording class " + args.classes[class_recorded] + ". press space to return")
finally:
  pipeline.stop()
