"""
Mask R-CNN
Train on the toy Hands dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Hands.py train --dataset=/path/to/Hands/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 Hands.py train --dataset=/path/to/Hands/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Hands.py train --dataset=/path/to/Hands/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Hands.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Hands.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
import api

ROOT_DIR = os.path.abspath("../assets/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class HandsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Hands"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + Hand

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class HandsDataset(utils.Dataset):

    def load_Hands(self, dataset_dir, subset):
        """Load a subset of the Hands dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        i = 0
        class_names = []
        for image_fpath in api.process_dir(dataset_dir, api.common_image_file_extensions):
            _, class_name, is_mask = api.get_img_info(image_fpath)
            if class_name is None:
                continue
            if class_name not in class_names:
                class_names.append(class_name)
                self.add_class("HandsDepthDataset", len(class_names), class_name)
            cls_id = class_names.index(class_name) + 1

            if is_mask is False:
                i = i + 1
                self.add_image(
                    "HandsDepthDataset",
                    image_id=i,  # use file name as a unique image id
                    path=image_fpath,
                    cls_id=cls_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        image_fpath_part1, image_fpath_ext = os.path.splitext(image_info["path"])
        mask_path = image_fpath_part1 + ".mask" + image_fpath_ext
        mask = skimage.color.rgb2gray(skimage.io.imread(mask_path))
        mask.shape = (mask.shape[0], mask.shape[1], 1)
        class_ids = np.array([image_info["cls_id"]])

        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "HandsDepthDataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, num_epopchs=30):
    """Train the model."""
    # Training dataset.
    dataset_train = HandsDataset()
    dataset_train.load_Hands(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HandsDataset()
    dataset_val.load_Hands(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=num_epopchs,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        images = []
        if os.path.isdir(image_path):
            for path in api.process_dir(image_path, api.common_image_file_extensions):
                images.append(path)
        else:
            images.append(image_path)

        for image_path in images:
            _, class_name, is_mask = api.get_img_info(image_path)
            if is_mask:
                print("Mask detected {}. Skipping".format(image_path))
                continue
            # Run model detection and generate the color splash effect
            print("Running on {}".format(image_path))
            # Read image
            image = skimage.io.imread(image_path)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            class_ids = r["class_ids"]
            print("class_ids for %s: %s" % (image_path, class_ids))
            # Color splash
            splash = color_splash(image, r['masks'])
            # Save output
            file_name = "splash_{:%Y%m%dT%H%M%S}_class_{}.png".format(datetime.datetime.now(),
                                                                      class_ids[0] if len(class_ids) > 0 else 0)
            skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Handss.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Hands/dataset/",
                        help='Directory of the Hands dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--num_epochs',
                        metavar="30",
                        default=30,
                        type=int,
                        help="number of epochs in training")
    parser.add_argument('--steps',
                        metavar="1000",
                        default=1000,
                        type=int,
                        help="number of steps per epoch in training")
    parser.add_argument('--confidence',
                        metavar="0.9",
                        default=0.9,
                        type=float,
                        help="confidence for inference")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HandsConfig()
    else:
        class InferenceConfig(HandsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.DETECTION_MIN_CONFIDENCE = args.confidence
    config.STEPS_PER_EPOCH = args.steps
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.num_epochs)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
