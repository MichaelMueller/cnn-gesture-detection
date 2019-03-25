import cv2
import os
import logging
import argparse
import api
import scipy.io as sio
import numpy
from collections.abc import Iterable

# args
parser = argparse.ArgumentParser(
    description='Recorder for training data.')
parser.add_argument('egohands_dataset_dir', type=str, help='the directory containing the egohands dataset')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log_level')
parser.add_argument('--log_file', type=str, default=None, help='log_file')
args = parser.parse_args()

api.setup_logging(args.log_level, args.log_file)
logger = logging.getLogger(__name__)

def print_arr(arr, level):
    logger = logging.getLogger(__name__)
    keys = list(range(0,len(arr)))
    if isinstance(arr, dict):
        keys = arr.keys()
        arr = arr.values()

    for key, value in zip(keys, arr):
        if isinstance(value, numpy.ndarray):
            print_arr(value, level + 1)
        elif type(value) is list:
            print_arr(value, level + 1)
        elif type(value) is tuple:
            print_arr(value, level + 1)
        else:
            for i in range(0,level):
                logger.info("    ")
            logger.info("{} = {}".format(key, value))

for mat_file in api.process_dir(args.egohands_dataset_dir, ["mat"]):
    logger.info("loading {}".format(mat_file))
    mat_contents = sio.loadmat(mat_file)
    polygons = mat_contents["polygons"]
    images = list(api.process_dir(os.path.dirname(mat_file), api.common_image_file_extensions))
    logger.info("polygon shape: {}".format(str(polygons.shape)))
    logger.info("found {} polygons for {} images".format(polygons.shape[0], len(images)))
    for (x, y), value in numpy.ndenumerate(polygons):
        #logger.info("x={}, y={}".format(x, y))
        #logger.info("polygons[{}][{}]={}".format(x, y, value))
        contours = value[3]
        logger.info("contours shape: {}".format(str(contours.shape)))
        logger.info("{}".format(contours))

        contours = numpy.asarray(contours.astype('float32', casting='same_kind'))
        source_image = cv2.imread(images[y])
        mask_image = api.create_mask_from_contour(source_image, [contours])
        mask_file_name = os.path.join(os.path.dirname(source_image), os.path.basename(source_image) + "_mask.png")
        cv2.imwrite(mask_file_name, mask_image)
