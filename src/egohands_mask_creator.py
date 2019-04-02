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
parser.add_argument('metadata_file', type=str, help='the directory containing the egohands dataset')
parser.add_argument('out_dir', type=str, help='the directory containing the egohands dataset')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log_level')
parser.add_argument('--log_file', type=str, default=None, help='log_file')
args = parser.parse_args()

api.setup_logging(args.log_level, args.log_file)
logger = logging.getLogger(__name__)


def print_arr(arr, level):
    logger = logging.getLogger(__name__)
    keys = list(range(0, len(arr)))
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
            for i in range(0, level):
                logger.info("    ")
            logger.info("{} = {}".format(key, value))


mat_file = args.metadata_file
logger.info("loading {}".format(mat_file))
mat_contents = sio.loadmat(mat_file)
images = list(api.process_dir(os.path.dirname(mat_file), api.common_image_file_extensions))
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

video = mat_contents["video"][0]
outimage_idx = 0
zfill = 7
for vid_row in range(video.shape[0]):
    video_id = video[vid_row]["video_id"]
    partner_video_id = video[vid_row]["partner_video_id"]
    ego_viewer_id = video[vid_row]["ego_viewer_id"]
    partner_id = video[vid_row]["partner_id"]
    location_id = video[vid_row]["location_id"]
    activity_id = video[vid_row]["activity_id"]
    labeled_frames = video[vid_row]["labelled_frames"]
    for frm_col in range(labeled_frames.shape[1]):
        frame_info = labeled_frames[0][frm_col]

        frame_num = frame_info["frame_num"][0][0]
        print("{}".format(frame_num))

        yourright = frame_info["yourright"]
        yourleft = frame_info["yourleft"]
        myright = frame_info["myright"]
        myleft = frame_info["myleft"]
        cnt_objects = [yourright, yourleft, myright, myleft]
        contours = []
        for cnt_object in cnt_objects:
            points = []
            for cnt_row in range(cnt_object.shape[0]):
                py_pointx = cnt_object[cnt_row][0]
                py_pointy = cnt_object[cnt_row][1]
                points.append([[py_pointx, py_pointy]])
            if len(points) > 0:
                contours.append(numpy.asarray(points).astype(numpy.int32))
        img_bname = "frame_" + str(frame_num).zfill(4) + ".jpg"
        img_folder = os.path.dirname(mat_file) + "/_LABELLED_SAMPLES/" + video_id[0]
        src_fpath = os.path.join(img_folder, img_bname)
        source_image = cv2.imread(src_fpath)
        mask_image = api.create_mask_from_contour(source_image, contours)
        keep_search = True
        while keep_search:
            png_bname = str(outimage_idx).zfill(zfill) + ".png"
            png_file_name = os.path.join(args.out_dir, png_bname)
            mask_bname = str(outimage_idx).zfill(zfill) + ".mask.png"
            mask_file_name = os.path.join(args.out_dir, mask_bname)

            if not os.path.exists(png_file_name) and not os.path.exists(mask_file_name):
                print("writing png {} and mask {} in folder {}".format(png_bname, mask_bname, args.out_dir))
                cv2.imwrite(png_file_name, source_image)
                cv2.imwrite(mask_file_name, mask_image)
                keep_search = False
            else:
                outimage_idx = outimage_idx + 1

        #png_bname = os.path.splitext(os.path.basename(src_fpath))[0] + ".png"
        #png_file_name = os.path.join(os.path.dirname(src_fpath), png_bname)
        #mask_bname = os.path.splitext(os.path.basename(src_fpath))[0] + ".mask.png"
        #mask_file_name = os.path.join(os.path.dirname(src_fpath), mask_bname)