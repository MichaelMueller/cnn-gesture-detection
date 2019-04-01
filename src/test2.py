from time import sleep

import api
import shutil
class_names = ["1", "2", "3", "4"]
for file in api.process_dir("C:/Users/muellerm/Desktop/git/cnn-gesture-detection/var/hands/val"):
    im_basename, class_name, is_mask = api.get_img_info(file)

    im_basename_zfill = str(int(im_basename)).zfill(7)
    new_name = file.replace(im_basename+".", im_basename_zfill+".")

    print("moving {} to {}".format(file, new_name))
    shutil.move(file, new_name)

