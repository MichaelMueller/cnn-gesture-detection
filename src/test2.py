from time import sleep

import api
import shutil
class_names = ["hand", "one", "two", "fist"]
for file in api.process_dir("C:/Users/muellerm/Desktop/git/cnn-gesture-detection/var/hands/train"):
    class_name, is_mask = api.get_img_info(file)
    if class_name is not None and class_name in class_names:
        idx = class_names.index(class_name) + 1
        new_name = file.replace("."+class_name+".", "."+str(idx)+".")

        print("moving {} to {}".format(file, new_name))
        shutil.move(file, new_name)

