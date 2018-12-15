import cv2
import numpy as np


def get_image(image_path, read_mode, cvt_mode=None):
    image = cv2.imread(image_path, read_mode)
    if cvt_mode is not None:
        image = cv2.cvtColor(image, cvt_mode)
    image_shape = image.shape
    image = np.reshape(image, (1,
                               image_shape[0],
                               image_shape[1],
                               1 if len(image_shape) < 3 else image_shape[2]))
    return image
