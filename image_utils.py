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


def get_colored_image(grayscale_image, color_out):
    """ An implementation of the Adobe Photoshop color blending mode.
    Preserves luminosity of the grayscale image and adopts the hue and
    saturation of the color image. """
    h, _, s = cv2.split(cv2.cvtColor(color_out[0], cv2.COLOR_RGB2HLS))
    l = grayscale_image[0]
    colored_image = cv2.merge((h, l, s))
    return cv2.cvtColor(colored_image, cv2.COLOR_HLS2RGB)
