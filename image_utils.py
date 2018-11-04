import cv2


def get_colored_image(grayscale_image, color_out):
    """ An implementation of the Adobe Photoshop color blending mode.
    Preserves luminosity of the grayscale image and adopts the hue and
    saturation of the color image. """
    h, _, s = cv2.split(cv2.cvtColor(color_out[0], cv2.COLOR_RGB2HLS))
    l = grayscale_image[0]
    colored_image = cv2.merge((h, l, s))
    return cv2.cvtColor(colored_image, cv2.COLOR_HLS2RGB)
