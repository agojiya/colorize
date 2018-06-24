from os import listdir
from os.path import isfile, join


def convert_to_grayscale(color_dir, grayscale_dir):
    import cv2

    images = [file for file in listdir(color_dir) if
              isfile(join(color_dir, file))]
    for image_file in images:
        image = cv2.imread(join(color_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(join(grayscale_dir, image_file), image)
