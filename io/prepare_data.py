from os import listdir
from os.path import isfile, join
from math import floor


def convert_to_grayscale(color_dir, grayscale_dir):
    import cv2

    if len(listdir(grayscale_dir)) != 0:
        return

    color_files = listdir(color_dir)
    interval = floor(len(color_files) / 10)

    print("Converting training images into grayscale", end='')
    counter = 0
    images = [file for file in color_files if
              isfile(join(color_dir, file))]
    for image_file in images:
        image = cv2.imread(join(color_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(join(grayscale_dir, image_file), image)

        counter += 1
        if counter % interval == 0:
            print('.', end='')
    print('DONE')
