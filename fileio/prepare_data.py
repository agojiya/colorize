from os import listdir
from os.path import isfile, join
from math import floor


def convert_to_grayscale(color_dir, grayscale_dir):
    if len(listdir(grayscale_dir)) != 0:
        return

    import cv2

    color_files = listdir(color_dir)
    interval = floor(len(color_files) / 50)

    print('Converting images', end='', flush=True)
    counter = 0
    images = [file for file in color_files if
              isfile(join(color_dir, file))]
    for image_file in images:
        image = cv2.imread(join(color_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(join(grayscale_dir, image_file), image)

        counter += 1
        if counter % interval == 0:
            print('.', end='', flush=True)
    print('DONE')


def filter_by_variance(color_dir, grayscale_dir):
    grayscale_image_files = listdir(grayscale_dir)
    if len(grayscale_image_files) < len(listdir(color_dir)):
        # There are fewer grayscale images than color images meaning that the
        # images have already been filtered
        return

    import cv2

    for image_file in grayscale_image_files:
        image = cv2.imread(join(grayscale_dir, image_file),
                           cv2.IMREAD_GRAYSCALE)
        _, stddev = cv2.meanStdDev(image)
        stddev = stddev[0][0]

        if stddev <= 10:
            # TODO: Remove the image
            pass
