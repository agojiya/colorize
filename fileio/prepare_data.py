from os import listdir, remove
from os.path import isfile, join
from math import floor


def convert_to_grayscale(color_dir, grayscale_dir):
    """ Converting color training images into grayscale training images. """
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


def filter_by_stddev(color_dir, grayscale_dir):
    """ Filtering grayscale images with low standard deviation. """
    grayscale_image_files = listdir(grayscale_dir)
    grayscale_size = len(grayscale_image_files)
    if grayscale_size < len(listdir(color_dir)):
        # There are fewer grayscale images than color images meaning that the
        # images have already been filtered
        return

    import cv2

    interval = floor(grayscale_size / 50)
    counter, remove_counter = 0, 0
    print('Filtering images', end='', flush=True)
    for image_file in grayscale_image_files:
        image_location = join(grayscale_dir, image_file)
        image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
        _, stddev = cv2.meanStdDev(image)
        stddev = stddev[0][0]

        if stddev <= 12:
            remove(image_location)
            remove_counter += 1

        counter += 1
        if counter % interval == 0:
            print('.', end='', flush=True)
    print('DONE', '(Removed', remove_counter, 'images)')
