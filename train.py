from fileio import prepare_data
from os import path

BASE_DIR = path.join('X:', 'open-images-v4')
TRAIN_COLOR_DIR = path.join(BASE_DIR, 'train')
TRAIN_DIR = path.join(BASE_DIR, 'train_grayscale')

prepare_data.convert_to_grayscale(color_dir=TRAIN_COLOR_DIR,
                                  grayscale_dir=TRAIN_DIR)
prepare_data.filter_by_stddev(color_dir=TRAIN_COLOR_DIR,
                              grayscale_dir=TRAIN_DIR)
