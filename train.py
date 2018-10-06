from os import path, listdir, mkdir
from fileio import prepare_data, utils

import tensorflow as tf
import model

import cv2
import numpy as np

BASE_DIR = path.join('X:', 'open-images-v4')
TRAIN_COLOR_DIR = path.join(BASE_DIR, 'train')
TRAIN_DIR = path.join(BASE_DIR, 'train_grayscale')

SAVE_DIR = path.join(BASE_DIR, 'colorize_saves')
SAVER_FORMAT = 'conv2d_3-{}-{}'
if not path.exists(SAVE_DIR):
    mkdir(SAVE_DIR)

prepare_data.convert_to_grayscale(color_dir=TRAIN_COLOR_DIR,
                                  grayscale_dir=TRAIN_DIR)
prepare_data.filter_by_stddev(color_dir=TRAIN_COLOR_DIR,
                              grayscale_dir=TRAIN_DIR)

grayscale_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
colorizer_out = model.create_model(grayscale_in)

color = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

loss = tf.reduce_sum(tf.squared_difference(colorizer_out, color))
optimizer = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    epoch, index = utils.get_highest_save_file(SAVE_DIR)
    if epoch != 0:
        save_file = SAVER_FORMAT.format(epoch, index)
        saver.restore(session, save_path=str(path.join(SAVE_DIR, save_file)))
        print('Loaded', epoch, 'epochs of training at', index)

    image_files = listdir(TRAIN_DIR)

    for i in range(index, len(image_files)):
        image_name = str(image_files[i])

        color_image = cv2.imread(str(path.join(TRAIN_COLOR_DIR, image_name)))
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_shape = color_image_rgb.shape
        color_image_rgb = np.reshape(color_image_rgb, (1,
                                                       color_shape[0],
                                                       color_shape[1],
                                                       color_shape[2]))

        grayscale_image = cv2.imread(str(path.join(TRAIN_DIR, image_name)),
                                     cv2.IMREAD_GRAYSCALE)
        grayscale_shape = grayscale_image.shape
        grayscale_image = np.reshape(grayscale_image, (1,
                                                       grayscale_shape[0],
                                                       grayscale_shape[1],
                                                       1))
