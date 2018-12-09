from os import path, listdir, mkdir
from fileio import prepare_data, utils

import tensorflow as tf
import model

import cv2
import numpy as np

from image_utils import get_image

N_TARGET_IMAGES = 1000

BASE_DIR = path.join('X:', 'open-images-v4')
TRAIN_COLOR_DIR = path.join(BASE_DIR, 'train')

SAVE_DIR = path.join(BASE_DIR, 'colorize_saves')
SAVER_FORMAT = 'conv2d_T-{}-{}'
if not path.exists(SAVE_DIR):
    mkdir(SAVE_DIR)


def get_save_params(epoch, index, length):
    return (epoch + 1 if index == length - 1 else epoch,
            0 if index == length - 1 else index + 1)


grayscale_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
color_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
colorizer_out = model.create_model(grayscale_in)

loss = tf.reduce_sum(tf.squared_difference(colorizer_out, color_in))
optimizer = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    epoch, index = utils.get_highest_save_file(SAVE_DIR)
    if epoch != 0:
        save_file = SAVER_FORMAT.format(epoch, index)
        saver.restore(session, save_path=str(path.join(SAVE_DIR, save_file)))
        print('Loaded', epoch, 'epochs of training at', index)
    else:
        epoch = 1

    width = len(str(index + N_TARGET_IMAGES))
    image_files = listdir(TRAIN_COLOR_DIR)
    length = len(image_files)
    for i in range(index, min(index + N_TARGET_IMAGES, length)):
        image_file = image_files[i]
        print(str(i + 1).zfill(width) + '/' + str(index + N_TARGET_IMAGES),
              image_file + ': ', end='', flush=True)
        grayscale_image = get_image(str(path.join(TRAIN_COLOR_DIR,
                                                  image_file)),
                                    cv2.IMREAD_GRAYSCALE)
        color_image = get_image(str(path.join(TRAIN_COLOR_DIR, image_file)),
                                cv2.IMREAD_COLOR, cv2.COLOR_BGR2RGB)

        image_loss, _ = session.run([loss, optimizer],
                                    feed_dict={grayscale_in: grayscale_image,
                                               color_in: color_image})
        loss_per_pixel = image_loss / np.size(grayscale_image)
        print(loss_per_pixel)

    save_params = get_save_params(epoch, i, length)
    saver.save(session, save_path=path.join(SAVE_DIR,
                                            SAVER_FORMAT.format(*save_params)))
    print("Saved epoch {} at index {}".format(*save_params))
