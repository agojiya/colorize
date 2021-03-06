from os import path, listdir, mkdir
from fileio import utils

import tensorflow as tf
import model

import cv2
import numpy as np

from image_utils import get_image

N_TARGET_IMAGES = 1000

BASE_DIR = path.join('X:', 'open-images-v4')
TRAIN_COLOR_DIR = path.join(BASE_DIR, 'train')

SAVE_DIR = path.join(BASE_DIR, 'colorize_saves')
SAVER_FORMAT = 'VGG_mirrored-Lab_space-{}-{}'
if not path.exists(SAVE_DIR):
    mkdir(SAVE_DIR)


def get_save_params(epoch, index, length):
    return (epoch + 1 if index == length - 1 else epoch,
            0 if index == length - 1 else index + 1)


grayscale_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
color_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 2])
colorizer_out = model.create_model(tf.image.random_brightness(grayscale_in,
                                                              max_delta=20))

loss = tf.losses.mean_squared_error(colorizer_out, color_in)
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
                                cv2.IMREAD_COLOR, cv2.COLOR_BGR2Lab)[0]
        color_image_ab = cv2.merge([color_image[:, :, 1],
                                    color_image[:, :, 2]])
        color_image_ab = np.expand_dims(color_image_ab, axis=0)

        c, image_loss, _ = session.run([colorizer_out, loss, optimizer],
                                       feed_dict={
                                           grayscale_in: grayscale_image,
                                           color_in: color_image_ab})
        print(image_loss)

    save_params = get_save_params(epoch, i, length)
    saver.save(session, save_path=path.join(SAVE_DIR,
                                            SAVER_FORMAT.format(*save_params)))
    print("Saved epoch {} at index {}".format(*save_params))
