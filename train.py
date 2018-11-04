from os import path, listdir, mkdir
from fileio import prepare_data, utils

import tensorflow as tf
import model

import cv2
import numpy as np

N_TARGET_IMAGES = 10

BASE_DIR = path.join('X:', 'open-images-v4')
TRAIN_COLOR_DIR = path.join(BASE_DIR, 'train')
TRAIN_DIR = path.join(BASE_DIR, 'train_grayscale')

SAVE_DIR = path.join(BASE_DIR, 'colorize_saves')
SAVER_FORMAT = 'conv2d_T-{}-{}'
if not path.exists(SAVE_DIR):
    mkdir(SAVE_DIR)

prepare_data.convert_to_grayscale(color_dir=TRAIN_COLOR_DIR,
                                  grayscale_dir=TRAIN_DIR)
prepare_data.filter_by_stddev(color_dir=TRAIN_COLOR_DIR,
                              grayscale_dir=TRAIN_DIR)


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

    width = len(str(N_TARGET_IMAGES))
    image_files = listdir(TRAIN_DIR)
    length = len(image_files)
    for i in range(index, min(index + N_TARGET_IMAGES, length)):
        image_file = image_files[i]
        print(str(i + 1).zfill(width) + '/' + str(N_TARGET_IMAGES),
              image_file + ': ', end='', flush=True)
        grayscale_image = get_image(str(path.join(TRAIN_DIR, image_file)),
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
