import tensorflow as tf

CONSTANT_255 = tf.constant(255, dtype=tf.float32)
CHANNEL_LABELS = ['H', 'S']


def create_model(grayscale_in):
    """ Creating a convolution section similar to
    https://arxiv.org/abs/1409.1556 (VGG-16) and transposed convolution
    sections for each channel similar to https://arxiv.org/abs/1505.04366 """
    conv_counts = [2, 2, 2, 3, 3]
    kernel_sizes = [3, 3, 3, 3, 3]
    filter_counts = [32, 64, 128, 128, 128]

    layers = [grayscale_in]
    for i in range(len(conv_counts)):
        count = conv_counts[i]
        kernel_size = kernel_sizes[i]
        filter_count = filter_counts[i]
        for j in range(count):
            layer_name = "c2d-{}-{}".format(i + 1, j + 1)
            layer = tf.layers.conv2d(inputs=layers[-1], filters=filter_count,
                                     kernel_size=kernel_size,
                                     activation=tf.nn.relu, padding="same",
                                     name=layer_name)
            layers.append(layer)
        pool_layer = tf.layers.max_pooling2d(inputs=layers[-1], pool_size=2,
                                             strides=2, padding="same",
                                             name="p-{}".format(i + 1))
        layers.append(pool_layer)

    kernel_sizes_reversed = list(reversed(kernel_sizes))
    filter_counts_reversed = list(reversed(filter_counts))
    hue_layers, sat_layers = [layers[-1]], \
                             [layers[-1]]
    channel_layers = [hue_layers, sat_layers]
    for channel in range(2):
        for i in range(len(conv_counts)):
            prev_layer = channel_layers[channel][-1]
            kernel_size = kernel_sizes_reversed[i]
            filter_count = filter_counts_reversed[i]
            layer_name = "c2dT-{}-{}".format(CHANNEL_LABELS[channel], i + 1)
            layer = tf.layers.conv2d_transpose(inputs=prev_layer,
                                               filters=filter_count,
                                               kernel_size=kernel_size,
                                               strides=2,
                                               activation=tf.nn.relu,
                                               padding="same",
                                               name=layer_name)
            channel_layers[channel].append(layer)
        reduced_output = tf.reduce_sum(channel_layers[channel][-1], axis=3)
        channel_layers[channel][-1] = reduced_output

    stacked_output = tf.stack([channel_layers[0][-1],
                               channel_layers[1][-1]],
                              axis=3)

    shape = tf.shape(grayscale_in)
    return tf.image.resize_image_with_crop_or_pad(image=stacked_output,
                                                  target_height=shape[1],
                                                  target_width=shape[2])
