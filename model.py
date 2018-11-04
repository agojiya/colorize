import tensorflow as tf

CONSTANT_255 = tf.constant(255, dtype=tf.float32)
CHANNEL_LABELS = ['R', 'G', 'B']


def create_model(grayscale_in):
    conv_counts = [1, 1, 1, 1, 1, 1]
    kernel_sizes = [2, 4, 4, 6, 6, 6]

    layers = [grayscale_in]
    for i in range(len(conv_counts)):
        count = conv_counts[i]
        kernel_size = kernel_sizes[i]
        for j in range(count):
            layer = tf.layers.conv2d(inputs=layers[-1], filters=16,
                                     kernel_size=kernel_size,
                                     activation=tf.nn.relu, padding="same",
                                     name="c2d-{}".format(i + j + 1))
            layers.append(layer)
        pool_layer = tf.layers.max_pooling2d(inputs=layers[-1], pool_size=2,
                                             strides=2, padding="same",
                                             name="p-{}".format(i + 1))
        layers.append(pool_layer)

    kernel_sizes_reversed = list(reversed(kernel_sizes))
    red_layers, green_layers, blue_layers = [layers[-1]], \
                                            [layers[-1]], \
                                            [layers[-1]]
    channel_layers = [red_layers, green_layers, blue_layers]
    for channel in range(3):
        for i in range(len(conv_counts)):
            prev_layer = channel_layers[channel][-1]
            kernel_size = kernel_sizes_reversed[i]
            layer_name = "c2dT-{}-{}".format(CHANNEL_LABELS[channel], i + 1)
            layer = tf.layers.conv2d_transpose(inputs=prev_layer, filters=16,
                                               kernel_size=kernel_size,
                                               strides=2,
                                               activation=tf.nn.relu,
                                               padding="same",
                                               name=layer_name)
            channel_layers[channel].append(layer)
        reduced_output = tf.reduce_sum(channel_layers[channel][-1], axis=3)
        channel_layers[channel][-1] = reduced_output

    output = [channel_layers[0][-1],
              channel_layers[1][-1],
              channel_layers[2][-1]]
    stacked_output = tf.stack(output, axis=3)
    shape = tf.shape(grayscale_in)
    return tf.image.resize_image_with_crop_or_pad(image=stacked_output,
                                                  target_height=shape[1],
                                                  target_width=shape[2])
