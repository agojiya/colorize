import tensorflow as tf

CONSTANT_255 = tf.constant(255, dtype=tf.float32)


def create_generator(generator_input):
    """ First attempt at a generator function. """
    red = create_generator_layer(generator_input, [32, 8, 64, 16, 128, 32],
                                 'R')
    green = create_generator_layer(generator_input, [32, 8, 64, 16, 128, 32],
                                   'G')
    blue = create_generator_layer(generator_input, [32, 8, 64, 16, 128, 32],
                                  'B')
    return tf.reshape(tf.stack([red, green, blue], axis=4),
                      shape=[-1, -1, -1, 3])


def create_generator_layer(generator_input, structure, label):
    """ Creates a conv2d network starting at <generator_input> according to
    <structure> with label <label>.
    Creates structure[i] conv2d layers with one filter and kernel size
    structure[i+1] for 0<=i<=len(structure)/2. """
    layers = [generator_input]
    for i in range(int(len(structure) / 2)):
        for layer_index in range(structure[i]):
            name = label + '-' + str(i) + '-' + str(layer_index)
            layer = tf.layers.conv2d(inputs=layers[-1], filters=1,
                                     kernel_size=structure[i + 1],
                                     activation=tf.nn.relu, name=name)
            layers.append(layer)
    return tf.multiply(layers[-1], CONSTANT_255)
