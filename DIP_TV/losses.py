# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import main
import parameters


if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


# https://github.com/keras-team/keras/blob/master/keras/losses.py#L256-L310
def mean_squared_error_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.keras.backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


# https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282
def total_variation(images):
    # The input is a batch of images with shape:
    # [batch, height, width, depth, channels].

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height, width and depth by slicing.
    pixel_dif1 = tf.math.abs(images[:, 1:, :, :, :] - images[:, :-1, :, :, :])
    pixel_dif2 = tf.math.abs(images[:, :, 1:, :, :] - images[:, :, :-1, :, :])
    pixel_dif3 = tf.math.abs(images[:, :, :, 1:, :] - images[:, :, :, :-1, :])

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = tf.math.reduce_sum(pixel_dif1) + tf.math.reduce_sum(pixel_dif2) + tf.math.reduce_sum(pixel_dif3)

    return tot_var


def total_variation_loss(_, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return parameters.total_variation_weight * tf.reduce_mean(total_variation(y_pred))


def mean_square_error_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return ((mean_squared_error_loss(y_true, y_pred)) +
            (parameters.total_variation_weight * total_variation_loss(y_true, y_pred))) / 2.0


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def correlation_coefficient_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mx = tf.keras.backend.mean(y_true)
    my = tf.keras.backend.mean(y_pred)

    xm = y_true - mx
    ym = y_pred - my

    r_num = tf.keras.backend.sum(tf.multiply(xm, ym))
    r_den = tf.keras.backend.sqrt(tf.multiply(tf.keras.backend.sum(tf.keras.backend.square(xm)),
                                              tf.keras.backend.sum(tf.keras.backend.square(ym))))
    r = r_num / r_den

    r = tf.keras.backend.maximum(tf.keras.backend.minimum(r, 1.0), -1.0)

    return 1 - tf.keras.backend.square(r)


def accuracy_correlation_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
