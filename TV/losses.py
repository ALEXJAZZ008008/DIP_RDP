# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import TV

if TV.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(TV.seed_value)


import parameters


def log_cosh_loss(y_true, y_pred):
    def _log_cosh(x):
        return (x + tf.math.softplus(-2.0 * x)) - tf.math.log(2.0)

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.math.reduce_mean(_log_cosh(y_pred - y_true))


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


def correlation_coefficient_loss(y_true, y_pred):
    def _correlation_coefficient(xm, ym):
        return 1.0 - tf.math.square(tf.math.maximum(tf.math.minimum(
            tf.math.reduce_sum((xm * ym)) / tf.math.sqrt(tf.math.reduce_sum(tf.math.square(xm)) *
                                                         tf.math.reduce_sum(tf.math.square(ym))), 1.0), -1.0))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return _correlation_coefficient(y_true - tf.math.reduce_mean(y_true), y_pred - tf.math.reduce_mean(y_pred))


def correlation_coefficient_accuracy(y_true, y_pred):
    return (correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0
