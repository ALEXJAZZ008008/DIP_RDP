# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import smoothing

if smoothing.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(smoothing.seed_value)


import parameters


def correlation_coefficient_loss(y_true, y_pred):
    def _correlation_coefficient(xm, ym):
        return (1.0 -
                tf.math.square(tf.math.maximum(tf.math.minimum(
                    tf.math.divide_no_nan(tf.math.reduce_sum((xm * ym)),
                                          tf.math.sqrt(tf.math.reduce_sum(tf.math.square(xm)) *
                                                       tf.math.reduce_sum(tf.math.square(ym)))), 1.0), -1.0)))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.cast(_correlation_coefficient(y_true - tf.math.reduce_mean(y_true),
                                            y_pred - tf.math.reduce_mean(y_pred)), dtype=tf.float32)


def correlation_coefficient_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.cast((correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0, dtype=tf.float32)
