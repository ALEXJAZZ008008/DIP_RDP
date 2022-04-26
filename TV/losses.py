# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import TV

if TV.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(TV.seed_value)


import parameters


def total_variation_loss(_, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_pred = y_pred - tf.reduce_min(y_pred)

    oon = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(1.0) + tf.math.square(1.0))).numpy()
    ooo = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(1.0) + tf.math.square(1.0) + tf.math.square(1.0))).numpy()

    total_variation_kernel = tf.constant([[[ooo, oon, ooo], [oon, 1.0, oon], [ooo, oon, ooo]],
                                          [[oon, 1.0, oon], [1.0, 0.0, 1.0], [oon, 1.0, oon]],
                                          [[ooo, oon, ooo], [oon, 1.0, oon], [ooo, oon, ooo]]], dtype=tf.float32)
    total_variation_kernel = tf.math.divide_no_nan(total_variation_kernel, tf.math.reduce_sum(total_variation_kernel))

    tvs = tf.reduce_sum(total_variation_kernel).numpy()

    total_variation_kernel = total_variation_kernel * -1.0
    total_variation_kernel = (total_variation_kernel +
                              tf.constant([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0], [0.0, tvs, 0.0], [0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=tf.float32))
    total_variation_kernel = total_variation_kernel[:, :, :, tf.newaxis, tf.newaxis]

    y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    y_pred = tf.nn.conv3d(input=y_pred,
                          filters=total_variation_kernel,
                          strides=[1, 1, 1, 1, 1],
                          padding="VALID",
                          dilations=[1, 1, 1, 1, 1])

    return parameters.total_variation_weight * tf.reduce_mean(tf.math.square(y_pred))


def correlation_coefficient_loss(y_true, y_pred):
    def _correlation_coefficient(xm, ym):
        return (1.0 -
                tf.math.square(tf.math.maximum(tf.math.minimum(
                    tf.math.divide_no_nan(tf.math.reduce_sum((xm * ym)),
                                          tf.math.sqrt(tf.math.reduce_sum(tf.math.square(xm)) *
                                                       tf.math.reduce_sum(tf.math.square(ym)))), 1.0), -1.0)))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # y_true = y_true - tf.reduce_min(y_true)
    # y_pred = y_pred - tf.reduce_min(y_pred)

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    # mask = tf.where(y_true)

    # y_true = y_true[mask]
    # y_pred = y_pred[mask]

    return _correlation_coefficient(y_true - tf.math.reduce_mean(y_true), y_pred - tf.math.reduce_mean(y_pred))


def correlation_coefficient_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    return tf.cast((correlation_coefficient_loss(y_true, y_pred) * -1.0) + 1.0, dtype=tf.float64)
