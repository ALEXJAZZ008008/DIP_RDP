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


def mean_squared_error_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.reduce_sum(tf.stack([mean_squared_error_loss(y_true, y_pred),
                                   total_variation_loss(y_true, y_pred)]))


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
