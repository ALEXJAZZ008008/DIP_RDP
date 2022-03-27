# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf


import DIP_RDP_iterative

if DIP_RDP_iterative.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(DIP_RDP_iterative.seed_value)


import parameters


def scaled_mean_squared_error_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.cast(tf.math.reduce_mean(tf.math.square(y_pred - y_true)) /
                   tf.math.reduce_mean(tf.math.square(y_true)), dtype=tf.float32)


def total_variation_loss(_, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    oon = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(1.0) + tf.math.square(1.0))).numpy()
    ooo = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(1.0) + tf.math.square(1.0) + tf.math.square(1.0))).numpy()
    tnn = tf.math.divide_no_nan(1.0, 2.0).numpy()
    ton = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(2.0) + tf.math.square(1.0))).numpy()
    too = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(2.0) + tf.math.square(1.0) + tf.math.square(1.0))).numpy()
    ttn = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(2.0) + tf.math.square(2.0))).numpy()
    tto = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(2.0) + tf.math.square(2.0) + tf.math.square(1.0))).numpy()
    ttt = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.math.square(2.0) + tf.math.square(2.0) + tf.math.square(2.0))).numpy()

    total_variation_kernel = tf.constant([[[ttt, tto, ttn, tto, ttt], [tto, too, ton, too, tto], [ttn, ton, tnn, ton, ttn], [tto, too, ton, too, tto], [ttt, tto, ttn, tto, ttt]],
                                          [[tto, too, ton, too, tto], [ttn, ooo, oon, ooo, ttn], [ton, oon, 1.0, oon, ton], [ttn, ooo, oon, ooo, ttn], [tto, too, ton, too, tto]],
                                          [[ttn, ton, tnn, ton, ttn], [ton, oon, 1.0, oon, ton], [tnn, 1.0, 0.0, 1.0, tnn], [ton, oon, 1.0, oon, ton], [ttn, ton, tnn, ton, ttn]],
                                          [[tto, too, ton, too, tto], [ttn, ooo, oon, ooo, ttn], [ton, oon, 1.0, oon, ton], [ttn, ooo, oon, ooo, ttn], [tto, too, ton, too, tto]],
                                          [[ttt, tto, ttn, tto, ttt], [tto, too, ton, too, tto], [ttn, ton, tnn, ton, ttn], [tto, too, ton, too, tto], [ttt, tto, ttn, tto, ttt]]], dtype=tf.float32)
    total_variation_kernel = tf.math.divide_no_nan(total_variation_kernel, tf.math.reduce_sum(total_variation_kernel))

    tvs = tf.reduce_sum(total_variation_kernel).numpy()

    total_variation_kernel = total_variation_kernel * -1.0
    total_variation_kernel = (total_variation_kernel +
                              tf.constant([[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, tvs, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32))
    total_variation_kernel = total_variation_kernel[:, :, :, tf.newaxis, tf.newaxis]

    y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    y_pred = tf.pad(y_pred, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    y_pred = tf.nn.conv3d(input=y_pred,
                          filters=total_variation_kernel,
                          strides=[1, 1, 1, 1, 1],
                          padding="VALID",
                          dilations=[1, 1, 1, 1, 1])

    return tf.cast(parameters.total_variation_weight * tf.reduce_mean(tf.math.square(y_pred)), dtype=tf.float32)


def scaled_mean_squared_error_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.cast(tf.reduce_sum(tf.stack([scaled_mean_squared_error_loss(y_true, y_pred),
                                           total_variation_loss(y_true, y_pred)])), dtype=tf.float32)


def l1_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return tf.cast(tf.math.reduce_mean(tf.math.abs(weight_matrix)), dtype=tf.float32)


def l2_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return tf.cast(tf.math.reduce_mean(tf.math.square(weight_matrix)), dtype=tf.float32)


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


def scale_regulariser(y_true, y_pred):
    def _scale_regulariser(_y_true, _y_pred):
        return parameters.scale_accuracy_scale * \
               (tf.math.divide_no_nan(tf.reduce_mean(tf.math.abs(_y_true - _y_pred)),
                                      tf.reduce_mean(tf.math.abs(_y_true))))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true = y_true - tf.math.reduce_min(y_true)
    y_pred = y_pred - tf.math.reduce_min(y_pred)

    return tf.cast(parameters.scale_loss_weight *
                   _scale_regulariser(tf.math.reduce_sum(y_true), tf.math.reduce_sum(y_pred)), dtype=tf.float32)


def scale_accuracy(y_true, y_pred):
    def _scale_accuracy(_y_true, _y_pred):
        return (1.0 - tf.math.erf(parameters.scale_accuracy_scale *
                                  tf.math.divide_no_nan(tf.reduce_mean(tf.math.abs(_y_true - _y_pred)),
                                                        tf.reduce_mean(tf.math.abs(_y_true)))))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true = y_true - tf.math.reduce_min(y_true)
    y_pred = y_pred - tf.math.reduce_min(y_pred)

    return tf.cast(_scale_accuracy(tf.math.reduce_sum(y_true), tf.math.reduce_sum(y_pred)), dtype=tf.float32)
