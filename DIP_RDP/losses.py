# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import tensorflow_probability as tfp


import DIP_RDP

if DIP_RDP.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(DIP_RDP.seed_value)


import parameters


def scaled_mean_squared_error_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.math.divide_no_nan(tf.math.reduce_mean(tf.math.square(y_pred - y_true)),
                                 tf.math.reduce_mean(tf.math.square(y_true)))


def scaled_log_cosh_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true = (y_true - tf.reduce_min(y_true)) + parameters.epsilon
    y_pred = (y_pred - tf.reduce_min(y_pred)) + parameters.epsilon

    return tf.math.reduce_mean(tf.math.divide_no_nan(tf.math.log(tf.math.cosh(y_pred - y_true)),
                                                     tf.math.log(tf.math.cosh(y_true))))


def scaled_kl_divergence_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true = (y_true - tf.reduce_min(y_true)) / tf.math.reduce_sum(y_true)
    y_pred = (y_pred - tf.reduce_min(y_pred)) / tf.math.reduce_sum(y_pred)

    y_true = tf.cast(tf.histogram_fixed_width_bins(y_true, [tf.reduce_min(y_true), tf.reduce_max(y_true)], nbins=64),
                     dtype=tf.float32)
    y_pred = tf.cast(tf.histogram_fixed_width_bins(y_pred, [tf.reduce_min(y_pred), tf.reduce_max(y_pred)], nbins=64),
                     dtype=tf.float32)

    y_true = (y_true - tf.reduce_min(y_true)) + parameters.epsilon
    y_pred = (y_pred - tf.reduce_min(y_pred)) + parameters.epsilon

    return tf.math.divide_no_nan(tf.reduce_mean(y_true * tf.math.log(tf.math.divide_no_nan(y_true, y_pred))),
                                 tf.reduce_mean(y_true * tf.math.log(y_true)))


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


def scaled_mean_squared_error_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.reduce_sum(tf.stack([scaled_mean_squared_error_loss(y_true, y_pred),
                                   total_variation_loss(y_true, y_pred)]))


def scaled_log_cosh_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.reduce_sum([scaled_log_cosh_loss(y_true, y_pred), total_variation_loss(y_true, y_pred)])


def scaled_kl_divergence_total_variation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return tf.reduce_sum([scaled_kl_divergence_loss(y_true, y_pred), total_variation_loss(y_true, y_pred)])


def scale_regulariser(y_true, y_pred):
    def _scale_regulariser(_y_true, _y_pred):
        return parameters.scale_accuracy_scale * \
               (tf.math.divide_no_nan(tf.reduce_sum(tf.math.abs(_y_true - _y_pred)),
                                      tf.reduce_sum(tf.math.abs(_y_true))))

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    y_true = y_true - tf.reduce_min(y_true)
    y_pred = y_pred - tf.reduce_min(y_pred)

    return parameters.scale_loss_weight * _scale_regulariser(tf.math.reduce_sum(y_true), tf.math.reduce_sum(y_pred))


def scale_accuracy(y_true, y_pred):
    def _scale_accuracy(_y_true, _y_pred):
        return (1.0 - tf.math.erf(parameters.scale_accuracy_scale *
                                  tf.math.divide_no_nan(tf.reduce_sum(tf.math.abs(_y_true - _y_pred)),
                                                        tf.reduce_sum(tf.math.abs(_y_true)))))

    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    y_true = y_true - tf.reduce_min(y_true)
    y_pred = y_pred - tf.reduce_min(y_pred)

    return _scale_accuracy(tf.math.reduce_sum(y_true), tf.math.reduce_sum(y_pred))


def covariance_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)
    weight_matrix = tf.reshape(weight_matrix, shape=(-1, weight_matrix.shape[-1]))

    return parameters.covariance_weight * tf.math.reduce_mean(tfp.stats.covariance(weight_matrix, sample_axis=-1,  # noqa
                                                                                   event_axis=0))


def l1_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return tf.math.reduce_mean(tf.math.abs(weight_matrix))


def activity_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return parameters.activity_regulariser_weight * l1_regulariser(weight_matrix)


def l2_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return tf.math.reduce_mean(tf.math.square(weight_matrix))


def kernel_regulariser(weight_matrix):
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float32)

    return parameters.kernel_regulariser_weight * l2_regulariser(weight_matrix)


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
