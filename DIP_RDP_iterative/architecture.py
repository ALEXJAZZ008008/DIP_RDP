# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
import tensorflow as tf
import tensorflow_addons as tfa
import keras as k


import DIP_RDP_iterative

if DIP_RDP_iterative.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(DIP_RDP_iterative.seed_value)


import parameters
import layers
import losses


def get_input(input_shape):
    print("get_input")

    x = tf.keras.layers.Input(input_shape)

    return x, x


def get_encoder(x):
    print("get_encoder")

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]
    layer_groups = parameters.layer_groups[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    if not isinstance(layer_groups, list):
        layer_groups = [layer_groups]

    unet_connections = []

    x = tf.keras.layers.Conv3D(filters=1,
                               kernel_size=(1, 1, 1),
                               strides=(1, 1, 1),
                               dilation_rate=(1, 1, 1),
                               groups=1,
                               padding="same",
                               kernel_initializer=tf.keras.initializers.Constant(1.0),
                               bias_initializer=tf.keras.initializers.Constant(0.0),
                               trainable=False,
                               name="input")(x)

    x = layers.get_gaussian_noise(x, parameters.input_gaussian_sigma)

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

        unet_connections.append(layers.get_gaussian_noise(x, parameters.skip_gaussian_sigma))

        x1 = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (2, 2, 2), layer_groups[i])

        x2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                          strides=(2, 2, 2))(x)

        x = tf.keras.layers.Concatenate()([x1, x2])

    return x, unet_connections


def get_latent(x):
    print("get_latent")

    layer_layers = [parameters.layer_layers[-1]]
    layer_depth = [parameters.layer_depth[-1]]
    layer_groups = [parameters.layer_groups[-1]]

    latnet_x = x

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

        latnet_x = tf.keras.layers.Conv3D(filters=layer_depth[i],
                                          kernel_size=(1, 1, 1),
                                          strides=(1, 1, 1),
                                          dilation_rate=(1, 1, 1),
                                          groups=layer_depth[i],
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.Constant(1.0),
                                          bias_initializer=tf.keras.initializers.Constant(0.0),
                                          trainable=False,
                                          name="latent")(x)

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

    return x, latnet_x


def get_decoder(x, unet_connections):
    print("get_decoder")

    layer_layers = parameters.layer_layers[:-1]
    layer_depth = parameters.layer_depth[:-1]
    layer_groups = parameters.layer_groups[:-1]

    if not isinstance(layer_layers, list):
        layer_layers = [layer_layers]

    if not isinstance(layer_depth, list):
        layer_depth = [layer_depth]

    if not isinstance(layer_groups, list):
        layer_groups = [layer_groups]

    layer_layers.reverse()
    layer_depth.reverse()
    layer_groups.reverse()

    for i in range(len(layer_depth)):
        x = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x)

        linear_kernel = tf.constant([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                                     [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                                     [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], dtype=tf.float16)
        linear_kernel = linear_kernel / tf.math.reduce_sum(linear_kernel)
        linear_kernel = linear_kernel[:, :, :, tf.newaxis, tf.newaxis]

        for j in range(x.shape[-1] - 1):
            linear_kernel = tf.pad(linear_kernel, [[0, 0], [0, 0], [0, 0], [0, 1], [0, 1]], "SYMMETRIC")

        x = layers.get_padding(x, (3, 3, 3))
        x = tf.keras.layers.Lambda(tf.nn.conv3d, arguments={"filters": linear_kernel,
                                                            "strides": [1, 1, 1, 1, 1],
                                                            "padding": "VALID",
                                                            "dilations": [1, 1, 1, 1, 1]})(x)

        x = tf.keras.layers.Concatenate()([x, unet_connections.pop()])

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i])

    x = layers.get_convolution_layer(x, 1, (3, 3, 3), (1, 1, 1), 1)

    x = tf.keras.layers.Conv3D(filters=1,
                               kernel_size=(1, 1, 1),
                               strides=(1, 1, 1),
                               dilation_rate=(1, 1, 1),
                               groups=1,
                               padding="same",
                               kernel_initializer=tf.keras.initializers.Constant(1.0),
                               bias_initializer=tf.keras.initializers.Constant(0.0),
                               trainable=False,
                               name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x)
    x, latent_x = get_latent(x)
    output_x = get_decoder(x, unet_connections)

    return input_x, latent_x, output_x


def get_model_all(input_shape):
    print("get_model_all")

    model = get_model(input_shape)

    loss = get_loss()
    optimiser = get_optimiser()

    gc.collect()
    tf.keras.backend.clear_session()

    return model, optimiser, loss


def get_model(input_shape):
    print("get_model")

    input_x, latent_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    gc.collect()
    tf.keras.backend.clear_session()

    return model


def get_optimiser():
    print("get_optimiser")

    if parameters.weight_decay > 0.0:
        optimiser = tf.keras.mixed_precision.LossScaleOptimizer(tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)(weight_decay=parameters.weight_decay, amsgrad=True, clipnorm=1.0))  # noqa
    else:
        optimiser = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(amsgrad=True, clipnorm=1.0))

    return optimiser


def get_loss():
    print("get_loss")

    if parameters.total_variation_bool:
        loss = losses.scaled_mean_squared_error_total_variation_loss
    else:
        loss = losses.scaled_mean_squared_error_loss

    return loss
