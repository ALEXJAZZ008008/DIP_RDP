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

    x = layers.get_convolution_layer(x, layer_depth[0], (1, 1, 1), (1, 1, 1), layer_groups[0], True, "input")

    x = layers.get_gaussian_noise(x, parameters.input_gaussian_sigma)

    for i in range(len(layer_layers)):
        x = layers.get_convolution_layer(x, layer_depth[i], (1, 1, 1), (1, 1, 1), layer_groups[i], False)

        x_skip = x

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i], False)

        x = tf.keras.layers.Add()([x, x_skip])

        unet_connections.append(layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i],
                                                             True, "latent_{0}".format(str(i))))

        x = layers.get_downsample_layer(x, layer_depth[i], (3, 3, 3), layer_groups[i])

    return x, unet_connections


def get_latent(x):
    print("get_latent")

    layer_layers = parameters.layer_layers[-1]
    layer_depth = parameters.layer_depth[-1]
    layer_groups = parameters.layer_groups[-1]

    x = layers.get_convolution_layer(x, layer_depth, (1, 1, 1), (1, 1, 1), layer_groups, False)

    x_skip = x

    for _ in range(layer_layers):
        x = layers.get_convolution_layer(x, layer_depth, (3, 3, 3), (1, 1, 1), layer_groups, False)

    x = tf.keras.layers.Add()([x, x_skip])

    latent_x = layers.get_convolution_layer(x, parameters.latent_depth, (3, 3, 3), (1, 1, 1), 1, True, "latent")

    x_skip = x

    for _ in range(layer_layers):
        x = layers.get_convolution_layer(latent_x, layer_depth, (3, 3, 3), (1, 1, 1), layer_groups, False)

    x = tf.keras.layers.Add()([x, x_skip])

    return x, latent_x


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
        x = layers.get_upsample_layer(x, layer_depth[i], (3, 3, 3), layer_groups[i])

        x = layers.get_concatenate_layer(x, unet_connections.pop(), layer_depth[i], (3, 3, 3), (1, 1, 1),
                                         layer_groups[i])

        x_skip = x

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], (3, 3, 3), (1, 1, 1), layer_groups[i], False)

        x = tf.keras.layers.Add()([x, x_skip])

    x = layers.get_convolution_layer(x, 1, (3, 3, 3), (1, 1, 1), 1, True, "output")

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, unet_connections = get_encoder(x)
    x, latent_x = get_latent(x)

    latent_layers = unet_connections.copy()
    latent_layers.append(latent_x)

    output_x = get_decoder(x, unet_connections)

    return input_x, latent_x, latent_layers, output_x


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

    input_x, latent_x, latent_layers, output_x = get_tensors(input_shape)

    outputs = [output_x]
    outputs.extend(latent_layers)

    model = k.Model(inputs=input_x, outputs=outputs)

    gc.collect()
    tf.keras.backend.clear_session()

    return model


def get_optimiser():
    print("get_optimiser")

    if DIP_RDP.bfloat_sixteen_bool:
        if parameters.weight_decay > 0.0:
            optimiser = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)(weight_decay=parameters.weight_decay, amsgrad=True, clipnorm=1.0)  # noqa
        else:
            optimiser = tf.keras.optimizers.Adam(amsgrad=True, clipnorm=1.0)
    else:
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
