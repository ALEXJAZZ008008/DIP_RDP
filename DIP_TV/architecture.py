# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import tensorflow as tf
import keras as k


import main
import parameters
import layers
import losses


if main.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(main.seed_value)


def get_input(input_shape):
    print("get_input")

    input_x = tf.keras.layers.Input(input_shape)

    return input_x, input_x


def get_encoder(x):
    print("get_encoder")

    layer_layers = [2, 2, 2]
    layer_depth = [16, 32, 64]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [1, 1, 1]

    res_connections = []

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

        res_connections.append(x)

        x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], layer_stride[i], layer_groups[i])

    return x, res_connections


def get_latent(x):
    print("get_latent")

    layer_layers = [2]
    layer_depth = [128]
    layer_kernel_size = [(3, 3, 3)]
    layer_groups = [1]

    for i in range(len(layer_layers)):
        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

    return x


def get_decoder(x, res_connections):
    print("get_decoder")

    layer_layers = [2, 2, 2]
    layer_depth = [64, 32, 16]
    layer_kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    layer_stride = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layer_groups = [1, 1, 1]

    for i in range(len(layer_depth)):
        x = tf.keras.layers.Conv3DTranspose(filters=layer_depth[i],
                                            kernel_size=layer_kernel_size[i],
                                            strides=(1, 1, 1),
                                            dilation_rate=(1, 1, 1),
                                            groups=layer_groups[i],
                                            padding="same",
                                            kernel_initializer="he_normal",
                                            bias_initializer=tf.keras.initializers.Constant(0.0))(x)

        x = tf.keras.layers.UpSampling3D(size=layer_stride[i])(x)

        res_connections_x = res_connections.pop()
        x = tf.keras.layers.Add()([x, res_connections_x])

        for _ in range(layer_layers[i]):
            x = layers.get_convolution_layer(x, layer_depth[i], layer_kernel_size[i], (1, 1, 1), layer_groups[i])

    # output
    x = tf.keras.layers.Conv3D(filters=1,
                               kernel_size=(3, 3, 3),
                               strides=(1, 1, 1),
                               dilation_rate=(1, 1, 1),
                               groups=1,
                               padding="same",
                               kernel_initializer="he_normal",
                               bias_initializer=tf.keras.initializers.Constant(0.0),
                               name="output")(x)

    return x


def get_tensors(input_shape):
    print("get_tensors")

    x, input_x = get_input(input_shape)

    x, res_connections = get_encoder(x)

    x = get_latent(x)

    output_x = get_decoder(x, res_connections)

    return input_x, output_x


def get_model(input_shape):
    print("get_model")

    input_x, output_x = get_tensors(input_shape)

    model = k.Model(inputs=input_x, outputs=[output_x])

    if parameters.total_variation_bool and parameters.total_variation_weight > 0.0:
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss={"output": losses.mean_squared_error_total_variation_loss}, loss_weights=[1.0],
                      metrics=[losses.accuracy_correlation_coefficient])
    else:
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss={"output": losses.mean_squared_error_loss}, loss_weights=[1.0],
                      metrics=[losses.accuracy_correlation_coefficient])

    return model
