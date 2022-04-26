# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
import tensorflow as tf


import TV

if TV.reproducible_bool:
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(TV.seed_value)


import losses


def get_model_all():
    print("get_model_all")

    loss = get_loss()
    optimiser = get_optimiser()

    gc.collect()
    tf.keras.backend.clear_session()

    return optimiser, loss


def get_optimiser():
    print("get_optimiser")

    optimiser = tf.keras.optimizers.Adam(learning_rate=1.0, amsgrad=True)

    return optimiser


def get_loss():
    print("get_loss")

    loss = losses.total_variation_loss

    return loss
