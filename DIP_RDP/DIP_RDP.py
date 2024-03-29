# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import gc
import os
import re
import shutil
import math
import random
import numpy as np
import scipy.constants
import scipy.stats
import scipy.ndimage
import tensorflow as tf
# from numba import cuda
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import gzip


reproducible_bool = True

if reproducible_bool:
    # Seed value (can actually be different for each attribution step)
    seed_value = 0

    # 1. Set "PYTHONHASHSEED" environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    # ONLY WORKS WITH TF V1
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)
else:
    random.seed()

# device = cuda.get_current_device()

float_sixteen_bool = True  # set the network to use float16 data
bfloat_sixteen_bool = False
cpu_bool = False  # if using CPU, set to true: disables mixed precision computation

# mixed precision float16 computation allows the network to use both float16 and float32 where necessary,
# this improves performance on the GPU.
if float_sixteen_bool and not cpu_bool:
    if bfloat_sixteen_bool:
        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
    else:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
else:
    float_sixteen_bool = False
    bfloat_sixteen_bool = False

    policy = tf.keras.mixed_precision.Policy(tf.dtypes.float32.name)

tf.keras.mixed_precision.set_global_policy(policy)


# gpus = tf.config.list_physical_devices("GPU")

# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

#     os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


import parameters
import preprocessing
import architecture
import losses


data_path = None
output_path = None

model_path = None


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    print("atoi")

    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    print("human_sorting")

    return [atoi(c) for c in re.split(r'(\d+)', string)]


def get_data_windows(data):
    print("get_data_windows")

    axial_size = data.shape[-1]

    windowed_full_input_axial_size = None

    if axial_size > parameters.data_window_size:
        if parameters.data_window_bool:
            number_of_windows = int(np.ceil(axial_size / parameters.data_window_size)) + 1
            number_of_overlaps = number_of_windows - 1

            overlap_size = \
                int(np.ceil((((parameters.data_window_size * number_of_windows) - axial_size) / number_of_overlaps)))
            overlap_index = int(np.ceil(parameters.data_window_size - overlap_size))

            windowed_full_input_axial_size = \
                (number_of_windows * parameters.data_window_size) - (number_of_overlaps * overlap_size)

            data = np.squeeze(preprocessing.data_upsample_pad([np.expand_dims(np.expand_dims(data, 0), -1)], "numpy",
                                                              (data.shape[0], data.shape[1],
                                                               windowed_full_input_axial_size)))

            current_data = []

            for i in range(number_of_windows):
                current_overlap_index = overlap_index * i

                current_data.append(
                    data[:, :, current_overlap_index:current_overlap_index + parameters.data_window_size])

            data = current_data
        else:
            data_centre = int(axial_size / 2.0)

            data_upper_window_size = int(parameters.data_window_size / 2.0)
            data_lower_window_size = data_upper_window_size

            if data_upper_window_size + data_lower_window_size < parameters.data_window_size:
                data_upper_window_size = data_upper_window_size + 1

            data = [data[:, :, data_centre - data_upper_window_size:data_centre + data_lower_window_size]]
    else:
        data = [data]

    data = np.asarray(data)

    if windowed_full_input_axial_size is None:
        windowed_full_input_axial_size = data.shape[-1]

    return data, windowed_full_input_axial_size


def normalise_voxel_sizes(data, voxel_sizes):
    print("normalise_voxel_sizes")

    min_voxel_sizes = np.min(voxel_sizes)
    data_shape = data.shape

    data = \
        np.squeeze(preprocessing.data_upsample([data], "numpy",
                                               (int(np.round((data_shape[0] / min_voxel_sizes) * voxel_sizes[0])),
                                                int(np.round((data_shape[1] / min_voxel_sizes) * voxel_sizes[1])),
                                                int(np.round((data_shape[2] / min_voxel_sizes) * voxel_sizes[2]))))[0])

    return data


def get_train_data():
    print("get_train_data")

    y_path = "{0}/y/".format(data_path)

    y_files = os.listdir(y_path)
    y_files.sort(key=human_sorting)
    y_files = ["{0}{1}".format(y_path, s) for s in y_files]

    example_data = nib.load(y_files[0])
    # voxel_sizes = example_data.header.get_zooms()

    y = []

    y_train_output_path = "{0}/y_train".format(output_path)

    if not os.path.exists(y_train_output_path):
        os.makedirs(y_train_output_path, mode=0o770)

    high_resolution_input_shape = None
    full_current_shape = None
    windowed_full_input_axial_size = None
    current_shape = None
    input_noise = None

    for i in range(len(y_files)):
        current_volume = nib.load(y_files[i])
        current_array = current_volume.get_fdata()

        if high_resolution_input_shape is None:
            high_resolution_input_shape = current_array.shape

        if parameters.data_crop_bool:
            current_array = \
                np.squeeze(preprocessing.data_downsampling_crop([current_array], "numpy",
                                                                [current_array.shape[0] -
                                                                 parameters.data_crop_amount[0],
                                                                 current_array.shape[1] -
                                                                 parameters.data_crop_amount[1],
                                                                 current_array.shape[2] -
                                                                 parameters.data_crop_amount[2]])[0])

        if full_current_shape is None:
            full_current_shape = current_array.shape

        # current_array = normalise_voxel_sizes(current_array, voxel_sizes)
        current_array, windowed_full_input_axial_size = get_data_windows(current_array)

        if current_shape is None:
            current_shape = current_array[0].shape

        current_y_train_path = "{0}/{1}.npy.gz".format(y_train_output_path, str(i))

        with gzip.GzipFile(current_y_train_path, "w") as file:
            np.save(file, current_array)  # noqa

        y.append(current_y_train_path)

    data_mask_path = "{0}/data_mask/".format(data_path)

    data_mask_train_output_path = "{0}/data_mask_train".format(output_path)

    if not os.path.exists(data_mask_train_output_path):
        os.makedirs(data_mask_train_output_path, mode=0o770)

    if os.path.exists(data_mask_path):
        data_mask = []

        data_mask_files = os.listdir(data_mask_path)
        data_mask_files.sort(key=human_sorting)
        data_mask_files = ["{0}{1}".format(data_mask_path, s) for s in data_mask_files]

        for i in range(len(data_mask_files)):
            current_volume = nib.load(data_mask_files[i])
            current_array = current_volume.get_fdata()

            if parameters.data_crop_bool:
                current_array = \
                    np.squeeze(preprocessing.data_downsampling_crop([current_array], "numpy",
                                                                    [current_array.shape[0] -
                                                                     parameters.data_crop_amount[0],
                                                                     current_array.shape[1] -
                                                                     parameters.data_crop_amount[1],
                                                                     current_array.shape[2] -
                                                                     parameters.data_crop_amount[2]])[0])

            # current_array = normalise_voxel_sizes(current_array, voxel_sizes)
            current_array, _ = get_data_windows(current_array)

            current_data_mask_train_path = "{0}/{1}.npy.gz".format(data_mask_train_output_path, str(i))

            with gzip.GzipFile(current_data_mask_train_path, "w") as file:
                np.save(file, current_array)  # noqa

            data_mask.append(current_data_mask_train_path)

        data_mask = np.asarray(data_mask)
    else:
        # data_mask = []

        # for i in range(len(y)):
        #     with gzip.GzipFile(y[i], "r") as file:
        #         current_array = np.load(file)  # noqa

        #     current_array = np.ones(current_array.shape)
        #     current_array = preprocessing.mask_fov(current_array)

        #     current_data_mask_train_path = "{0}/{1}.npy.gz".format(data_mask_train_output_path, str(i))

        #     with gzip.GzipFile(current_data_mask_train_path, "w") as file:
        #         np.save(file, current_array)  # noqa

        #     data_mask.append(current_data_mask_train_path)

        # data_mask = np.asarray(data_mask)

        data_mask = None

    loss_mask_path = "{0}/loss_mask/".format(data_path)

    loss_mask_train_output_path = "{0}/loss_mask_train".format(output_path)

    if not os.path.exists(loss_mask_train_output_path):
        os.makedirs(loss_mask_train_output_path, mode=0o770)

    if os.path.exists(loss_mask_path):
        loss_mask = []

        loss_mask_files = os.listdir(loss_mask_path)
        loss_mask_files.sort(key=human_sorting)
        loss_mask_files = ["{0}{1}".format(loss_mask_path, s) for s in loss_mask_files]

        for i in range(len(loss_mask_files)):
            current_volume = nib.load(loss_mask_files[i])
            current_array = current_volume.get_fdata()

            if parameters.data_crop_bool:
                current_array = \
                    np.squeeze(preprocessing.data_downsampling_crop([current_array], "numpy",
                                                                    [current_array.shape[0] -
                                                                     parameters.data_crop_amount[0],
                                                                     current_array.shape[1] -
                                                                     parameters.data_crop_amount[1],
                                                                     current_array.shape[2] -
                                                                     parameters.data_crop_amount[2]])[0])

            # current_array = normalise_voxel_sizes(current_array, voxel_sizes)
            current_array, _ = get_data_windows(current_array)

            current_loss_mask_train_path = "{0}/{1}.npy.gz".format(loss_mask_train_output_path, str(i))

            with gzip.GzipFile(current_loss_mask_train_path, "w") as file:
                np.save(file, current_array)  # noqa

            loss_mask.append(current_loss_mask_train_path)
    else:
        loss_mask = []

        for i in range(len(y)):
            with gzip.GzipFile(y[i], "r") as file:
                current_array = np.load(file)  # noqa

            current_array = np.ones(current_array.shape)
            # current_array = preprocessing.mask_fov(current_array)

            current_loss_mask_train_path = "{0}/{1}.npy.gz".format(loss_mask_train_output_path, str(i))

            with gzip.GzipFile(current_loss_mask_train_path, "w") as file:
                np.save(file, current_array)  # noqa

            loss_mask.append(current_loss_mask_train_path)

    x = []

    x_train_output_path = "{0}/x_train".format(output_path)

    if not os.path.exists(x_train_output_path):
        os.makedirs(x_train_output_path, mode=0o770)

    for i in range(len(y)):
        with gzip.GzipFile(y[i], "r") as file:
            current_array = np.load(file)  # noqa

        if parameters.data_input_bool:
            if parameters.data_gaussian_smooth_sigma_xy > 0.0 or parameters.data_gaussian_smooth_sigma_z > 0.0:
                if loss_mask is not None:
                    with gzip.GzipFile(loss_mask[i], "r") as file:
                        current_loss_mask = np.load(file)  # noqa

                    current_array = current_array * current_loss_mask

                current_array = scipy.ndimage.gaussian_filter(current_array,
                                                              sigma=(0.0,
                                                                     parameters.data_gaussian_smooth_sigma_xy,
                                                                     parameters.data_gaussian_smooth_sigma_xy,
                                                                     parameters.data_gaussian_smooth_sigma_z),
                                                              mode="edge")
        else:
            current_array = np.zeros(current_array.shape)

        # if parameters.input_gaussian_weight > 0.0 or parameters.input_poisson_weight > 0.0:
        #     if input_noise is None:
        #         if parameters.noise_path is not None:
        #             input_noise_path = "{0}/noise.npy.gk".format(parameters.noise_path)
        #         else:
        #             input_noise_path = None

        #         if input_noise_path is not None and os.path.exists(input_noise_path):
        #             print("Previous input noise found!")

        #             with gzip.GzipFile(input_noise_path, "r") as file:
        #                 input_noise = np.load(file)  # noqa
        #         else:
        #             if parameters.input_gaussian_weight > 0.0:
        #                 input_noise = np.random.normal(loc=np.mean(current_array), scale=np.std(current_array),
        #                                                size=current_array.shape)
        #             else:
        #                 if parameters.input_poisson_weight > 0.0:
        #                     input_noise = np.ones(current_array.shape)
        #                     input_noise = (input_noise / np.sum(input_noise)) * np.sum(current_array)

        #                     input_noise = np.random.poisson(lam=input_noise, size=current_array.shape)

        #             if data_mask is not None:
        #                 with gzip.GzipFile(data_mask[i], "r") as file:
        #                     current_data_mask = np.load(file)  # noqa

        #                 input_noise_mean = np.mean(input_noise)
        #                 input_noise_std = np.std(input_noise)

        #                 data_mask_noise = input_noise * current_data_mask

        #                 data_mask_noise = (data_mask_noise - np.mean(data_mask_noise)) / np.std(data_mask_noise)
        #                 data_mask_noise = (data_mask_noise * input_noise_std) + input_noise_mean
        #             else:
        #                 data_mask_noise = input_noise

                    # if loss_mask is not None:
                    #     with gzip.GzipFile(loss_mask[i], "r") as file:
                    #         current_loss_mask = np.load(file)  # noqa

                    #     loss_mask_current_array = input_noise * ((current_loss_mask * -1.0) + 1.0)
                    # else:
                    #     loss_mask_current_array = 0.0

                    # input_noise = data_mask_noise + loss_mask_current_array

        #             input_noise = data_mask_noise

        #             if parameters.noise_path is not None:
        #                 input_noise_path = "{0}/noise.npy.gk".format(output_path)

        #                 with gzip.GzipFile(input_noise_path, "w") as file:
        #                     np.save(file, input_noise)  # noqa

        #     if parameters.input_gaussian_weight > 0.0:
                # current_array = ((current_array + (parameters.input_gaussian_weight * input_noise)) /
                #                  (1.0 + parameters.input_gaussian_weight))

        #         pass
        #     else:
        #         if parameters.input_poisson_weight > 0.0:
        #             current_array = ((current_array + (parameters.input_poisson_weight * input_noise)) /
        #                              (1.0 + parameters.input_poisson_weight))

        #     if parameters.noise_path is not None:
        #         input_noise = None

        current_x_train_path = "{0}/{1}.npy.gz".format(x_train_output_path, str(i))

        with gzip.GzipFile(current_x_train_path, "w") as file:
            np.save(file, current_array)  # noqa

        x.append(current_x_train_path)

    gt_path = "{0}/gt/".format(data_path)

    if os.path.exists(gt_path):
        gt_files = os.listdir(gt_path)
        gt_files.sort(key=human_sorting)
        gt_files = ["{0}{1}".format(gt_path, s) for s in gt_files]

        gt = []

        gt_train_output_path = "{0}/gt_train".format(output_path)

        if not os.path.exists(gt_train_output_path):
            os.makedirs(gt_train_output_path, mode=0o770)

        for i in range(len(gt_files)):
            current_array = nib.load(gt_files[i]).get_fdata()

            if parameters.data_crop_bool:
                current_array = \
                    np.squeeze(preprocessing.data_downsampling_crop([current_array], "numpy",
                                                                    [current_array.shape[0] -
                                                                     parameters.data_crop_amount[0],
                                                                     current_array.shape[1] -
                                                                     parameters.data_crop_amount[1],
                                                                     current_array.shape[2] -
                                                                     parameters.data_crop_amount[2]])[0])

            # current_array = normalise_voxel_sizes(current_array, voxel_sizes)
            current_array, _ = get_data_windows(current_array)

            if data_mask is not None:
                with gzip.GzipFile(data_mask[i], "r") as file:
                    current_data_mask = np.load(file)  # noqa

                current_array = current_array * current_data_mask

            current_gt_train_path = "{0}/{1}.npy.gz".format(gt_train_output_path, str(i))

            with gzip.GzipFile(current_gt_train_path, "w") as file:
                np.save(file, current_array)  # noqa

            gt.append(current_gt_train_path)

        gt = np.asarray(gt)
    else:
        gt = None

    return x, y, example_data, high_resolution_input_shape, full_current_shape, windowed_full_input_axial_size, current_shape, gt, data_mask, loss_mask


def get_preprocessed_train_data():
    print("get_preprocessed_train_data")

    x, y, example_data, high_resolution_input_shape, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt, data_mask, loss_mask = \
        get_train_data()

    x = preprocessing.data_upsample_pad(x, "path")
    y = preprocessing.data_upsample_pad(y, "path")

    if gt is not None:
        gt = preprocessing.data_upsample_pad(gt, "path")

    if data_mask is not None:
        data_mask = preprocessing.data_upsample_pad(data_mask, "path")

    if loss_mask is not None:
        loss_mask = preprocessing.data_upsample_pad(loss_mask, "path")

    x, x_preprocessing_steps = preprocessing.data_preprocessing(x, "path", True, False)
    y, y_preprocessing_steps = preprocessing.data_preprocessing(y, "path", False, False)

    gt_preprocessing_steps = None

    if gt is not None:
        gt, gt_preprocessing_steps = preprocessing.data_preprocessing(gt, "path", False, False)

    return x, y, example_data, x_preprocessing_steps, y_preprocessing_steps, gt_preprocessing_steps, high_resolution_input_shape, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt, data_mask, loss_mask


# https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    print("get_model_memory_usage")

    shapes_mem_count = 0
    internal_model_mem_count = 0

    for l in model.layers:
        layer_type = l.__class__.__name__

        if layer_type == "Model":
            internal_model_mem_count += get_model_memory_usage(batch_size, l)

        single_layer_mem = 1
        out_shape = l.output_shape

        if type(out_shape) is list:
            out_shape = out_shape[0]

        for s in out_shape:
            if s is None:
                continue

            single_layer_mem = single_layer_mem * s

        shapes_mem_count = shapes_mem_count + single_layer_mem

    trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0

    if tf.keras.backend.floatx() == "float16":
        number_size = 2.0

    if tf.keras.backend.floatx() == "float64":
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count

    return gbytes


def get_bayesian_train_prediction(model, x_train_iteration):
    x_prediction_1 = model(x_train_iteration, training=True)

    gc.collect()
    tf.keras.backend.clear_session()

    x_prediction = tf.math.reduce_mean([x_prediction_1[0]], axis=0)
    x_latent = tf.math.reduce_mean([x_prediction_1[1]], axis=0)

    x_prediction_uncertainty = tf.math.reduce_mean(tf.math.reduce_std([x_prediction_1[0]], axis=0))
    x_latent_uncertainty = tf.math.reduce_mean(tf.math.reduce_std([x_prediction_1[1]], axis=0))

    return x_prediction, x_prediction_uncertainty, x_latent, x_latent_uncertainty


def train_backup(model, optimiser, loss_list, previous_model_weight_list, previous_optimiser_weight_list,
                 current_loss_increase_patience):
    print("train_backup")

    print("WARNING: Loss increased above threshold or has become NaN; backing up...")

    if parameters.backtracking_weight_percentage is None:
        raise Exception("Model not saved")

    tape = None

    if len(loss_list) > 1:
        loss_list.pop()
        previous_model_weight_list.pop()
        previous_optimiser_weight_list.pop()

    with open(previous_model_weight_list[-1], "rb") as file:
        model.set_weights(pickle.load(file))

    for layer in model.trainable_weights:
        layer.assign_add(np.random.normal(loc=0.0, scale=parameters.backtracking_weight_perturbation, size=layer.shape))

    try:
        with open(previous_optimiser_weight_list[-1], "rb") as file:
            optimiser.set_weights(pickle.load(file))
    except:
        optimiser = architecture.get_optimiser()

    if len(loss_list) > 1:
        loss_list.pop()
        previous_model_weight_list.pop()
        previous_optimiser_weight_list.pop()

    current_loss_increase_patience = current_loss_increase_patience + 1

    return tape, model, optimiser, loss_list, previous_model_weight_list, previous_optimiser_weight_list, current_loss_increase_patience


def train_step(model, optimiser, loss, x_train_iteration, y_train_iteration, loss_mask_train_iteration, loss_list,
               previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list,
               model_output_path):
    current_loss_increase_patience = 0

    while True:
        previous_model_weights = model.get_weights()
        previous_optimiser_weights = optimiser.get_weights()

        if parameters.backtracking_weight_percentage is not None:
            previous_model_weight_list.append("{0}/model_{1}.pkl".format(model_output_path, str(len(previous_model_weight_list))))

            with open(previous_model_weight_list[-1], "wb") as file:
                pickle.dump(previous_model_weights, file)

            previous_optimiser_weight_list.append("{0}/optimiser_{1}.pkl".format(model_output_path, str(len(previous_optimiser_weight_list))))

            with open(previous_optimiser_weight_list[-1], "wb") as file:
                pickle.dump(previous_optimiser_weights, file)

        x_train_iteration_jitter, y_train_iteration_jitter, loss_mask_train_iteration = \
            preprocessing.introduce_jitter(x_train_iteration, y_train_iteration, loss_mask_train_iteration)

        if bfloat_sixteen_bool:
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                tape.reset()

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x_prediction, x_prediction_uncertainty, x_latent, x_latent_uncertainty = get_bayesian_train_prediction(model, x_train_iteration_jitter)  # Logits for this minibatch

                # current_y_train_iteration = y_train_iteration * loss_mask_train_iteration
                # x_prediction = x_prediction * loss_mask_train_iteration

                # Compute the loss value for this minibatch.
                # current_loss = tf.math.reduce_sum([loss(current_y_train_iteration, x_prediction),
                #                                    parameters.uncertainty_weight *
                #                                    tf.cast(x_prediction_uncertainty, dtype=tf.float32),
                #                                    losses.scale_regulariser(y_train_iteration_jitter, x_prediction),
                #                                    losses.covariance_regulariser(x_latent),
                #                                    parameters.kernel_regulariser_weight *
                #                                    tf.math.reduce_mean(model.losses[::2]),
                #                                    parameters.activity_regulariser_weight *
                #                                    tf.math.reduce_mean(model.losses[1::2])])

                # current_loss = loss(y_train_iteration, x_prediction)

                current_loss = tf.math.reduce_sum([loss(y_train_iteration, x_prediction),
                                                   tf.math.reduce_sum(model.losses)])

                # current_loss = tf.math.reduce_sum([loss(current_y_train_iteration, x_prediction),
                #                                    losses.covariance_regulariser(x_latent),
                #                                    tf.math.reduce_sum(model.losses)])
        else:
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                tape.reset()

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x_prediction, x_prediction_uncertainty, x_latent, x_latent_uncertainty = get_bayesian_train_prediction(model, x_train_iteration_jitter)  # Logits for this minibatch

                # current_y_train_iteration = y_train_iteration * loss_mask_train_iteration
                # x_prediction = x_prediction * loss_mask_train_iteration

                # Compute the loss value for this minibatch.
                # current_loss = tf.math.reduce_sum([loss(current_y_train_iteration, x_prediction),
                #                                    parameters.uncertainty_weight *
                #                                    tf.cast(x_prediction_uncertainty, dtype=tf.float32),
                #                                    losses.scale_regulariser(y_train_iteration_jitter, x_prediction),
                #                                    losses.covariance_regulariser(x_latent),
                #                                    parameters.kernel_regulariser_weight *
                #                                    tf.math.reduce_mean(model.losses[::2]),
                #                                    parameters.activity_regulariser_weight *
                #                                    tf.math.reduce_mean(model.losses[1::2])])

                # current_loss = loss(y_train_iteration, x_prediction)

                current_loss = tf.math.reduce_sum([loss(y_train_iteration, x_prediction),
                                                   tf.math.reduce_sum(model.losses)])

                # current_loss = tf.math.reduce_sum([loss(y_train_iteration, x_prediction),
                #                                    losses.covariance_regulariser(x_latent),
                #                                    tf.math.reduce_sum(model.losses)])

                current_loss = optimiser.get_scaled_loss(current_loss)

        loss_list.append(current_loss)

        if np.isnan(loss_list[-1].numpy()) or np.isinf(loss_list[-1].numpy()):
            tape, model, optimiser, loss_list, previous_model_weight_list, previous_optimiser_weight_list, current_loss_increase_patience = \
                train_backup(model, optimiser, loss_list, previous_model_weight_list, previous_optimiser_weight_list,
                             current_loss_increase_patience)

            continue

        if len(loss_list) > 1:
            current_patience = parameters.patience

            if current_patience < 2:
                current_patience = 2

            loss_gradient = np.gradient(loss_list[-current_patience:])[-1]

            if not np.allclose(loss_gradient, np.zeros(loss_gradient.shape), rtol=0.0, atol=parameters.plateau_cutoff):
                if parameters.backtracking_weight_percentage is not None:
                    if not (loss_list[-1] * (parameters.backtracking_weight_percentage / 100.0) < loss_list[-2]):
                        if current_loss_increase_patience >= parameters.patience:
                            print("WARNING: Loss increased above threshold; patience reached, allowing anyway!")

                            break
                        else:
                            tape, model, optimiser, loss_list, previous_model_weight_list, previous_optimiser_weight_list, current_loss_increase_patience = \
                                train_backup(model, optimiser, loss_list, previous_model_weight_list,
                                             previous_optimiser_weight_list, current_loss_increase_patience)
                    else:
                        break
                else:
                    break
            else:
                break
        else:
            break

    gc.collect()
    tf.keras.backend.clear_session()

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(current_loss, model.trainable_weights)

    del tape

    gc.collect()
    tf.keras.backend.clear_session()

    grads = optimiser.get_unscaled_gradients(grads)

    grads, _ = tf.clip_by_global_norm(grads, 1.0)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimiser.apply_gradients(zip(grads, model.trainable_weights))

    gc.collect()
    tf.keras.backend.clear_session()

    if parameters.model_average_bool:
        average_model_list.append(model.get_weights())
        average_optimiser_list.append(optimiser.get_weights())

        average_optimiser_list_length = len(average_optimiser_list)

        if average_optimiser_list_length > 1:
            for i in range(average_optimiser_list_length):
                for j in range(i, average_optimiser_list_length):
                    if (np.array(average_optimiser_list[i], dtype=object).shape !=
                            np.array(average_optimiser_list[j], dtype=object).shape):
                        average_optimiser_list[i] = average_optimiser_list[j]

            if all([np.allclose(x, y) for x, y in zip(average_model_list[-1], average_model_list[-2])]):
                average_model_list.pop()
                average_optimiser_list.pop()
            else:
                if parameters.model_average_gaussian_sigma > 0.0:
                    truncated_half_normal_kernel = \
                        [1.0 / (parameters.model_average_gaussian_sigma * math.sqrt(2.0 * math.pi)) *
                         math.exp(
                             -math.pow(float(x), 2.0) / (2.0 * math.pow(parameters.model_average_gaussian_sigma, 2.0)))
                         for x in range(len(average_model_list))]

                    truncated_half_normal_kernel.reverse()

                    truncated_half_normal_kernel = np.array(truncated_half_normal_kernel)
                    truncated_half_normal_kernel = truncated_half_normal_kernel / np.sum(truncated_half_normal_kernel)

                    if parameters.model_average_window_bool:
                        if len(average_model_list) >= parameters.model_average_window_length:
                            model.set_weights(np.sum(np.array(average_model_list, dtype=object) *
                                                     np.expand_dims(truncated_half_normal_kernel, axis=-1), axis=0))
                            optimiser.set_weights(np.sum(np.array(average_optimiser_list, dtype=object) *
                                                         np.expand_dims(truncated_half_normal_kernel, axis=-1), axis=0))

                            average_model_list = []
                            average_optimiser_list = []
                        else:
                            model.set_weights(previous_model_weights)

                            if (np.array(previous_optimiser_weights, dtype=object).shape ==
                                    np.array(optimiser.get_weights(), dtype=object).shape):
                                optimiser.set_weights(previous_optimiser_weights)
                    else:
                        model.set_weights(np.sum(np.array(average_model_list, dtype=object) *
                                                 np.expand_dims(truncated_half_normal_kernel, axis=-1), axis=0))
                        optimiser.set_weights(np.sum(np.array(average_optimiser_list, dtype=object) *
                                                     np.expand_dims(truncated_half_normal_kernel, axis=-1), axis=0))
                else:
                    if parameters.model_average_window_bool:
                        if len(average_model_list) >= parameters.model_average_window_length:
                            model.set_weights(np.mean(np.array(average_model_list, dtype=object), axis=0))
                            optimiser.set_weights(np.mean(np.array(average_optimiser_list, dtype=object), axis=0))

                            average_model_list = []
                            average_optimiser_list = []
                        else:
                            model.set_weights(previous_model_weights)

                            if (np.array(previous_optimiser_weights, dtype=object).shape ==
                                    np.array(optimiser.get_weights(), dtype=object).shape):
                                optimiser.set_weights(previous_optimiser_weights)
                    else:
                        model.set_weights(np.mean(np.array(average_model_list, dtype=object), axis=0))
                        optimiser.set_weights(np.mean(np.array(average_optimiser_list, dtype=object), axis=0))

                if parameters.model_average_accumulate_bool:
                    average_model_list.pop()
                    average_model_list.append(model.get_weights())

                    average_optimiser_list.pop()
                    average_optimiser_list.append(optimiser.get_weights())

                if len(average_model_list) > parameters.model_average_window_length:
                    average_model_list.pop(0)
                    average_optimiser_list.pop(0)

        gc.collect()
        tf.keras.backend.clear_session()

    # return model, optimiser, loss_list, x_prediction_uncertainty, previous_model_weight_list, previous_optimiser_weight_list
    return model, optimiser, loss_list, tf.constant(0.0, dtype=tf.float32), previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list


def get_bayesian_test_prediction(model, x_train_iteration, bayesian_iteration, bayesian_bool):
    print("get_bayesian_test_prediction")

    x_prediction_list = []

    if not bayesian_bool:
        bayesian_iteration = 1

    for i in range(bayesian_iteration):
        x_prediction_list.append(model(x_train_iteration, training=bayesian_bool))

        gc.collect()
        tf.keras.backend.clear_session()

    x_prediction = []
    x_latent = []

    for i in range(len(x_prediction_list)):
        x_prediction.append(x_prediction_list[i][0])
        x_latent.append(x_prediction_list[i][1])

    x_prediction_uncertainty_volume = tf.math.reduce_std(x_prediction, axis=0)
    x_latent_uncertainty_volume = tf.math.reduce_std(x_latent, axis=0)

    x_prediction = tf.math.reduce_mean(x_prediction, axis=0)
    x_latent = tf.math.reduce_mean(x_latent, axis=0)

    x_prediction_uncertainty = tf.math.reduce_mean(x_prediction_uncertainty_volume)
    x_latent_uncertainty = tf.math.reduce_mean(x_latent_uncertainty_volume)

    return x_prediction, x_prediction_uncertainty, x_prediction_uncertainty_volume, x_latent, x_latent_uncertainty, x_latent_uncertainty_volume


def test_step(model, loss, x_train_iteration, y_train_iteration, loss_mask_train_iteration, evaluation_loss_list):
    print("test_step")

    x_prediction, x_prediction_uncertainty, x_prediction_uncertainty_volume, x_latent, x_latent_uncertainty, x_latent_uncertainty_volume = \
        get_bayesian_test_prediction(model, x_train_iteration, parameters.bayesian_iterations,
                                     parameters.bayesian_test_bool)

    current_y_train_iteration = y_train_iteration * loss_mask_train_iteration
    current_x_prediction = x_prediction * loss_mask_train_iteration

    evaluation_loss_list.append(loss(current_y_train_iteration, current_x_prediction))

    current_accuracy = [losses.correlation_coefficient_accuracy(current_y_train_iteration, current_x_prediction),
                        losses.scale_accuracy(current_y_train_iteration, current_x_prediction)]

    return x_prediction, evaluation_loss_list, x_prediction_uncertainty, x_prediction_uncertainty_volume, current_accuracy


def test_backup(model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list, average_model_list,
                previous_optimiser_weight_list, average_optimiser_list):
    print("train_backup")

    print("WARNING: Accuracy has become NaN; backing up...")

    if parameters.backtracking_weight_percentage is None:
        raise Exception("Model not saved")

    if len(loss_list) > 1:
        loss_list.pop()
        evaluation_loss_list.pop()
        previous_model_weight_list.pop()

        if len(average_model_list) > 0:
            average_model_list.pop()

        previous_optimiser_weight_list.pop()

        if len(average_optimiser_list) > 0:
            average_optimiser_list.pop()

    with open(previous_model_weight_list[-1], "rb") as file:
        model.set_weights(pickle.load(file))

    for layer in model.trainable_weights:
        layer.assign_add(np.random.normal(loc=0.0, scale=parameters.backtracking_weight_perturbation, size=layer.shape))

    try:
        with open(previous_optimiser_weight_list[-1], "rb") as file:
            optimiser.set_weights(pickle.load(file))
    except:
        optimiser = architecture.get_optimiser()

    if len(loss_list) > 1:
        loss_list.pop()
        evaluation_loss_list.pop()
        previous_model_weight_list.pop()

        if len(average_model_list) > 0:
            average_model_list.pop()

        previous_optimiser_weight_list.pop()

    return model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list


def output_window_predictions(x_prediction, y_train_iteration, gt_prediction, loss_mask_train_iteration,
                              x_prediction_uncertainty_volume, x_preprocessing_steps, y_preprocessing_steps,
                              gt_preprocessing_steps, window_input_shape, current_output_path, j):
    print("output_window_predictions")

    x_prediction = preprocessing.data_preprocessing([x_prediction], "numpy", False, False, y_preprocessing_steps)[0][0]
    # x_prediction = preprocessing.mask_fov(x_prediction)
    x_prediction = np.squeeze(preprocessing.data_downsampling_crop([x_prediction], "numpy", window_input_shape)[0])

    x_prediction_uncertainty_volume = preprocessing.data_preprocessing([x_prediction_uncertainty_volume], "numpy",
                                                                       False, True, y_preprocessing_steps)[0][0]
    # x_prediction_uncertainty_volume = preprocessing.mask_fov(x_prediction_uncertainty_volume)
    x_prediction_uncertainty_volume = np.squeeze(preprocessing.data_downsampling_crop([x_prediction_uncertainty_volume],
                                                                                      "numpy", window_input_shape)[0])

    inverse_loss_mask_train_iteration = (loss_mask_train_iteration * -1.0) + 1.0

    if tf.reduce_sum(tf.cast(inverse_loss_mask_train_iteration, dtype=tf.float32)) > 0.0:
        inverse_loss_mask_train_iteration = \
            np.squeeze(preprocessing.data_downsampling_crop([inverse_loss_mask_train_iteration], "numpy",
                                                            window_input_shape)[0])

        current_x_prediction = x_prediction * inverse_loss_mask_train_iteration
    else:
        current_x_prediction = x_prediction

    if gt_prediction is not None:
        gt_prediction = preprocessing.data_preprocessing([gt_prediction], "numpy", False, False,
                                                         gt_preprocessing_steps)[0][0]
        gt_prediction = np.squeeze(preprocessing.data_downsampling_crop([gt_prediction], "numpy",
                                                                        window_input_shape)[0])

        if tf.reduce_sum(tf.cast(inverse_loss_mask_train_iteration, dtype=tf.float32)) > 0.0:
            current_gt_prediction = gt_prediction * inverse_loss_mask_train_iteration
        else:
            current_gt_prediction = gt_prediction

        current_output_accuracy = \
            [losses.correlation_coefficient_accuracy(current_gt_prediction, current_x_prediction).numpy(),
             losses.scale_accuracy(current_gt_prediction, current_x_prediction).numpy()]
    else:
        y_train_iteration = preprocessing.data_preprocessing([y_train_iteration], "numpy", False, False,
                                                             y_preprocessing_steps)[0][0]
        y_train_iteration = np.squeeze(preprocessing.data_downsampling_crop([y_train_iteration], "numpy",
                                                                            window_input_shape)[0])

        if tf.reduce_sum(tf.cast(inverse_loss_mask_train_iteration, dtype=tf.float32)) > 0.0:
            current_y_train_iteration = y_train_iteration * inverse_loss_mask_train_iteration
        else:
            current_y_train_iteration = y_train_iteration

        current_output_accuracy = \
            [losses.correlation_coefficient_accuracy(current_y_train_iteration, current_x_prediction).numpy(),
             losses.scale_accuracy(current_y_train_iteration, current_x_prediction).numpy()]

    print("Output accuracy:\t{0:<20}".format(str(current_output_accuracy[0])))
    print("Output scale accuracy:\t{0:<20}".format(str(current_output_accuracy[1])))
    print("Output total accuracy:\t{0:<20}".format(str(np.mean(current_output_accuracy))))

    print("Output uncertainty:\t{0:<20}".format(str(tf.math.reduce_mean(x_prediction_uncertainty_volume).numpy())))

    output_volume = nib.Nifti1Image(x_prediction, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}_output.nii.gz".format(current_output_path, str(j))
    nib.save(output_volume, current_data_path)

    output_uncertainty_volume = nib.Nifti1Image(x_prediction_uncertainty_volume, np.eye(4), nib.Nifti1Header())

    current_uncertainty_data_path = "{0}/{1}_uncertainty.nii.gz".format(current_output_path, str(j))
    nib.save(output_uncertainty_volume, current_uncertainty_data_path)

    return current_data_path, current_uncertainty_data_path, current_output_accuracy


def output_patient_time_point_predictions(window_data_paths, example_data, windowed_full_input_axial_size,
                                          high_resolution_input_shape, full_input_shape, current_output_path,
                                          current_output_prefix, i):
    print("output_patient_time_point_predictions")

    if len(window_data_paths) > 1:
        number_of_windows = int(np.ceil(full_input_shape[-1] / parameters.data_window_size)) + 1
        number_of_overlaps = number_of_windows - 1

        overlap_size = int(np.ceil((((parameters.data_window_size * number_of_windows) -
                                     full_input_shape[-1]) / number_of_overlaps)))
        overlap_index = int(np.ceil(parameters.data_window_size - overlap_size))

        output_ramp_filter_increasing = np.zeros((full_input_shape[0], full_input_shape[1], overlap_size))
        output_ramp_filter_increment = 1.0 / (overlap_size + 1.0)

        for l in range(overlap_size):
            output_ramp_filter_increasing[:, :, l] = output_ramp_filter_increment * (l + 1.0)

        output_ramp_filter_decreasing = np.flip(output_ramp_filter_increasing, axis=-1)

        output_arrays = []

        for l in range(number_of_windows):
            current_overlap_index = overlap_index * l

            current_data = nib.load(window_data_paths[l]).get_fdata()

            if l != 0:
                current_data[:, :, :overlap_size] = current_data[:, :, :overlap_size] * output_ramp_filter_increasing

            if l != number_of_windows - 1:
                current_data[:, :, -overlap_size:] = current_data[:, :, -overlap_size:] * output_ramp_filter_decreasing

            current_output_array = np.zeros((full_input_shape[0], full_input_shape[1], windowed_full_input_axial_size))
            current_output_array[:, :, current_overlap_index:current_overlap_index + parameters.data_window_size] = \
                current_data

            current_output_array[np.isclose(current_output_array, 0.0)] = np.nan

            output_arrays.append(current_output_array)

        output_array = np.nansum(np.asarray(output_arrays), axis=0)
        output_array = np.nan_to_num(output_array, copy=False)
    else:
        output_array = nib.load(window_data_paths[0]).get_fdata()

    output_array = np.squeeze(preprocessing.data_downsampling([output_array], "numpy", full_input_shape)[0])
    output_array = np.squeeze(preprocessing.data_upsample_pad([output_array], "numpy", high_resolution_input_shape,
                                                              "constant")[0])

    output_volume = nib.Nifti1Image(output_array, example_data.affine, example_data.header)

    current_data_path = "{0}/{1}_{2}.nii.gz".format(current_output_path, str(i), current_output_prefix)
    nib.save(output_volume, current_data_path)

    return current_data_path


def train_model():
    print("train_model")

    # get data and lables
    x, y, example_data, x_preprocessing_steps, y_preprocessing_steps, gt_preprocessing_steps, high_resolution_input_shape, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt, data_mask, loss_mask = \
        get_preprocessed_train_data()

    with gzip.GzipFile(x[0], "r") as file:
        input_shape = np.load(file)[0].shape  # noqa

    model, optimiser, loss = architecture.get_model_all(input_shape)
    model.summary()

    model_update_path = "{0}/model.pkl".format(model_path)

    if os.path.exists(model_update_path):
        print("Previous model found!")

        with open(model_update_path, "rb") as file:
            model.set_weights(pickle.load(file))

    print("Memory usage:\t{0}".format(str(get_model_memory_usage(1, model))))

    tf.keras.utils.plot_model(model, to_file="{0}/model.pdf".format(output_path), show_shapes=True, show_dtype=True,
                              show_layer_names=True, expand_nested=True)

    output_paths = []
    output_uncertainty_paths = []
    output_accuracies = []

    average_model_list = []
    average_optimiser_list = []

    for i in range(len(y)):
        if parameters.new_model_patient_bool and not parameters.new_model_window_bool:
            if os.path.exists(model_update_path):
                print("Previous model found!")

                with open(model_update_path, "rb") as file:
                    model.set_weights(pickle.load(file))
            else:
                del model

                gc.collect()
                tf.keras.backend.clear_session()

                model = architecture.get_model(input_shape)

                average_model_list = []

        if parameters.new_optimiser_patient_bool and not parameters.new_optimiser_window_bool:
            del optimiser

            gc.collect()
            tf.keras.backend.clear_session()

            optimiser = architecture.get_optimiser()

            average_optimiser_list = []

        print("Patient/Time point:\t{0}".format(str(i)))

        patient_output_path = "{0}/{1}".format(output_path, str(i))

        # create output directory
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path, mode=0o770)

        with gzip.GzipFile(x[i], "r") as file:
            number_of_windows = np.load(file).shape[0]  # noqa

        current_window_data_paths = []
        current_window_uncertainty_data_paths = []
        current_output_accuracies = []

        for j in range(number_of_windows):
            if parameters.new_model_window_bool:
                if os.path.exists(model_update_path):
                    print("Previous model found!")

                    with open(model_update_path, "rb") as file:
                        model.set_weights(pickle.load(file))
                else:
                    del model

                    gc.collect()
                    tf.keras.backend.clear_session()

                    model = architecture.get_model(input_shape)

                    average_model_list = []

            if parameters.new_optimiser_window_bool:
                del optimiser

                gc.collect()
                tf.keras.backend.clear_session()

                optimiser = architecture.get_optimiser()

                average_optimiser_list = []

            print("Window:\t{0}".format(str(j)))

            window_output_path = "{0}/{1}".format(patient_output_path, str(j))

            # create output directory
            if not os.path.exists(window_output_path):
                os.makedirs(window_output_path, mode=0o770)

            plot_output_path = "{0}/plots".format(window_output_path)

            # create output directory
            if not os.path.exists(plot_output_path):
                os.makedirs(plot_output_path, mode=0o770)

            model_output_path = "{0}/models".format(window_output_path)

            # create output directory
            if not os.path.exists(model_output_path):
                os.makedirs(model_output_path, mode=0o770)

            with gzip.GzipFile(x[i], "r") as file:
                x_train_iteration = np.asarray([np.load(file)[j]])  # noqa

            with gzip.GzipFile(y[i], "r") as file:
                y_train_iteration = np.asarray([np.load(file)[j]])  # noqa

            x_train_iteration = tf.convert_to_tensor(x_train_iteration)
            y_train_iteration = tf.convert_to_tensor(y_train_iteration)

            if float_sixteen_bool:
                if bfloat_sixteen_bool:
                    x_train_iteration = tf.cast(x_train_iteration, dtype=tf.bfloat16)
                    y_train_iteration = tf.cast(y_train_iteration, dtype=tf.bfloat16)
                else:
                    x_train_iteration = tf.cast(x_train_iteration, dtype=tf.float16)
                    y_train_iteration = tf.cast(y_train_iteration, dtype=tf.float16)
            else:
                x_train_iteration = tf.cast(x_train_iteration, dtype=tf.float32)
                y_train_iteration = tf.cast(y_train_iteration, dtype=tf.float32)

            if gt is not None:
                with gzip.GzipFile(gt[i], "r") as file:
                    gt_prediction = np.asarray([np.load(file)[j]])  # noqa

                gt_prediction = tf.convert_to_tensor(gt_prediction)

                if float_sixteen_bool:
                    if bfloat_sixteen_bool:
                        gt_prediction = tf.cast(gt_prediction, dtype=tf.bfloat16)
                    else:
                        gt_prediction = tf.cast(gt_prediction, dtype=tf.float16)
                else:
                    gt_prediction = tf.cast(gt_prediction, dtype=tf.float32)
            else:
                gt_prediction = None

            if loss_mask is not None:
                with gzip.GzipFile(loss_mask[i], "r") as file:
                    loss_mask_train_iteration = np.asarray([np.load(file)[j]])  # noqa

                loss_mask_train_iteration = tf.convert_to_tensor(loss_mask_train_iteration)

                if float_sixteen_bool:
                    if bfloat_sixteen_bool:
                        loss_mask_train_iteration = tf.cast(loss_mask_train_iteration, dtype=tf.bfloat16)
                    else:
                        loss_mask_train_iteration = tf.cast(loss_mask_train_iteration, dtype=tf.float16)
                else:
                    loss_mask_train_iteration = tf.cast(loss_mask_train_iteration, dtype=tf.float32)
            else:
                loss_mask_train_iteration = None

            total_iterations = 0

            loss_list = []
            evaluation_loss_list = []
            evaluation_accuracy_list = []
            previous_model_weight_list = []
            previous_optimiser_weight_list = []

            current_accuracy_nan_patience = 0

            total_gt_current_accuracy = None
            max_accuracy = tf.cast(0.0, dtype=tf.float64)
            max_accuracy_iteration = 0

            x_output = None
            y_output = None
            x_prediction_uncertainty_volume = None

            while True:
                if current_accuracy_nan_patience >= parameters.patience:
                    break

                print("Iteration:\t{0:<20}\tTotal iterations:\t{1:<20}".format(str(len(loss_list)), str(total_iterations)))

                model, optimiser, loss_list, uncertainty, previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list = \
                    train_step(model, optimiser, loss, x_train_iteration, y_train_iteration, loss_mask_train_iteration,
                               loss_list, previous_model_weight_list, average_model_list,
                               previous_optimiser_weight_list, average_optimiser_list, model_output_path)

                x_prediction, evaluation_loss_list, x_prediction_uncertainty, x_prediction_uncertainty_volume, current_accuracy = \
                    test_step(model, loss, x_train_iteration, y_train_iteration, loss_mask_train_iteration,
                              evaluation_loss_list)

                if ((np.isnan(current_accuracy[0].numpy()) or np.isinf(current_accuracy[0].numpy())) or
                        (np.isnan(current_accuracy[1].numpy()) or np.isinf(current_accuracy[1].numpy()))):
                    model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list = \
                        test_backup(model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list,
                                    average_model_list, previous_optimiser_weight_list, average_optimiser_list)

                    current_accuracy_nan_patience = current_accuracy_nan_patience + 1

                    continue
                else:
                    if gt is None:
                        current_accuracy_nan_patience = 0

                total_current_accuracy = tf.math.reduce_mean([tf.cast(current_accuracy[0], dtype=tf.float64),
                                                              tf.cast(current_accuracy[1], dtype=tf.float64)])

                evaluation_accuracy_list.append(total_current_accuracy)

                print("Loss:\t{0:<20}\tAccuracy:\t{1:<20}\tScale accuracy:\t{2:<20}\tTotal accuracy:\t{3:<20}\tUncertainty:\t{4:<20}".format(str(loss_list[-1].numpy()), str(current_accuracy[0].numpy()), str(current_accuracy[1].numpy()), str(total_current_accuracy.numpy()), str(uncertainty.numpy())))

                if gt is not None:
                    inverse_loss_mask_train_iteration = (loss_mask_train_iteration * -1.0) + 1.0

                    if tf.reduce_sum(tf.cast(inverse_loss_mask_train_iteration, dtype=tf.float32)) > 0.0:
                        current_gt_prediction = gt_prediction * inverse_loss_mask_train_iteration
                        current_x_prediction = x_prediction * inverse_loss_mask_train_iteration
                    else:
                        current_gt_prediction = gt_prediction
                        current_x_prediction = x_prediction

                    gt_accuracy = [losses.correlation_coefficient_accuracy(current_gt_prediction, current_x_prediction),
                                   losses.scale_accuracy(current_gt_prediction, current_x_prediction)]

                    if ((np.isnan(gt_accuracy[0].numpy()) or np.isinf(gt_accuracy[0].numpy())) or
                            (np.isnan(gt_accuracy[1].numpy()) or np.isinf(gt_accuracy[1].numpy()))):
                        model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list = \
                            test_backup(model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list,
                                        average_model_list, previous_optimiser_weight_list, average_optimiser_list)

                        current_accuracy_nan_patience = current_accuracy_nan_patience + 1

                        continue
                    else:
                        current_accuracy_nan_patience = 0

                    total_gt_current_accuracy = tf.math.reduce_mean([tf.cast(gt_accuracy[0], dtype=tf.float64),
                                                                     tf.cast(gt_accuracy[1], dtype=tf.float64)])

                    print("GT loss:\t{0:<20}\tGT accuracy:\t{1:<20}\tGT scale accuracy:\t{2:<20}\tGT total accuracy:\t{3:<20}\tGT uncertainty:\t{4:<20}".format(str(evaluation_loss_list[-1].numpy()), str(gt_accuracy[0].numpy()), str(gt_accuracy[1].numpy()), str(total_gt_current_accuracy.numpy()), str(x_prediction_uncertainty.numpy())))

                    if max_accuracy < total_gt_current_accuracy:
                        max_accuracy = total_gt_current_accuracy
                        max_accuracy_iteration = len(loss_list) - 1
                else:
                    if max_accuracy < total_current_accuracy:
                        max_accuracy = total_current_accuracy
                        max_accuracy_iteration = len(loss_list) - 1

                if np.isnan(evaluation_loss_list[-1].numpy()) or np.isinf(evaluation_loss_list[-1].numpy()):
                    model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list, average_model_list, previous_optimiser_weight_list, average_optimiser_list = \
                        test_backup(model, optimiser, loss_list, evaluation_loss_list, previous_model_weight_list,
                                    average_model_list, previous_optimiser_weight_list, average_optimiser_list)

                    continue

                x_output = np.squeeze(x_prediction.numpy()).astype(np.float64)
                y_output = np.squeeze(y_train_iteration.numpy()).astype(np.float64)

                vmin = np.percentile(y_output, 0)
                vmax = np.percentile(y_output, 99.5)

                plt.figure()

                if gt is not None:
                    plt.subplot(2, 3, 1)
                else:
                    plt.subplot(2, 2, 1)

                plt.imshow(x_output[:, :, int(x_output.shape[2] / 2)], cmap="Greys", vmin=vmin, vmax=vmax)

                if gt is not None:
                    plt.subplot(2, 3, 4)
                else:
                    plt.subplot(2, 2, 3)

                plt.imshow(x_output[:, :, int(x_output.shape[2] / 2)], cmap="Greys")

                if gt is not None:
                    plt.subplot(2, 3, 2)
                else:
                    plt.subplot(2, 2, 2)

                plt.imshow(y_output[:, :, int(y_output.shape[2] / 2)], cmap="Greys", vmin=vmin, vmax=vmax)

                if gt is not None:
                    plt.subplot(2, 3, 5)
                else:
                    plt.subplot(2, 2, 4)

                plt.imshow(y_output[:, :, int(y_output.shape[2] / 2)], cmap="Greys")

                if gt is not None:
                    gt_plot = np.squeeze(gt_prediction).astype(np.float64)

                    plt.subplot(2, 3, 3)
                    plt.imshow(gt_plot[:, :, int(gt_plot.shape[2] / 2)], cmap="Greys")

                plt.tight_layout()
                plt.savefig("{0}/{1}.png".format(plot_output_path, str(len(loss_list) - 1)), format="png", dpi=600,
                            bbox_inches="tight")
                plt.close()

                plot_files = os.listdir(plot_output_path)

                for l in range(len(plot_files)):
                    current_plot_file = plot_files[l].strip()

                    split_array = current_plot_file.split(".png")

                    if len(split_array) >= 2:
                        if int(split_array[0]) > len(loss_list) - 1:
                            os.remove("{0}/{1}".format(plot_output_path, current_plot_file))

                model_files = os.listdir(model_output_path)

                for l in range(len(model_files)):
                    current_plot_file = model_files[l].strip()

                    split_array = current_plot_file.split("model_")[-1]
                    split_array = split_array.split("optimiser_")[-1]

                    split_array = split_array.split(".pkl")

                    if len(split_array) >= 2:
                        if int(split_array[0]) > len(loss_list) - 1:
                            os.remove("{0}/{1}".format(model_output_path, current_plot_file))

                evaluation_loss_list_len = len(evaluation_loss_list)

                if (evaluation_loss_list_len > 1 and
                        evaluation_loss_list_len >= parameters.loss_scaling_patience_skip and
                        evaluation_loss_list_len >= parameters.patience and
                        evaluation_accuracy_list[-1] != evaluation_accuracy_list[-2]):
                    current_patience = parameters.patience

                    if current_patience < 2:
                        current_patience = 2

                    loss_gradient = np.gradient(evaluation_loss_list)[-current_patience:]

                    print("Plateau cutoff:\t{0:<20}\tMax distance to plateau cutoff:\t{1:<20}\tMean gradient direction:\t{2:<20}".format(str(parameters.plateau_cutoff), str(np.max(np.abs(loss_gradient) - parameters.plateau_cutoff)), str(np.mean(loss_gradient))))

                    # if (np.allclose(np.abs(loss_gradient), np.zeros(loss_gradient.shape), rtol=0.0,
                    #                 atol=parameters.plateau_cutoff) or
                    #         np.alltrue(loss_gradient - parameters.plateau_cutoff > 0.0)):
                    if np.allclose(np.abs(loss_gradient), np.zeros(loss_gradient.shape), rtol=0.0,
                                   atol=parameters.plateau_cutoff):
                        print("Reached plateau: Exiting...")
                        print("Maximum accuracy:\t{0:<20}\tMaximum accuracy iteration:\t{1:<20}".format(str(max_accuracy.numpy()), str(max_accuracy_iteration)))

                        if gt is not None:
                            accuracy_loss = np.abs(total_gt_current_accuracy - max_accuracy)
                        else:
                            accuracy_loss = np.abs(total_current_accuracy - max_accuracy)

                        print("Accuracy loss:\t{0:<20}".format(str(accuracy_loss)))

                        break

                total_iterations = total_iterations + 1

            if current_accuracy_nan_patience >= parameters.patience:
                print("Error: Input not suitable; continuing")

                continue

            if parameters.bayesian_test_bool != parameters.bayesian_output_bool:
                x_prediction, x_prediction_uncertainty, x_prediction_uncertainty_volume, x_latent, x_latent_uncertainty, x_latent_uncertainty_volume = \
                    get_bayesian_test_prediction(model, x_train_iteration, parameters.bayesian_iterations,
                                                 parameters.bayesian_output_bool)

                x_output = np.squeeze(x_prediction.numpy()).astype(np.float64)

            if gt_prediction is not None:
                gt_prediction = gt_prediction.numpy()

            loss_mask_train_iteration = np.squeeze(loss_mask_train_iteration.numpy()).astype(np.float64)

            current_window_data_path, current_window_uncertainty_data_path, current_output_accuracy = \
                output_window_predictions(x_output, y_output, gt_prediction, loss_mask_train_iteration,
                                          np.squeeze(x_prediction_uncertainty_volume.numpy()).astype(np.float64),
                                          x_preprocessing_steps, y_preprocessing_steps, gt_preprocessing_steps,
                                          window_input_shape, window_output_path, j)

            current_window_data_paths.append(current_window_data_path)
            current_window_uncertainty_data_paths.append(current_window_uncertainty_data_path)
            current_output_accuracies.append(current_output_accuracy)

            if parameters.new_model_window_bool:
                with open(model_path, "wb") as file:  # noqa
                    pickle.dump(model.get_weights(), file)

                # with open("{0}/model.pkl".format(window_output_path), "wb") as file:
                #     pickle.dump(model.get_weights(), file)

                # with open("{0}/optimiser.pkl".format(window_output_path), "wb") as file:
                #     pickle.dump(optimiser.get_weights(), file)

        try:
            output_paths.append(output_patient_time_point_predictions(current_window_data_paths, example_data,
                                                                      windowed_full_input_axial_size,
                                                                      high_resolution_input_shape, full_input_shape,
                                                                      patient_output_path, "output", i))

            output_uncertainty_paths.append(output_patient_time_point_predictions(current_window_uncertainty_data_paths,
                                                                                  example_data,
                                                                                  windowed_full_input_axial_size,
                                                                                  high_resolution_input_shape,
                                                                                  full_input_shape, patient_output_path,
                                                                                  "uncertainty", i))

            output_accuracies.append(current_output_accuracies)

            if parameters.new_model_patient_bool and not parameters.new_model_window_bool:
                with open(model_path, "wb") as file:  # noqa
                    pickle.dump(model.get_weights(), file)

                # with open("{0}/model.pkl".format(patient_output_path), "wb") as file:
                #     pickle.dump(model.get_weights(), file)

                # with open("{0}/optimiser.pkl".format(patient_output_path), "wb") as file:
                #     pickle.dump(optimiser.get_weights(), file)
        except:
            print("Error: Input not suitable; continuing")

    if not parameters.new_model_patient_bool and not parameters.new_model_window_bool:
        with open(model_path, "wb") as file:  # noqa
            pickle.dump(model.get_weights(), file)

        # with open("{0}/model.pkl".format(output_path), "wb") as file:
        #     pickle.dump(model.get_weights(), file)

        # with open("{0}/optimiser.pkl".format(output_path), "wb") as file:
        #     pickle.dump(optimiser.get_weights(), file)

    print("Output accuracy:\t{0:<20}".format(str(np.mean(np.array(output_accuracies)[:, :, 0]))))
    print("Output scale accuracy:\t{0:<20}".format(str(np.mean(np.array(output_accuracies)[:, :, 1]))))
    print("Output total accuracy:\t{0:<20}".format(str(np.mean(np.array(output_accuracies)))))

    return output_paths, output_uncertainty_paths


def main(input_data_path=None, input_output_path=None, input_model_path=None):
    print("main")

    global data_path
    global output_path
    global model_path

    if input_data_path is not None:
        data_path = input_data_path
    else:
        data_path = parameters.data_path

    if input_output_path is not None:
        output_path = input_output_path
    else:
        output_path = parameters.output_path

    if input_model_path is not None:
        model_path = input_model_path
    else:
        model_path = parameters.model_path

    # if debugging, remove previous output directory
    if input_output_path is None:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)

    # create log file and begin writing to it
    logfile_path = "{0}/logfile.log".format(output_path)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)

    # import transcript
    # logfile = transcript.transcript_start(logfile_path)

    output_paths, output_uncertainty_paths = train_model()

    # import python_email_notification
    # python_email_notification.main()

    # device.reset()

    # transcript.transcript_stop(logfile)

    return output_paths, output_uncertainty_paths, model_path


if __name__ == "__main__":
    main()
