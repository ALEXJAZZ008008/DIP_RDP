# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


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
import tensorflow.keras as k
import pickle
import matplotlib.pyplot as plt
import nibabel as nib


reproducible_bool = True

if reproducible_bool:
    # Seed value (can actually be different for each attribution step)
    seed_value = 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

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

float_sixteen_bool = True  # set the network to use float16 data
cpu_bool = False  # if using CPU, set to true: disables mixed precision computation

# mixed precision float16 computation allows the network to use both float16 and float32 where necessary,
# this improves performance on the GPU.
if float_sixteen_bool and not cpu_bool:
    policy = k.mixed_precision.Policy("mixed_float16")
    k.mixed_precision.set_global_policy(policy)
else:
    policy = k.mixed_precision.Policy(tf.dtypes.float32.name)
    k.mixed_precision.set_global_policy(policy)


import parameters
import preprocessing
import architecture


data_path = "{0}/DIP_RDP_data/xcat/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output".format(os.getcwd())


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
            number_of_windows = int(np.ceil(axial_size / parameters.data_window_size))
            number_of_overlaps = number_of_windows - 1

            overlap_size = \
                int(np.ceil((((parameters.data_window_size * number_of_windows) - axial_size) / number_of_overlaps)))
            overlap_index = int(np.ceil(parameters.data_window_size - overlap_size))

            windowed_full_input_axial_size = \
                (number_of_windows * parameters.data_window_size) - (number_of_overlaps * overlap_size)

            data = np.squeeze(preprocessing.data_upsample([np.expand_dims(np.expand_dims(data, 0), -1)], "numpy",
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

    return data, windowed_full_input_axial_size


def get_train_data():
    print("get_train_data")

    y_path = "{0}y/".format(data_path)

    y_files = os.listdir(y_path)
    y_files.sort(key=human_sorting)
    y_files = ["{0}{1}".format(y_path, s) for s in y_files]

    x = []
    y = []

    x_train_output_path = "{0}/x_train".format(output_path)

    if not os.path.exists(x_train_output_path):
        os.makedirs(x_train_output_path, mode=0o770)

    y_train_output_path = "{0}/y_train".format(output_path)

    if not os.path.exists(y_train_output_path):
        os.makedirs(y_train_output_path, mode=0o770)

    full_current_shape = None
    windowed_full_input_axial_size = None
    current_shape = None
    gaussian_noise = None

    for i in range(len(y_files)):
        current_volume = nib.load(y_files[i])
        current_array = current_volume.get_data()

        if full_current_shape is None:
            full_current_shape = current_array.shape

        current_array, windowed_full_input_axial_size = get_data_windows(current_array)

        if current_shape is None:
            current_shape = current_array[0].shape

        current_y_train_path = "{0}/{1}.npy".format(y_train_output_path, str(i))
        np.save(current_y_train_path, current_array)
        y.append(current_y_train_path)

        if parameters.data_input_bool:
            if parameters.data_gaussian_smooth_sigma_xy > 0.0 or parameters.data_gaussian_smooth_sigma_z > 0.0:
                current_array = scipy.ndimage.gaussian_filter(current_array,
                                                              sigma=(0.0,
                                                                     parameters.data_gaussian_smooth_sigma_xy,
                                                                     parameters.data_gaussian_smooth_sigma_xy,
                                                                     parameters.data_gaussian_smooth_sigma_z),
                                                              mode="mirror")

                current_array = preprocessing.redistribute([current_array], "numpy")[0]
        else:
            current_array = np.zeros(current_array.shape)

        if parameters.input_gaussian_weight > 0.0:
            if parameters.data_input_bool:
                current_array, _ = preprocessing.data_preprocessing(current_array, "numpy")

            if gaussian_noise is None:
                gaussian_noise = preprocessing.redistribute([np.random.normal(size=current_array.shape)], "numpy")[0]

            current_array = (current_array + (parameters.input_gaussian_weight * gaussian_noise)) / \
                            (1.0 + parameters.input_gaussian_weight)

        current_x_train_path = "{0}/{1}.npy".format(x_train_output_path, str(i))
        np.save(current_x_train_path, current_array)
        x.append(current_x_train_path)

    y = np.asarray(y)

    gt_path = "{0}gt/".format(data_path)

    if os.path.exists(gt_path):
        gt_files = os.listdir(gt_path)
        gt_files.sort(key=human_sorting)
        gt_files = ["{0}{1}".format(gt_path, s) for s in gt_files]

        gt = []

        gt_train_output_path = "{0}/gt_train".format(output_path)

        if not os.path.exists(gt_train_output_path):
            os.makedirs(gt_train_output_path, mode=0o770)

        for i in range(len(gt_files)):
            current_array = nib.load(gt_files[i]).get_data()

            current_array, _ = get_data_windows(current_array)

            current_gt_train_path = "{0}/{1}.npy".format(gt_train_output_path, str(i))
            np.save(current_gt_train_path, current_array)
            gt.append(current_gt_train_path)

        gt = np.asarray(gt)
    else:
        gt = None

    return x, y, full_current_shape, windowed_full_input_axial_size, current_shape, gt


def get_preprocessed_train_data():
    print("get_preprocessed_train_data")

    x, y, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt = get_train_data()

    x = preprocessing.data_upsample(x, "path")
    y = preprocessing.data_upsample(y, "path")

    if gt is not None:
        gt = preprocessing.data_upsample(gt, "path")

    x, _ = preprocessing.data_preprocessing(x, "path")
    y, preprocessing_steps = preprocessing.data_preprocessing(y, "path")

    if gt is not None:
        gt, _ = preprocessing.data_preprocessing(gt, "path")

    return x, y, preprocessing_steps, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt


# https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    print("get_model_memory_usage")

    shapes_mem_count = 0
    internal_model_mem_count = 0

    for l in model.layers:
        layer_type = l.__class__.__name__

        if layer_type == 'Model':
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

    trainable_count = np.sum([k.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([k.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0

    if k.backend.floatx() == 'float16':
        number_size = 2.0

    if k.backend.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count

    return gbytes


def get_bayesian_train_prediction(model, x_train_iteration, previous_x_prediction_list):
    x_prediction = model(x_train_iteration, training=True)

    previous_x_prediction_list.append(tf.cast(x_prediction, dtype=tf.float32))

    x_prediction_uncertainty = tf.math.reduce_mean(tf.math.reduce_variance(previous_x_prediction_list, axis=0))

    return x_prediction, x_prediction_uncertainty, previous_x_prediction_list


def train_backup(model, optimiser, loss_list, previous_x_prediction_list, previous_model_weight_list,
                 previous_optimiser_weight_list, current_loss_increase_patience):
    print("train_backup")

    print("WARNING: Loss increased above threshold or has become NaN; backing up...")

    if len(loss_list) > 1:
        loss_list.pop()
        previous_x_prediction_list.pop()
        previous_model_weight_list.pop()
        previous_optimiser_weight_list.pop()

    model.set_weights(pickle.load(open(previous_model_weight_list[-1], "rb")))

    if len(previous_optimiser_weight_list) > 1:
        optimiser.set_weights(pickle.load(open(previous_optimiser_weight_list[-1], "rb")))

    current_loss_increase_patience = current_loss_increase_patience + 1

    if len(loss_list) > 0:
        loss_list.pop()
        previous_model_weight_list.pop()
        previous_optimiser_weight_list.pop()

        if len(previous_x_prediction_list) > 0:
            previous_x_prediction_list.pop()

    return model, optimiser, loss_list, previous_x_prediction_list, previous_model_weight_list,\
           previous_optimiser_weight_list, current_loss_increase_patience


def train_step(model, optimiser, loss, x_train_iteration, y_train_iteration, loss_list, previous_x_prediction_list,
               previous_model_weight_list, previous_optimiser_weight_list, relative_plateau_cutoff,
               model_output_path):
    current_loss_increase_patience = 0

    while True:
        previous_model_weight_list.append("{0}/model_{1}.pkl".format(model_output_path, str(len(loss_list))))
        pickle.dump(model.get_weights(), open(previous_model_weight_list[-1], "wb"))

        previous_optimiser_weight_list.append("{0}/optimiser_{1}.pkl".format(model_output_path, str(len(loss_list))))
        pickle.dump(optimiser.get_weights(), open(previous_optimiser_weight_list[-1], "wb"))

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            tape.reset()

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits, uncertainty, previous_x_prediction_list = \
                get_bayesian_train_prediction(model, x_train_iteration, previous_x_prediction_list)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            current_loss = tf.math.reduce_sum([loss(y_train_iteration, logits),
                                               parameters.uncertainty_weight * uncertainty,
                                               tf.math.reduce_mean(tf.where(tf.math.is_nan(model.losses[::3]),
                                                                            tf.zeros_like(model.losses[::3]),
                                                                            model.losses[::3])),
                                               tf.math.reduce_mean([
                                                   tf.math.reduce_mean(tf.where(tf.math.is_nan(model.losses[1::3]),
                                                                                tf.zeros_like(model.losses[1::3]),
                                                                                model.losses[1::3])),
                                                   tf.math.reduce_mean(tf.where(tf.math.is_nan(model.losses[2::3]),
                                                                                tf.zeros_like(model.losses[2::3]),
                                                                                model.losses[2::3]))
                                               ])])

        loss_list.append(current_loss)

        if math.isnan(loss_list[-1]) or math.isinf(loss_list[-1]):
            model, optimiser, loss_list, previous_x_prediction_list, previous_model_weight_list, \
            previous_optimiser_weight_list, current_loss_increase_patience = \
                train_backup(model, optimiser, loss_list, previous_x_prediction_list,
                             previous_model_weight_list, previous_optimiser_weight_list,
                             current_loss_increase_patience)

            continue

        if relative_plateau_cutoff is None:
            relative_plateau_cutoff = parameters.plateau_cutoff * loss_list[-1]

        if len(loss_list) > 1:
            loss_gradient = np.gradient(loss_list)[-1]

            if not np.allclose(loss_gradient, np.zeros(loss_gradient.shape), atol=relative_plateau_cutoff):
                if not (loss_list[-1] * (parameters.backtracking_weight_percentage / 100.0) < loss_list[-2]):
                    if current_loss_increase_patience >= parameters.patience:
                        print("WARNING: Loss increased above threshold; patience reached, allowing anyway!")

                        break
                    else:
                        model, optimiser, loss_list, previous_x_prediction_list, previous_model_weight_list,\
                        previous_optimiser_weight_list, current_loss_increase_patience = \
                            train_backup(model, optimiser, loss_list, previous_x_prediction_list,
                                         previous_model_weight_list, previous_optimiser_weight_list,
                                         current_loss_increase_patience)
                else:
                    break
            else:
                break
        else:
            break

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_list[-1], model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimiser.apply_gradients(zip(grads, model.trainable_weights))

    return model, loss_list, uncertainty, previous_x_prediction_list, previous_model_weight_list,\
           previous_optimiser_weight_list, relative_plateau_cutoff


def get_bayesian_test_prediction(model, x_train_iteration):
    print("get_bayesian_test_prediction")

    x_prediction_list = []

    for i in range(parameters.bayesian_test_iterations):
        x_prediction_list.append(model(x_train_iteration, training=False))

    x_prediction = tf.math.reduce_mean(x_prediction_list, axis=0)

    x_prediction_uncertainty = tf.math.reduce_mean(tf.math.reduce_variance(x_prediction_list, axis=0))

    return x_prediction, x_prediction_uncertainty


def test_step(model, x_train_iteration, y_train_iteration):
    print("test_step")

    x_prediction, x_prediction_uncertainty = \
        get_bayesian_test_prediction(model, x_train_iteration)
    x_prediction = np.squeeze(x_prediction)

    current_accuracy = get_evaluation_score(x_prediction, y_train_iteration)

    return x_prediction, x_prediction_uncertainty, current_accuracy


def get_evaluation_score(prediction, gt):
    print("get_evaluation_score")

    if float_sixteen_bool:
        prediction = prediction.astype(np.float32)

    accuracy = np.ma.corrcoef(np.ma.masked_invalid(np.ravel(np.squeeze(gt))),
                              np.ma.masked_invalid(np.ravel(np.squeeze(prediction))))[0, 1]

    return accuracy


def output_window_predictions(x_prediction, y_train_iteration, preprocessing_steps, window_input_shape,
                              current_output_path, j):
    print("output_window_predictions")

    x_prediction = x_prediction.astype(np.float64)
    y_train_iteration = y_train_iteration.astype(np.float64)

    x_prediction = preprocessing.redistribute([x_prediction], "numpy", True, [y_train_iteration], "numpy")[0]
    x_prediction = preprocessing.data_preprocessing([x_prediction], "numpy", preprocessing_steps)[0][0]

    x_prediction = np.squeeze(preprocessing.data_upsample([x_prediction], "numpy", window_input_shape)[0])

    output_volume = nib.Nifti1Image(x_prediction, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}.nii".format(current_output_path, str(j))
    nib.save(output_volume, current_data_path)

    return current_data_path


def output_patient_time_point_predictions(window_data_paths, windowed_full_input_axial_size, full_input_shape,
                                          current_output_path, i):
    print("output_patient_time_point_predictions")

    if len(window_data_paths) > 1:
        number_of_windows = int(np.ceil(full_input_shape[-1] / parameters.data_window_size))
        number_of_overlaps = number_of_windows - 1

        overlap_size = int(np.ceil((((parameters.data_window_size * number_of_windows) -
                                     full_input_shape[-1]) / number_of_overlaps)))
        overlap_index = int(np.ceil(parameters.data_window_size - overlap_size))

        output_arrays = []

        for l in range(number_of_windows):
            current_overlap_index = overlap_index * l

            current_output_array = np.zeros((full_input_shape[0], full_input_shape[1], windowed_full_input_axial_size))

            current_output_array[:, :, current_overlap_index:current_overlap_index + parameters.data_window_size] = \
                nib.load(window_data_paths[l]).get_data()
            current_output_array[np.isclose(current_output_array, 0.0)] = np.nan

            output_arrays.append(current_output_array)

        output_array = np.nanmean(np.asarray(output_arrays), axis=0)
        output_array = np.nan_to_num(output_array, copy=False)
    else:
        output_array = nib.load(window_data_paths[0]).get_data()

    output_array = np.squeeze(preprocessing.data_upsample([output_array], "numpy", full_input_shape)[0])

    output_volume = nib.Nifti1Image(output_array, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}.nii".format(current_output_path, str(i))
    nib.save(output_volume, current_data_path)

    return current_data_path


def train_model():
    print("train_model")

    # get data and lables
    x, y, preprocessing_steps, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt = \
        get_preprocessed_train_data()

    data_shape = np.load(x[0])[0].shape

    model, optimiser, loss = architecture.get_model(data_shape)
    model.summary()

    print("Memory usage:\t{0}".format(str(get_model_memory_usage(1, model))))

    k.utils.plot_model(model, to_file="{0}/model.pdf".format(output_path), show_shapes=True, show_dtype=True,
                       show_layer_names=True, expand_nested=True)

    for i in range(len(y)):
        print("Patient/Time point:\t{0}".format(str(i)))

        patient_output_path = "{0}/{1}".format(output_path, str(i))

        # create output directory
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path, mode=0o770)

        number_of_windows = np.load(x[i]).shape[0]

        current_window_data_paths = []

        for j in range(number_of_windows):
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

            x_train_iteration = np.asarray([np.load(x[i])[j]])
            y_train_iteration = np.asarray([np.load(y[i])[j]])

            if float_sixteen_bool:
                x_train_iteration = x_train_iteration.astype(np.float16)
                y_train_iteration = y_train_iteration.astype(np.float16)
            else:
                x_train_iteration = x_train_iteration.astype(np.float32)
                y_train_iteration = y_train_iteration.astype(np.float32)

            if gt is not None:
                gt_prediction = np.asarray([np.load(gt[i])[j]])
            else:
                gt_prediction = None

            total_iterations = 0

            loss_list = []
            previous_x_prediction_list = []
            previous_model_weight_list = []
            previous_optimiser_weight_list = []

            relative_plateau_cutoff = None

            gt_accuracy = None
            max_accuracy = 0.0
            max_accuracy_iteration = 0

            while True:
                print("Iteration:\t{0:<20}\tTotal iterations:\t{1:<20}".format(str(len(loss_list)), str(total_iterations)))

                model, loss_list, uncertainty, previous_x_prediction_list, previous_model_weight_list,\
                previous_optimiser_weight_list, relative_plateau_cutoff = \
                    train_step(model, optimiser, loss, x_train_iteration, y_train_iteration, loss_list,
                               previous_x_prediction_list, previous_model_weight_list, previous_optimiser_weight_list,
                               relative_plateau_cutoff, model_output_path)

                if len(previous_x_prediction_list) > parameters.bayesian_test_iterations - 1:
                    previous_x_prediction_list.pop(0)

                x_prediction, x_prediction_uncertainty, current_accuracy = \
                    test_step(model, x_train_iteration, y_train_iteration)

                print("Loss:\t{0:<20}\tAccuracy:\t{1:<20}\tUncertainty:\t{2:<20}".format(str(loss_list[-1].numpy()), str(current_accuracy), str(uncertainty.numpy())))

                if gt is not None:
                    gt_accuracy = get_evaluation_score(x_prediction, gt_prediction)

                    if max_accuracy < gt_accuracy:
                        max_accuracy = gt_accuracy
                        max_accuracy_iteration = len(loss_list)

                    print("GT accuracy:\t{0:<20}\tGT uncertainty:\t{1:<20}".format(str(gt_accuracy), str(x_prediction_uncertainty.numpy())))
                else:
                    if max_accuracy < current_accuracy:
                        max_accuracy = current_accuracy
                        max_accuracy_iteration = len(loss_list)

                x_prediction = np.squeeze(x_prediction).astype(np.float64)
                y_plot = np.squeeze(y_train_iteration).astype(np.float64)

                plt.figure()

                if gt is not None:
                    plt.subplot(1, 3, 1)
                else:
                    plt.subplot(1, 2, 1)

                plt.imshow(x_prediction[:, :, int(x_prediction.shape[2] / 2)], cmap="Greys")

                if gt is not None:
                    plt.subplot(1, 3, 2)
                else:
                    plt.subplot(1, 2, 2)

                plt.imshow(y_plot[:, :, int(y_plot.shape[2] / 2)], cmap="Greys")

                if gt_prediction is not None:
                    gt_plot = np.squeeze(gt_prediction).astype(np.float64)

                    plt.subplot(1, 3, 3)
                    plt.imshow(gt_plot[:, :, int(gt_plot.shape[2] / 2)], cmap="Greys")

                plt.tight_layout()
                plt.savefig("{0}/{1}.png".format(plot_output_path, str(len(loss_list))), format="png", dpi=600,
                            bbox_inches="tight")
                plt.close()

                plot_files = os.listdir(plot_output_path)

                for l in range(len(plot_files)):
                    current_plot_file = plot_files[l].strip()

                    split_array = current_plot_file.split(".png")

                    if len(split_array) >= 2:
                        if int(split_array[0]) > len(loss_list):
                            os.remove("{0}/{1}".format(plot_output_path, current_plot_file))

                model_files = os.listdir(model_output_path)

                for l in range(len(model_files)):
                    current_plot_file = model_files[l].strip()

                    split_array = current_plot_file.split("model_")[-1]
                    split_array = split_array.split("optimiser_")[-1]

                    split_array = split_array.split(".pkl")

                    if len(split_array) >= 2:
                        if int(split_array[0]) > len(loss_list):
                            os.remove("{0}/{1}".format(model_output_path, current_plot_file))

                if len(loss_list) >= parameters.patience:
                    loss_gradient = np.gradient(loss_list)[-parameters.patience:]

                    if np.allclose(loss_gradient, np.zeros(loss_gradient.shape), atol=relative_plateau_cutoff):
                        print("Reached plateau: Exiting...")
                        print("Maximum accuracy:\t{0:<20}\tMaximum accuracy iteration:\t{1:<20}".format(str(max_accuracy), str(max_accuracy_iteration)))

                        if gt is not None:
                            accuracy_loss = np.abs(gt_accuracy - max_accuracy)
                        else:
                            accuracy_loss = np.abs(current_accuracy - max_accuracy)

                        print("Accuracy loss:\t{0:<20}".format(str(accuracy_loss)))

                        break

                total_iterations = total_iterations + 1

            current_window_data_paths.append(output_window_predictions(x_prediction, y_train_iteration,
                                                                       preprocessing_steps, window_input_shape,
                                                                       window_output_path, j))

        output_patient_time_point_predictions(current_window_data_paths, windowed_full_input_axial_size,
                                              full_input_shape, patient_output_path, i)


def main():
    print("main")

    # if debugging, remove previous output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)

    # create log file and begin writing to it
    logfile_path = "{0}/logfile.log".format(output_path)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)

    import transcript
    transcript.start(logfile_path)

    train_model()

    import python_email_notification
    python_email_notification.main()

    transcript.stop()

    return True


if __name__ == "__main__":
    main()
