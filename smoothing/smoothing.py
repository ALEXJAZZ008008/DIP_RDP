# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import re
import shutil
import random
import numpy as np
import scipy.signal
import scipy.ndimage
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
else:
    random.seed()


import parameters
import preprocessing
import losses


data_path = None
output_path = None


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


def get_train_data():
    print("get_train_data")

    y_path = "{0}/y/".format(data_path)

    y_files = os.listdir(y_path)
    y_files.sort(key=human_sorting)
    y_files = ["{0}{1}".format(y_path, s) for s in y_files]

    example_data = nib.load(y_files[0])

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

    for i in range(len(y_files)):
        current_volume = nib.load(y_files[i])
        current_array = current_volume.get_fdata()

        if parameters.data_window_bool:
            if full_current_shape is None:
                full_current_shape = current_array.shape

        current_array, windowed_full_input_axial_size = get_data_windows(current_array)

        if full_current_shape is None:
            full_current_shape = current_array.shape[1:]

        if current_shape is None:
            current_shape = current_array[0].shape

        current_y_train_path = "{0}/{1}.npy.gz".format(y_train_output_path, str(i))

        with gzip.GzipFile(current_y_train_path, "w") as file:
            np.save(file, current_array)

        y.append(current_y_train_path)

        current_x_train_path = "{0}/{1}.npy.gz".format(x_train_output_path, str(i))

        with gzip.GzipFile(current_x_train_path, "w") as file:
            np.save(file, current_array)

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

            current_array, _ = get_data_windows(current_array)

            current_gt_train_path = "{0}/{1}.npy.gz".format(gt_train_output_path, str(i))

            with gzip.GzipFile(current_gt_train_path, "w") as file:
                np.save(file, current_array)

            gt.append(current_gt_train_path)

        gt = np.asarray(gt)
    else:
        gt = None

    return x, y, example_data, full_current_shape, windowed_full_input_axial_size, current_shape, gt


def get_preprocessed_train_data():
    print("get_preprocessed_train_data")

    x, y, example_data, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt = \
        get_train_data()

    x = preprocessing.data_upsample_pad(x, "path")
    y = preprocessing.data_upsample_pad(y, "path")

    if gt is not None:
        gt = preprocessing.data_upsample_pad(gt, "path")

    x, _ = preprocessing.data_preprocessing(x, "path")
    y, preprocessing_steps = preprocessing.data_preprocessing(y, "path")

    if gt is not None:
        gt, _ = preprocessing.data_preprocessing(gt, "path")

    return x, y, example_data, preprocessing_steps, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt


# https://stackoverflow.com/questions/10623448/making-gaussians-constrained-by-the-fwhm
def gamma2sigma(example_data, gamma=6.4):
    print("gamma2sigma")

    data_voxel_sizes = example_data.header.get_zooms()

    sigma = ((gamma / data_voxel_sizes[1]) * np.sqrt(2.0)) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)

    return sigma


def resolution_smoothing(data_array, example_data):
    print("resolution_smoothing")

    kernel_size = np.max(data_array.shape)
    sigma = gamma2sigma(example_data, 6.4)

    kernel = scipy.signal.windows.gaussian(kernel_size, sigma)
    kernel = kernel / np.sum(kernel)

    kernel = np.expand_dims(kernel, axis=0)
    kernel = np.expand_dims(kernel, axis=2)

    kernel = np.expand_dims(kernel, axis=0)
    kernel = np.expand_dims(kernel, axis=-1)

    data_array = scipy.ndimage.convolve(data_array, kernel, mode="nearest")

    kernel = np.moveaxis(kernel, (0, 1, 2, 3, 4), (4, 0, 1, 3, 2))

    data_array = scipy.ndimage.convolve(data_array, kernel, mode="nearest")

    kernel = np.array([1.0, 4.0, 1.0])
    kernel = kernel / np.sum(kernel)

    kernel = np.expand_dims(kernel, axis=0)
    kernel = np.expand_dims(kernel, axis=0)

    kernel = np.expand_dims(kernel, axis=0)
    kernel = np.expand_dims(kernel, axis=-1)

    data_array = scipy.ndimage.convolve(data_array, kernel, mode="nearest")

    return data_array


def test_step(x_train_iteration, example_data, y_train_iteration):
    print("test_step")

    x_train_iteration = resolution_smoothing(x_train_iteration, example_data)

    current_accuracy = [losses.correlation_coefficient_accuracy(y_train_iteration, x_train_iteration)]

    return x_train_iteration, current_accuracy


def output_window_predictions(x_prediction, y_train_iteration, gt_prediction, preprocessing_steps, window_input_shape,
                              current_output_path, j):
    print("output_window_predictions")

    x_prediction, _ = preprocessing.data_preprocessing([x_prediction], "numpy", preprocessing_steps)
    x_prediction = np.squeeze(preprocessing.data_downsampling_crop([x_prediction], "numpy", window_input_shape)[0])

    if gt_prediction is not None:
        gt_prediction, _ = preprocessing.data_preprocessing([gt_prediction], "numpy", preprocessing_steps)
        gt_prediction = np.squeeze(preprocessing.data_downsampling_crop([gt_prediction], "numpy",
                                                                        window_input_shape)[0])

        current_output_accuracy = \
            [losses.correlation_coefficient_accuracy(gt_prediction, x_prediction).numpy()]
    else:
        y_train_iteration, _ = preprocessing.data_preprocessing([y_train_iteration], "numpy", preprocessing_steps)
        y_train_iteration = np.squeeze(preprocessing.data_downsampling_crop([y_train_iteration], "numpy",
                                                                            window_input_shape)[0])

        current_output_accuracy = \
            [losses.correlation_coefficient_accuracy(y_train_iteration, x_prediction).numpy()]

    print("Output accuracy:\t{0:<20}".format(str(current_output_accuracy[0])))

    output_volume = nib.Nifti1Image(x_prediction, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}_output.nii.gz".format(current_output_path, str(j))
    nib.save(output_volume, current_data_path)

    return current_data_path


def output_patient_time_point_predictions(window_data_paths, example_data, windowed_full_input_axial_size,
                                          full_input_shape, current_output_path, current_output_prefix, i):
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

            current_data = nib.load(window_data_paths[l]).get_data()

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

    output_array = np.squeeze(preprocessing.data_downsampling_crop([output_array], "numpy", full_input_shape)[0])

    example_data_header = example_data.header

    output_array = np.squeeze(preprocessing.data_downsampling([output_array], "numpy",
                                                              example_data_header.get_data_shape())[0])

    output_volume = nib.Nifti1Image(output_array, example_data.affine, example_data_header)

    current_data_path = "{0}/{1}_{2}.nii.gz".format(current_output_path, str(i), current_output_prefix)
    nib.save(output_volume, current_data_path)

    return current_data_path


def train_model():
    print("train_model")

    # get data and lables
    x, y, example_data, preprocessing_steps, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt = \
        get_preprocessed_train_data()

    output_paths = []

    for i in range(len(y)):
        print("Patient/Time point:\t{0}".format(str(i)))

        patient_output_path = "{0}/{1}".format(output_path, str(i))

        # create output directory
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path, mode=0o770)

        with gzip.GzipFile(x[i], "r") as file:
            number_of_windows = np.load(file).shape[0]

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

            with gzip.GzipFile(x[i], "r") as file:
                x_train_iteration = np.asarray([np.load(file)[j]])

            with gzip.GzipFile(y[i], "r") as file:
                y_train_iteration = np.asarray([np.load(file)[j]])

            x_train_iteration = x_train_iteration.astype(np.float32)
            y_train_iteration = y_train_iteration.astype(np.float32)

            if gt is not None:
                with gzip.GzipFile(gt[i], "r") as file:
                    gt_prediction = np.asarray([np.load(file)[j]])

                gt_prediction = gt_prediction.astype(np.float32)
            else:
                gt_prediction = None

            x_prediction, current_accuracy = test_step(x_train_iteration, example_data, y_train_iteration)

            print("Accuracy:\t{0:<20}".format(str(current_accuracy[0].numpy())))

            if gt is not None:
                gt_accuracy = [losses.correlation_coefficient_accuracy(gt_prediction, x_prediction)]

                print("GT accuracy:\t{0:<20}".format(str(str(gt_accuracy[0].numpy()))))

            x_output = np.squeeze(x_prediction).astype(np.float64)
            y_output = np.squeeze(y_train_iteration).astype(np.float64)

            plt.figure()

            if gt is not None:
                plt.subplot(1, 3, 1)
            else:
                plt.subplot(1, 2, 1)

            plt.imshow(x_output[:, :, int(x_output.shape[2] / 2)], cmap="Greys")

            if gt is not None:
                plt.subplot(1, 3, 2)
            else:
                plt.subplot(1, 2, 2)

            plt.imshow(y_output[:, :, int(y_output.shape[2] / 2)], cmap="Greys")

            if gt is not None:
                gt_plot = np.squeeze(gt_prediction).astype(np.float64)

                plt.subplot(1, 3, 3)
                plt.imshow(gt_plot[:, :, int(gt_plot.shape[2] / 2)], cmap="Greys")

            plt.tight_layout()
            plt.savefig("{0}/{1}.png".format(plot_output_path, str(0)), format="png", dpi=600,
                        bbox_inches="tight")
            plt.close()

            current_window_data_path = \
                output_window_predictions(x_output, y_output, gt_prediction, preprocessing_steps, window_input_shape,
                                          window_output_path, j)

            current_window_data_paths.append(current_window_data_path)

        try:
            output_paths.append(output_patient_time_point_predictions(current_window_data_paths, example_data,
                                                                      windowed_full_input_axial_size,
                                                                      full_input_shape, patient_output_path, "output",
                                                                      i))
        except:
            print("Error: Input not suitable; continuing")

    return output_paths


def main(input_data_path=None, input_output_path=None):
    print("main")

    global data_path
    global output_path

    if input_data_path is not None:
        data_path = input_data_path
    else:
        data_path = parameters.data_path

    if input_output_path is not None:
        output_path = input_output_path
    else:
        output_path = parameters.output_path

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

    output_paths = train_model()

    # transcript.transcript_stop(logfile)

    return output_paths


if __name__ == "__main__":
    main()
