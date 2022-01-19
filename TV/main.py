# Copyright University College London 2021
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import re
import shutil
import random
import numpy as np
import skimage.restoration
import matplotlib.pyplot as plt
import nibabel as nib
import gzip


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
    # tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    # ONLY WORKS WITH TF V1
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)
else:
    random.seed()


import transcript
import parameters
import preprocessing


data_path = "{0}/DIP_RDP_data/static_mean_thorax_simulation_noisy/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/TV/".format(os.getcwd())


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

    current_volume = None
    full_current_shape = None
    current_shape = None
    windowed_full_input_axial_size = None

    for i in range(len(y_files)):
        current_volume = nib.load(y_files[i])
        current_array = current_volume.get_data()

        if full_current_shape is None:
            full_current_shape = current_array.shape

        current_array, windowed_full_input_axial_size = get_data_windows(current_array)

        if current_shape is None:
            current_shape = current_array[0].shape

        current_y_train_path = "{0}/{1}.npy".format(y_train_output_path, str(i))

        with gzip.GzipFile(current_y_train_path, "w") as file:
            np.save(file, current_array)

        y.append(current_y_train_path)

        current_x_train_path = "{0}/{1}.npy".format(x_train_output_path, str(i))

        with gzip.GzipFile(current_x_train_path, "w") as file:
            np.save(file, current_array)

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

            with gzip.GzipFile(current_gt_train_path, "w") as file:
                np.save(file, current_array)

            gt.append(current_gt_train_path)

        gt = np.asarray(gt)
    else:
        gt = None

    return x, y, current_volume, full_current_shape, windowed_full_input_axial_size, current_shape, gt


def get_preprocessed_train_data():
    print("get_preprocessed_train_data")

    x, y, input_volume, full_input_shape, windowed_full_input_axial_size, window_input_shape, gt = get_train_data()

    x = preprocessing.data_upsample(x, "path")
    y = preprocessing.data_upsample(y, "path")
    gt = preprocessing.data_upsample(gt, "path")

    x, _ = preprocessing.data_preprocessing(x, "path")
    y, input_preprocessing = preprocessing.data_preprocessing(y, "path")
    gt, _ = preprocessing.data_preprocessing(gt, "path")

    return x, y, input_volume, full_input_shape, windowed_full_input_axial_size, window_input_shape, input_preprocessing, gt


def total_variation_denoising(data):
    print("total_variation_denoising")

    data_copy = data.copy()

    data_mean = np.mean(data_copy)
    data_std = np.std(data_copy)

    data_copy = (data_copy - data_mean) / data_std

    data_copy = skimage.restoration.denoise_tv_chambolle(data_copy,
                                                         weight=parameters.total_variation_weight,
                                                         eps=parameters.total_variation_eps,
                                                         n_iter_max=parameters.total_variation_max_iteration,
                                                         multichannel=False)

    data_copy = (data_copy * data_std) + data_mean

    return data_copy


def get_evaluation_score(x_prediction, gt):
    print("get_evaluation_score")

    accuracy = np.corrcoef(np.ravel(np.squeeze(gt)), np.ravel(np.squeeze(x_prediction)))[0, 1]

    return accuracy


def output_window_predictions(x_prediction, input_volume, window_input_shape, input_preprocessing, current_output_path,
                              i, j):
    print("output_window_predictions")

    x_prediction = preprocessing.data_preprocessing([x_prediction], "numpy", [input_preprocessing[i]])[0]
    x_prediction = np.squeeze(preprocessing.data_upsample([x_prediction], "numpy", window_input_shape)[0])

    output_volume = nib.Nifti1Image(x_prediction, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}.nii.gz".format(current_output_path, str(j))
    nib.save(output_volume, current_data_path)

    return current_data_path


def output_patient_time_point_predictions(window_data_paths, windowed_full_input_axial_size, full_input_shape,
                                          current_output_path, i):
    print("output_patient_time_point_predictions")

    if len(window_data_paths) > 1:
        number_of_windows = int(np.ceil(full_input_shape[-1] / parameters.data_window_size))
        number_of_overlaps = number_of_windows - 1

        overlap_size = \
            int(np.ceil((((parameters.data_window_size * number_of_windows) -
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
        output_array = np.nan_to_num(output_array)
    else:
        output_array = nib.load(window_data_paths[0]).get_data()

    output_array = np.squeeze(preprocessing.data_upsample([output_array], "numpy", full_input_shape)[0])

    output_volume = nib.Nifti1Image(output_array, np.eye(4), nib.Nifti1Header())

    current_data_path = "{0}/{1}.nii.gz".format(current_output_path, str(i))
    nib.save(output_volume, current_data_path)

    return current_data_path


def train_model():
    print("train_model")

    # get data and lables
    x, y, input_volume, full_input_shape, windowed_full_input_axial_size, window_input_shape, input_preprocessing, \
    gt = \
        get_preprocessed_train_data()

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

            if gt is not None:
                with gzip.GzipFile(gt[i], "r") as file:
                    gt_prediction = np.asarray([np.load(file)[j]])
            else:
                gt_prediction = None

            x_prediction = total_variation_denoising(x_train_iteration)

            if gt is not None:
                gt_accuracy = get_evaluation_score(x_prediction, gt_prediction)

                print("GT accuracy:\t{0}".format(str(gt_accuracy)))

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
            plt.savefig("{0}/0.png".format(plot_output_path),
                        format="png", dpi=600, bbox_inches="tight")
            plt.close()

            current_window_data_paths.append(output_window_predictions(x_prediction, input_volume, window_input_shape,
                                                                       input_preprocessing, window_output_path, i, j))

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

    transcript.start(logfile_path)

    train_model()

    transcript.stop()

    return True


if __name__ == "__main__":
    main()