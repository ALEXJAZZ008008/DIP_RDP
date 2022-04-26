# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import numpy as np
import scipy.ndimage
from sklearn.preprocessing import StandardScaler
import gzip

import main
import parameters


if main.reproducible_bool:
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(main.seed_value)


def get_next_geometric_value(an, a0):
    print("get_next_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.ceil(n) - 1))

    return an


def get_previous_geometric_value(an, a0):
    print("get_previous_geometric_value")

    n = math.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2, (math.floor(n) - 1))

    return an


def data_upsample(data, data_type, new_resolution=None):
    print("data_upsample")

    geometric_sequence_a0 = 2

    for i in range(len(data)):
        if data_type == "path":
            with gzip.GzipFile(data[i], "r") as file:
                data_copy = np.load(file)  # noqa
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            dimension_x_upscale_factor = new_resolution[0] / data_copy.shape[1]
            dimension_y_upscale_factor = new_resolution[1] / data_copy.shape[2]
            dimension_z_upscale_factor = new_resolution[2] / data_copy.shape[3]
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            dimension_x_upscale_factor = \
                get_next_geometric_value(data_copy_shape[1], geometric_sequence_a0) / data_copy.shape[1]
            dimension_y_upscale_factor = \
                get_next_geometric_value(data_copy_shape[2], geometric_sequence_a0) / data_copy.shape[2]
            dimension_z_upscale_factor = \
                get_next_geometric_value(data_copy_shape[3], geometric_sequence_a0) / data_copy.shape[3]

        if not np.isclose(dimension_x_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_y_upscale_factor, 1.0, rtol=0.0, atol=1e-05) or \
                not np.isclose(dimension_z_upscale_factor, 1.0, rtol=0.0, atol=1e-05):
            data_copy = scipy.ndimage.zoom(data_copy, (1, dimension_x_upscale_factor, dimension_y_upscale_factor,
                                                       dimension_z_upscale_factor), order=3, mode="mirror",
                                           prefilter=True)

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            with gzip.GzipFile(data[i], "w") as file:
                np.save(file, data_copy)  # noqa
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


def data_upsample_pad(data, data_type, new_resolution=None, pad_mode="edge"):
    print("data_upsample_pad")

    for i in range(len(data)):
        if data_type == "path":
            with gzip.GzipFile(data[i], "r") as file:
                data_copy = np.load(file)  # noqa
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            if data_copy.shape[1] % 2.0 != new_resolution[0] % 2.0:
                dimension_x_pad_factor = 1
            else:
                dimension_x_pad_factor = 0

            if data_copy.shape[2] % 2.0 != new_resolution[1] % 2.0:
                dimension_y_pad_factor = 1
            else:
                dimension_y_pad_factor = 0

            if data_copy.shape[3] % 2.0 != new_resolution[2] % 2.0:
                dimension_z_pad_factor = 1
            else:
                dimension_z_pad_factor = 0
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            if (data_copy.shape[1] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[1], 2) % 2.0):
                dimension_x_pad_factor = 1
            else:
                dimension_x_pad_factor = 0

            if (data_copy.shape[2] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[2], 2) % 2.0):
                dimension_y_pad_factor = 1
            else:
                dimension_y_pad_factor = 0

            if (data_copy.shape[3] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[3], 2) % 2.0):
                dimension_z_pad_factor = 1
            else:
                dimension_z_pad_factor = 0

        if dimension_x_pad_factor != 0 or dimension_y_pad_factor != 0 or dimension_z_pad_factor != 0:
            if pad_mode == "constant":
                data_copy = np.pad(data_copy, ((0, 0), (dimension_x_pad_factor, 0), (dimension_y_pad_factor, 0),
                                               (dimension_z_pad_factor, 0)), mode=pad_mode)

        if new_resolution is None:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            new_resolution = [get_next_geometric_value(data_copy_shape[1], 2),
                              get_next_geometric_value(data_copy_shape[2], 2),
                              get_next_geometric_value(data_copy_shape[3], 2)]

        dimension_x_pad_factor = int(np.abs((new_resolution[0] - data_copy.shape[1]) / 2.0))
        dimension_y_pad_factor = int(np.abs((new_resolution[1] - data_copy.shape[2]) / 2.0))
        dimension_z_pad_factor = int(np.abs((new_resolution[2] - data_copy.shape[3]) / 2.0))

        if dimension_x_pad_factor != 0 or dimension_y_pad_factor != 0 or dimension_z_pad_factor != 0:
            if pad_mode == "constant":
                data_copy = np.pad(data_copy, ((0, 0), (dimension_x_pad_factor, dimension_x_pad_factor),
                                               (dimension_y_pad_factor, dimension_y_pad_factor),
                                               (dimension_z_pad_factor, dimension_z_pad_factor)), mode=pad_mode)

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            with gzip.GzipFile(data[i], "w") as file:
                np.save(file, data_copy)  # noqa
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


def data_downsampling_crop(data, data_type, new_resolution=None):
    print("data_downsampling_crop")

    for i in range(len(data)):
        if data_type == "path":
            with gzip.GzipFile(data[i], "r") as file:
                data_copy = np.load(file)  # noqa
        else:
            if data_type == "numpy":
                data_copy = data[i].copy()
            else:
                data_copy = None

        data_copy = np.squeeze(data_copy)

        if data_copy.ndim < 4:
            data_copy = np.expand_dims(data_copy, 0)

        if new_resolution is not None:
            if data_copy.shape[1] % 2.0 != new_resolution[0] % 2.0:
                dimension_x_crop_factor = 1
            else:
                dimension_x_crop_factor = 0

            if data_copy.shape[2] % 2.0 != new_resolution[1] % 2.0:
                dimension_y_crop_factor = 1
            else:
                dimension_y_crop_factor = 0

            if data_copy.shape[3] % 2.0 != new_resolution[2] % 2.0:
                dimension_z_crop_factor = 1
            else:
                dimension_z_crop_factor = 0
        else:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            if (data_copy.shape[1] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[1], 2) % 2.0):
                dimension_x_crop_factor = 1
            else:
                dimension_x_crop_factor = 0

            if (data_copy.shape[2] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[2], 2) % 2.0):
                dimension_y_crop_factor = 1
            else:
                dimension_y_crop_factor = 0

            if (data_copy.shape[3] % 2.0 !=
                    get_next_geometric_value(data_copy_shape[3], 2) % 2.0):
                dimension_z_crop_factor = 1
            else:
                dimension_z_crop_factor = 0

        if dimension_x_crop_factor != 0 or dimension_y_crop_factor != 0 or dimension_z_crop_factor != 0:
            data_copy = data_copy[:, dimension_x_crop_factor or None:, dimension_y_crop_factor or None:,
                        dimension_z_crop_factor or None:]

        if new_resolution is None:
            data_copy_shape = list(data_copy.shape)

            for j in range(1, len(data_copy_shape)):
                if data_copy_shape[j] < parameters.data_window_size:
                    data_copy_shape[j] = parameters.data_window_size

            new_resolution = [get_previous_geometric_value(data_copy_shape[1], 2),
                              get_previous_geometric_value(data_copy_shape[2], 2),
                              get_previous_geometric_value(data_copy_shape[3], 2)]

        dimension_x_crop_factor = int(np.abs(np.floor((data_copy.shape[1] - new_resolution[0]) / 2.0)))
        dimension_y_crop_factor = int(np.abs(np.floor((data_copy.shape[2] - new_resolution[1]) / 2.0)))
        dimension_z_crop_factor = int(np.abs(np.floor((data_copy.shape[3] - new_resolution[2]) / 2.0)))

        if dimension_x_crop_factor != 0 or dimension_y_crop_factor != 0 or dimension_z_crop_factor != 0:
            data_copy = data_copy[:, dimension_x_crop_factor or None:-dimension_x_crop_factor or None,
                        dimension_y_crop_factor or None:-dimension_y_crop_factor or None,
                        dimension_z_crop_factor or None:-dimension_z_crop_factor or None]

        data_copy = np.expand_dims(data_copy, -1)

        if data_type == "path":
            with gzip.GzipFile(data[i], "w") as file:
                np.save(file, data_copy)  # noqa
        else:
            if data_type == "numpy":
                data[i] = data_copy

    return data


def data_preprocessing(data, data_type, preprocessing_steps=None):
    print("data_preprocessing")

    if preprocessing_steps is None:
        preprocessing_steps = StandardScaler()

        for i in range(len(data)):
            if data_type == "path":
                with gzip.GzipFile(data[i], "r") as file:
                    data_copy = np.load(file)  # noqa
            else:
                if data_type == "numpy":
                    data_copy = data[i].copy()
                else:
                    data_copy = None

            # data_copy = mask_fov(data_copy)

            data_copy = data_copy.reshape(-1, 1)

            preprocessing_steps.partial_fit(data_copy)

        for i in range(len(data)):
            if data_type == "path":
                with gzip.GzipFile(data[i], "r") as file:
                    data_copy = np.load(file)  # noqa
            else:
                if data_type == "numpy":
                    data_copy = data[i].copy()
                else:
                    data_copy = None

            # data_copy = mask_fov(data_copy)

            data_copy_shape = data_copy.shape
            data_copy = data_copy.reshape(-1, 1)

            data_copy = preprocessing_steps.transform(data_copy)

            data_copy = data_copy.reshape(data_copy_shape)

            if data_type == "path":
                with gzip.GzipFile(data[i], "w") as file:
                    np.save(file, data_copy)  # noqa
            else:
                if data_type == "numpy":
                    data[i] = data_copy
    else:
        for i in range(len(data)):
            if data_type == "path":
                with gzip.GzipFile(data[i], "r") as file:
                    data_copy = np.load(file)  # noqa
            else:
                if data_type == "numpy":
                    data_copy = data[i].copy()
                else:
                    data_copy = None

            data_copy_shape = data_copy.shape
            data_copy = data_copy.reshape(-1, 1)

            data_copy = preprocessing_steps.inverse_transform(data_copy)

            data_copy = data_copy.reshape(data_copy_shape)

            if data_type == "path":
                with gzip.GzipFile(data[i], "w") as file:
                    np.save(file, data_copy)  # noqa
            else:
                if data_type == "numpy":
                    data[i] = data_copy

    return data, preprocessing_steps
