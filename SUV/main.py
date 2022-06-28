# Copyright University College London 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import shutil
import errno
import numpy as np
import scipy.ndimage
import nibabel as nib
from sklearn.preprocessing import StandardScaler


def mkdir_p(path):
    print("mkdir_p")

    try:
        os.makedirs(path, mode=0o770)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

    return True


def data_downsampling_crop(data, new_resolution):
    print("data_downsampling_crop")

    data_copy = data.copy()

    if data_copy.shape[0] % 2.0 != new_resolution[0] % 2.0:
        dimension_x_crop_factor = 1
    else:
        dimension_x_crop_factor = 0

    if data_copy.shape[1] % 2.0 != new_resolution[1] % 2.0:
        dimension_y_crop_factor = 1
    else:
        dimension_y_crop_factor = 0

    if data_copy.shape[2] % 2.0 != new_resolution[2] % 2.0:
        dimension_z_crop_factor = 1
    else:
        dimension_z_crop_factor = 0

    if dimension_x_crop_factor != 0 or dimension_y_crop_factor != 0 or dimension_z_crop_factor != 0:
        data_copy = data_copy[dimension_x_crop_factor or None:,
                              dimension_y_crop_factor or None:,
                              dimension_z_crop_factor or None:]

    dimension_x_crop_factor = int(np.abs(np.floor((data_copy.shape[0] - new_resolution[0]) / 2.0)))
    dimension_y_crop_factor = int(np.abs(np.floor((data_copy.shape[1] - new_resolution[1]) / 2.0)))
    dimension_z_crop_factor = int(np.abs(np.floor((data_copy.shape[2] - new_resolution[2]) / 2.0)))

    if dimension_x_crop_factor != 0 or dimension_y_crop_factor != 0 or dimension_z_crop_factor != 0:
        data_copy = data_copy[dimension_x_crop_factor or None:-dimension_x_crop_factor or None,
                              dimension_y_crop_factor or None:-dimension_y_crop_factor or None,
                              dimension_z_crop_factor or None:-dimension_z_crop_factor or None]

        data = data_copy

    return data


def preprocessing(data_array_list, crop_amount, average_list, offset_list, bias_list, rescale_bool_list,
                  rescale_to_index):
    print("preprocessing")

    data_array_list_len = len(data_array_list)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = data_downsampling_crop(current_data_array,
                                                    [current_data_array.shape[0] - crop_amount[0],
                                                     current_data_array.shape[1] - crop_amount[1],
                                                     current_data_array.shape[2] - crop_amount[2]])

        data_array_list[i] = current_data_array

    # for i in range(len(data_array_list)):
    #     current_data_array = data_array_list[i]

    #     current_data_array = current_data_array[:, :, offset_list[i][0]:offset_list[i][1]] + bias_list[i]
    #     current_data_array = current_data_array[:, :, 5:41]

    #     if average_list[i] is not None:
    #         average_list_len = len(average_list[i])

    #         for j in range(average_list_len):
    #             current_data_array = current_data_array + data_array_list[average_list[i][j]]

    #         current_data_array = current_data_array / (average_list_len + 1.0)

    #     data_array_list[i] = current_data_array

    rescale_scale = np.sum(data_array_list[rescale_to_index])

    for i in range(data_array_list_len):
        if rescale_bool_list[i]:
            current_data_array = data_array_list[i]

            current_data_array = (current_data_array / np.sum(current_data_array)) * rescale_scale

            data_array_list[i] = current_data_array

    standard_scaler = StandardScaler()

    for i in range(data_array_list_len):
        if not rescale_bool_list[i]:
            current_data_array = data_array_list[i]
            current_data_array = current_data_array.reshape(-1, 1)

            standard_scaler.partial_fit(current_data_array)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]
        current_data_array_shape = current_data_array.shape

        current_data_array = current_data_array.reshape(-1, 1)

        current_data_array = standard_scaler.transform(current_data_array)

        current_data_array = current_data_array.reshape(current_data_array_shape)

        data_array_list[i] = current_data_array

    data_array_min = np.min(data_array_list[0])

    for i in range(1, data_array_list_len):
        current_min = np.min(data_array_list[i])

        if current_min < data_array_min:
            data_array_min = current_min

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = current_data_array - data_array_min

        data_array_list[i] = current_data_array

    return data_array_list


def mask(data_array_list, mask_data_array=None, roi_position=[[0, -1], [0, -1], [0, -1]]):
    print("mask")

    if mask_data_array is not None:
        mask_data_array = np.bitwise_and(np.isfinite(mask_data_array), mask_data_array != 0)

    for i in range(len(data_array_list)):
        current_data_array_list = data_array_list[i]

        if mask_data_array is not None:
            current_data_array_list = current_data_array_list * mask_data_array

            nzero = np.nonzero(current_data_array_list)

            top = np.min(nzero[0])
            bottom = np.max(nzero[0])

            left = np.min(nzero[1])
            right = np.max(nzero[1])

            nearest = np.min(nzero[2])
            farthest = np.max(nzero[2])

            current_data_array_list = current_data_array_list[top:bottom + 1, left:right + 1, nearest:farthest + 1]
        else:
            current_data_array_list = current_data_array_list[roi_position[0][0]:roi_position[0][1],
                                                              roi_position[1][0]:roi_position[1][1],
                                                              roi_position[2][0]:roi_position[2][1]]

        data_array_list[i] = current_data_array_list

    return data_array_list


def suv_max(data_array_list):
    print("suv_max")

    data_array_list_max = []

    for i in range(len(data_array_list)):
        data_array_list_max.append(np.max(data_array_list[i]))

    return data_array_list_max


def suv_mean(data_array_list):
    print("suv_mean")

    data_array_list_mean = []

    for i in range(len(data_array_list)):
        data_array_list_mean.append(np.mean(data_array_list[i]))

    return data_array_list_mean


def suv_median(data_array_list):
    print("suv_median")

    data_array_list_median = []

    for i in range(len(data_array_list)):
        data_array_list_median.append(np.median(data_array_list[i]))

    return data_array_list_median


def get_peak_kernel(data_array_voxel_sizes_mm):
    print("get_peak_kernel")

    minimum_voxel_size_mm = np.min(data_array_voxel_sizes_mm)
    sphere_diameter_mm = 12.0

    size = int(np.ceil(sphere_diameter_mm / minimum_voxel_size_mm))
    radius = int(np.ceil((sphere_diameter_mm / 2.0) / minimum_voxel_size_mm))

    if not size % 2:
        size = size + 1

    peak_kernel = np.zeros((size, size, size))

    ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
    x0 = int(np.floor(peak_kernel.shape[0] / 2.0))
    y0 = int(np.floor(peak_kernel.shape[1] / 2.0))
    z0 = int(np.floor(peak_kernel.shape[2] / 2.0))

    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                if deb >= 0:
                    peak_kernel[x, y, z] = 1

    peak_kernel = scipy.ndimage.zoom(peak_kernel, (minimum_voxel_size_mm / data_array_voxel_sizes_mm[0],
                                                   minimum_voxel_size_mm / data_array_voxel_sizes_mm[1],
                                                   minimum_voxel_size_mm / data_array_voxel_sizes_mm[2]),
                                     order=1, mode="nearest", prefilter=True)

    peak_kernel = peak_kernel / np.sum(peak_kernel)

    return peak_kernel


def suv_peak(data_array_list, data_array_voxel_sizes_mm):
    print("suv_peak")

    peak_kernel = get_peak_kernel(data_array_voxel_sizes_mm)

    data_array_list_peak = []

    for i in range(len(data_array_list)):
        current_data_array_list = data_array_list[i]
        current_data_array_list = scipy.ndimage.convolve(current_data_array_list, peak_kernel, mode="nearest")

        data_array_list_peak.append(np.max(current_data_array_list))

    return data_array_list_peak


def main():
    print("main")

    output_path = "{0}/test/".format(os.getcwd())

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    mask_path = None

    data_paths = ["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_16.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_16.nii.gz",
                  "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/15/15.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"]

    crop_amount = [256, 6, 0]

    roi_position = [[79, 90], [119, 132], [14, 23]]

    rescale_bool_list = [True, True, True, True, False, True]
    rescale_to_index = 4

    average_list = [None, None, None, None, None, None]

    offset_list = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
    bias_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    data_legend = ["Ground Truth",
                   "Noisy",
                   "TV",
                   "Original DIP",
                   "New DIP Sequential",
                   "New DIP Combined"]

    mask_data_array = None

    if mask_path is not None:
        mask_data_array = nib.load(mask_path).get_fdata()

        mask_data_array = preprocessing([mask_data_array], crop_amount, [None], [[0, -1]], [0.0], [False], 0)[0]

    data_array_list = []

    data_paths_len = len(data_paths)

    for i in range(data_paths_len):
        data_array_list.append(nib.load(data_paths[i]).get_fdata())

    data_array_voxel_sizes_mm = nib.load(data_paths[0]).header.get_zooms()

    data_array_list = preprocessing(data_array_list, crop_amount, average_list, offset_list, bias_list,
                                    rescale_bool_list, rescale_to_index)
    data_array_list = mask(data_array_list, mask_data_array, roi_position)

    data_array_list_max = suv_max(data_array_list)
    data_array_list_mean = suv_mean(data_array_list)
    data_array_list_median = suv_median(data_array_list)
    data_array_list_peak = suv_peak(data_array_list, data_array_voxel_sizes_mm)

    outputs = []

    for i in range(data_paths_len):
        outputs.append("{0:<50}:\tSUVmax:\t{1:<20}\tSUVmean:\t{2:<20}\tSUVmedian:\t{3:<20}\tSUVpeak:\t{4:<20}".format(data_legend[i], str(data_array_list_max[i]), str(data_array_list_mean[i]), str(data_array_list_median[i]), str(data_array_list_peak[i])))

    for i in range(data_paths_len):
        print(outputs[i])

    with open("{0}/output.txt".format(output_path), mode="w") as file:
        for i in range(data_paths_len):
            file.write(outputs[i])

    return True


if __name__ == "__main__":
    main()
