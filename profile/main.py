# Copyright University College London 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import shutil
import errno
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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


def preprocessing(data_array_list, crop_amount, rescale_bool_list, rescale_to_index):
    print("preprocessing")

    data_array_list_len = len(data_array_list)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = data_downsampling_crop(current_data_array,
                                                    [current_data_array.shape[0] - crop_amount[0],
                                                     current_data_array.shape[1] - crop_amount[1],
                                                     current_data_array.shape[2] - crop_amount[2]])

        data_array_list[i] = current_data_array

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


def profiles(data_array_list, average_list, offset_list, bias_list):
    print("profiles")

    data_array_list_profiles = []

    for i in range(len(data_array_list)):
        current_data_array = data_array_list[i]

        if average_list[i] is not None:
            average_list_len = len(average_list[i])

            for j in range(average_list_len):
                current_data_array = current_data_array + data_array_list[average_list[i][j]]

            current_data_array = current_data_array / (average_list_len + 1.0)

        current_data_array = current_data_array[:, :, offset_list[i][0]:offset_list[i][1]] + bias_list[i]
        # current_data_array = current_data_array[79:90, 119:132, 0:-1]
        current_data_array = current_data_array[int(np.floor((79 + 90) / 2.0)) + 1:int(np.floor((79 + 90) / 2.0)) + 2,
                                                int(np.floor((119 + 132) / 2.0)) + 1:int(np.floor((119 + 132) / 2.0)) + 2,
                                                0:-1]

        data_array_list_profiles.append(np.mean(np.mean(current_data_array, axis=0), axis=0))

    return data_array_list_profiles


def main():
    print("main")

    output_path = "{0}/test/".format(os.getcwd())

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    data_paths = ["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_16.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_16.nii.gz",
                  "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/15/15.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                  "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"]

    crop_amount = [256, 6, 0]

    rescale_bool_list = [True, True, True, True, False, True]
    rescale_to_index = 4

    average_list = [None, None, None, None, None, None]

    offset_list = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
    bias_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    data_linestyle = ["-", "--", "--", "--", "-", "-"]

    data_legend = ["Ground Truth",
                   "Noisy",
                   "TV",
                   "Original DIP",
                   "New DIP Sequential",
                   "New DIP Combined"]

    data_array_list = []

    data_paths_len = len(data_paths)

    for i in range(data_paths_len):
        data_array_list.append(nib.load(data_paths[i]).get_fdata())

    data_array_list = preprocessing(data_array_list, crop_amount, rescale_bool_list, rescale_to_index)
    output_profiles = profiles(data_array_list, average_list, offset_list, bias_list)

    fontsize = 20

    fig, ax = plt.subplots()

    for i in range(len(output_profiles)):
        ax.plot(output_profiles[i], linestyle=data_linestyle[i])

    ax.set_title("Profile Through a Lesion", fontsize=fontsize)
    ax.set_xlabel("Axial Position", fontsize=fontsize)
    ax.set_ylabel("SUV", fontsize=fontsize)

    ax.legend(data_legend, fontsize=fontsize / 1.5, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    # plt.show()
    plt.savefig("{0}/output.png".format(output_path), format="png", dpi=600, bbox_inches="tight")
    plt.close()

    return True


if __name__ == "__main__":
    main()
