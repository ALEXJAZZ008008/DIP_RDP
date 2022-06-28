# Copyright University College London 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import shutil
import errno
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

    for i in range(len(data)):
        data_copy = data[i].copy()

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

        data[i] = data_copy

    return data


def preprocessing(data_array_list, crop_amount, rescale_bool_list, rescale_to_index):
    print("preprocessing")

    data_array_list_len = len(data_array_list)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = data_downsampling_crop(current_data_array,
                                                    [current_data_array[0].shape[0] - crop_amount[0],
                                                     current_data_array[0].shape[1] - crop_amount[1],
                                                     current_data_array[0].shape[2] - crop_amount[2]])

        data_array_list[i] = current_data_array

    for i in range(data_array_list_len):
        data_array_list[i] = np.array(data_array_list[i])

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


def profiles(data_array_list, uncertainty_data_array_list, roi_position, average_list, offset_list, bias_list):
    print("profiles")

    data_array_list_profiles = []

    data_array_list_i_len = len(data_array_list[0])

    for i in range(len(data_array_list)):
        current_data_array_list_profiles = []

        for j in range(data_array_list_i_len):
            current_data_array = data_array_list[i][j]

            if average_list[i] is not None:
                average_list_len = len(average_list[i])

                for k in range(average_list_len):
                    current_data_array = current_data_array + data_array_list[average_list[i][k]]

                current_data_array = current_data_array / (average_list_len + 1.0)

            current_data_array = current_data_array[:, :, offset_list[i][0]:offset_list[i][1]] + bias_list[i]
            current_data_array = current_data_array[roi_position[0][0]:roi_position[0][1],
                                                    roi_position[1][0]:roi_position[1][1],
                                                    roi_position[2][0]:roi_position[2][1]]

            # current_data_array_list_profiles.append(np.mean(current_data_array))
            current_data_array_list_profiles.append(
                current_data_array[int(np.floor(current_data_array.shape[0] / 2.0)),
                                   int(np.floor(current_data_array.shape[1] / 2.0)),
                                   int(np.floor(current_data_array.shape[2] / 2.0))])

        x = np.arange(0, data_array_list_i_len)
        y = np.array(current_data_array_list_profiles)

        poly = PolynomialFeatures(degree=3, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))

        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(poly_features)

        data_array_list_profiles.append(y_predicted)

        if uncertainty_data_array_list[i] is not None:
            current_uncertainty_data_array_list_profiles = []

            for j in range(data_array_list_i_len):
                current_uncertainty_data_array = data_array_list[i][j]

                if average_list[i] is not None:
                    average_list_len = len(average_list[i])

                    for k in range(average_list_len):
                        current_uncertainty_data_array = current_uncertainty_data_array + \
                                                         data_array_list[average_list[i][k]]

                    current_uncertainty_data_array = current_uncertainty_data_array / (average_list_len + 1.0)

                current_uncertainty_data_array = \
                    current_uncertainty_data_array[:, :, offset_list[i][0]:offset_list[i][1]] + bias_list[i]
                current_uncertainty_data_array = current_uncertainty_data_array[roi_position[0][0]:roi_position[0][1],
                                                                                roi_position[1][0]:roi_position[1][1],
                                                                                roi_position[2][0]:roi_position[2][1]]

                # current_uncertainty_data_array_list_profiles.append(np.mean(current_uncertainty_data_array))
                current_uncertainty_data_array_list_profiles.append(
                    current_uncertainty_data_array[int(np.floor(current_uncertainty_data_array.shape[0] / 2.0)),
                                                   int(np.floor(current_uncertainty_data_array.shape[1] / 2.0)),
                                                   int(np.floor(current_uncertainty_data_array.shape[2] / 2.0))])

            sample_weight = np.array(current_uncertainty_data_array_list_profiles)

            sample_weight = (sample_weight * -1)
            sample_weight = sample_weight - np.min(sample_weight)
            sample_weight = (sample_weight / np.sum(sample_weight)) * data_array_list_i_len

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(poly_features, y, sample_weight=sample_weight)
            y_predicted = poly_reg_model.predict(poly_features)

            data_array_list_profiles.append(y_predicted)

    return data_array_list_profiles


def main():
    print("main")

    output_path = "{0}/test/".format(os.getcwd())

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    # data_paths = [["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_1.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_2.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_3.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_4.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_5.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_6.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_7.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_8.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_9.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_10.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_11.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_12.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_13.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_14.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_15.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_16.nii.gz"],
    #               ["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_1.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_2.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_3.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_4.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_5.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_6.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_7.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_8.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_9.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_10.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_11.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_12.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_13.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_14.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_15.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/y/dynamicXcat_WB_noisy_16.nii.gz"],
    #               ["/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/0/0_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/1/1_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/2/2_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/3/3_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/4/4_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/5/5_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/6/6_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/7/7_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/8/8_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/9/9_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/10/10_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/11/11_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/12/12_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/13/13_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/14/14_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"],
    #               ["/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/0/0.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/1/1.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/2/2.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/3/3.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/4/4.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/5/5.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/6/6.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/7/7.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/8/8.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/9/9.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/10/10.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/11/11.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/12/12.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/13/13.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/14/14.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/15/15.nii.gz"],
    #               ["/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/0/0_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/1/1_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/2/2_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/3/3_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/4/4_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/5/5_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/6/6_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/7/7_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/8/8_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/9/9_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/10/10_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/11/11_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/12/12_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/13/13_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/14/14_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"],
    #               ["/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/0/0_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/1/1_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/2/2_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/3/3_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/4/4_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/5/5_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/6/6_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/7/7_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/8/8_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/9/9_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/10/10_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/11/11_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/12/12_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/13/13_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/14/14_output.nii.gz",
    #                "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"]]

    # uncertainty_data_paths = [None,
    #                           None,
    #                           None,
    #                           None,
    #                           ["/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/0/0_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/1/1_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/2/2_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/3/3_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/4/4_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/5/5_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/6/6_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/7/7_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/8/8_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/9/9_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/10/10_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/11/11_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/12/12_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/13/13_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/14/14_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_uncertainty.nii.gz"],
    #                           ["/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/0/0_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/1/1_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/2/2_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/3/3_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/4/4_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/5/5_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/6/6_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/7/7_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/8/8_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/9/9_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/10/10_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/11/11_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/12/12_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/13/13_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/14/14_uncertainty.nii.gz",
    #                            "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_uncertainty.nii.gz"]]

    data_paths = [["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_1.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_2.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_3.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_4.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_5.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_6.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_7.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_8.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_9.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_10.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_11.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_12.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_13.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_14.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_15.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/gt/dynamicXcat_WB_16.nii.gz"],
                  ["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_1.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_2.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_3.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_4.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_5.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_6.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_7.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_8.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_9.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_10.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_11.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_12.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_13.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_14.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_15.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_liver_simulation/y/dynamicXcat_WB_noisy_16.nii.gz"],
                  ["/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/0/0_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/1/1_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/2/2_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/3/3_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/4/4_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/5/5_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/6/6_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/7/7_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/8/8_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/9/9_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/10/10_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/11/11_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/12/12_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/13/13_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/14/14_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_liver_simulation/15/15_output.nii.gz"],
                  ["/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/0/0.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/1/1.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/2/2.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/3/3.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/4/4.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/5/5.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/6/6.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/7/7.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/8/8.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/9/9.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/10/10.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/11/11.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/12/12.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/13/13.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/14/14.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_liver_simulation/15/15.nii.gz"],
                  ["/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/0/0_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/1/1_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/2/2_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/3/3_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/4/4_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/5/5_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/6/6_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/7/7_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/8/8_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/9/9_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/10/10_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/11/11_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/12/12_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/13/13_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/14/14_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/15/15_output.nii.gz"],
                  ["/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/0/0_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/1/1_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/2/2_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/3/3_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/4/4_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/5/5_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/6/6_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/7/7_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/8/8_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/9/9_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/10/10_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/11/11_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/12/12_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/13/13_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/14/14_output.nii.gz",
                   "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/15/15_output.nii.gz"]]

    uncertainty_data_paths = [None,
                              None,
                              None,
                              None,
                              ["/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/0/0_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/1/1_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/2/2_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/3/3_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/4/4_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/5/5_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/6/6_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/7/7_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/8/8_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/9/9_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/10/10_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/11/11_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/12/12_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/13/13_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/14/14_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_liver_simulation/15/15_uncertainty.nii.gz"],
                              ["/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/0/0_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/1/1_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/2/2_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/3/3_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/4/4_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/5/5_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/6/6_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/7/7_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/8/8_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/9/9_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/10/10_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/11/11_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/12/12_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/13/13_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/14/14_uncertainty.nii.gz",
                               "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_liver_simulation/15/15_uncertainty.nii.gz"]]

    crop_amount = [256, 6, 0]

    # roi_position = [[79, 90], [119, 132], [14, 23]]
    roi_position = [[90, 115], [125, 155], [20, 35]]

    rescale_bool_list = [False, False, False, False, False, False]
    rescale_to_index = 0

    average_list = [None, None, None, None, None, None]

    offset_list = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
    bias_list = [0.0, 0.0, 1.5, 1.5, 1.5, 1.5]

    plot_order = [0, 1, 3, 2, 5, 4, 7, 6]

    data_linestyle = ["-", "-", "-", "-", "-", "--", "-", "--"]

    data_legend = ["Ground Truth",
                   "Noisy",
                   "TV",
                   "Original DIP",
                   "New DIP Sequential",
                   "New DIP Uncertainty Sequential",
                   "New DIP Combined",
                   "New DIP Uncertainty Combined"]

    data_array_list = []

    data_paths_len = len(data_paths)
    data_paths_i_len = len(data_paths[0])

    for i in range(data_paths_len):
        current_data_array_list = []

        for j in range(data_paths_i_len):
            current_data_array_list.append(nib.load(data_paths[i][j]).get_fdata())

        data_array_list.append(current_data_array_list)

    uncertainty_data_array_list = []

    for i in range(data_paths_len):
        if uncertainty_data_paths[i] is not None:
            current_uncertainty_data_array_list = []

            for j in range(data_paths_i_len):
                current_uncertainty_data_array_list.append(nib.load(uncertainty_data_paths[i][j]).get_fdata())

            uncertainty_data_array_list.append(current_uncertainty_data_array_list)
        else:
            uncertainty_data_array_list.append(None)

    data_array_list = preprocessing(data_array_list, crop_amount, rescale_bool_list, rescale_to_index)
    output_profiles = profiles(data_array_list, uncertainty_data_array_list, roi_position, average_list, offset_list,
                               bias_list)

    fontsize = 20

    fig, ax = plt.subplots()

    for i in range(len(output_profiles)):
        ax.plot(output_profiles[plot_order[i]], linestyle=data_linestyle[i])

    ax.set_title("Time Activity Curve", fontsize=fontsize)
    ax.set_xlabel("Time", fontsize=fontsize)
    ax.set_ylabel("SUV", fontsize=fontsize)

    ax.legend(data_legend, fontsize=fontsize / 1.5, bbox_to_anchor=(1.0, 1.0), loc="upper left")

    # plt.show()
    plt.savefig("{0}/output.png".format(output_path), format="png", dpi=600, bbox_inches="tight")
    plt.close()

    return True


if __name__ == "__main__":
    main()
