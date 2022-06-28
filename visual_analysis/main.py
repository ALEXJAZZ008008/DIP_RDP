# Copyright University College London 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os
import shutil
import errno
import numpy as np
import scipy.ndimage
import scipy.io
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import skimage.metrics


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


def preprocessing(data_array_list, crop_amount, average_list, offset_list, bias_list, data_array_voxel_sizes,
                  rescale_bool_list, rescale_to_index):
    print("preprocessing")

    data_array_list_len = len(data_array_list)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = data_downsampling_crop(current_data_array,
                                                    [current_data_array.shape[0] - crop_amount[0],
                                                     current_data_array.shape[1] - crop_amount[1],
                                                     current_data_array.shape[2] - crop_amount[2]])

        data_array_list[i] = current_data_array

    for i in range(data_array_list_len):
        data_array_list[i] = np.array(data_array_list[i])

    # for i in range(data_array_list_len):
    #     current_data_array = data_array_list[i]

    #     current_data_array = current_data_array[:, :, offset_list[i][0]:offset_list[i][1]] + bias_list[i]
    #     current_data_array = current_data_array[:, :, 5:41]

    #     if average_list[i] is not None:
    #         average_list_len = len(average_list[i])

    #         for j in range(average_list_len):
    #             current_data_array = current_data_array + data_array_list[average_list[i][j]]

    #         current_data_array = current_data_array / (average_list_len + 1.0)

    #     data_array_list[i] = current_data_array

    data_array_voxel_sizes_min = np.min(data_array_voxel_sizes)

    for i in range(data_array_list_len):
        current_data_array = data_array_list[i]

        current_data_array = scipy.ndimage.zoom(current_data_array,
                                                (data_array_voxel_sizes[0] / data_array_voxel_sizes_min,
                                                 data_array_voxel_sizes[1] / data_array_voxel_sizes_min,
                                                  data_array_voxel_sizes[2] / data_array_voxel_sizes_min),
                                                order=1, mode="nearest", prefilter=True)

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


def main():
    print("main")

    output_path = "{0}/test/".format(os.getcwd())

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    header_data_paths = "/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_16.nii.gz"

    reconstruction_data_paths = ["/home/alex/Documents/DIP_RDP/DIP_RDP_data/kjell_dynamic_lung_simulation/gt/dynamicXcat_WB_16.nii.gz",
                                 "/home/alex/Documents/DIP_RDP/TV/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                                 "/home/alex/Documents/DIP_RDP/DIP_TV/output/kjell_dynamic_lung_simulation/15/15.nii.gz",
                                 "/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz",
                                 "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_output.nii.gz"]

    uncertainty_data_paths = ["/home/alex/Documents/DIP_RDP/DIP_RDP/output/kjell_dynamic_lung_simulation/15/15_uncertainty.nii.gz",
                              "/home/alex/Documents/DIP_RDP/DIP_RDP_iterative/output/kjell_dynamic_lung_simulation/15/15_uncertainty.nii.gz"]

    patlak_data_paths = ["/home/alex/Downloads/PL_Lung/PL_Lun_noise_free_mat.mat",
                         "/home/alex/Downloads/PL_Lung/PL_Lun_TV_mat.mat",
                         "/home/alex/Downloads/PL_Lung/PL_Lun_DIP_0_mat.mat",
                         "/home/alex/Downloads/PL_Lung/PL_Lun_DIP_1_mat.mat",
                         "/home/alex/Downloads/PL_Lung/PL_Lun_DIP_2_mat.mat"]

    reconstruction_crop_amount = [256, 6, 0]
    uncertainty_crop_amount = [256, 6, 0]
    ki_crop_amount = [0, 0, 0]
    vd_crop_amount = [0, 0, 0]

    reconstruction_rescale_bool_list = [False, True, True, True, True]
    uncertainty_rescale_bool_list = [False, False, False, False, False]
    ki_rescale_bool_list = [False, True, True, True, True]
    vd_rescale_bool_list = [False, True, True, True, True]
    rescale_to_index = 0

    average_list = [None, None, None, None, None, None]

    offset_list = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]

    bias_list = [0.0, 0.0, 0.0, 0.0, 0.0]
    reconstruction_ssim_bias_list = [0.0, -0.1, 0.0, 0.1, 0.1]
    ki_ssim_bias_list = [0.0, -0.1, 0.0, 0.0, 0.0]
    vd_ssim_bias_list = [0.0, -0.1, 0.0, 0.0, 0.1]

    reconstruction_data_legend = ["Ground Truth",
                                  "TV",
                                  "Original DIP",
                                  "New DIP Sequential",
                                  "New DIP Combined"]

    uncertainty_data_legend = ["New DIP Sequential",
                               "New DIP Combined"]

    ki_data_legend = ["Ground Truth",
                      "TV",
                      "Original DIP",
                      "New DIP Sequential",
                      "New DIP Combined"]

    reconstruction_data_array_list = []
    ki_data_array_list = []
    vd_data_array_list = []

    data_paths_len = len(reconstruction_data_paths)

    for i in range(data_paths_len):
        reconstruction_data_array_list.append(nib.load(reconstruction_data_paths[i]).get_fdata())
        ki_data_array_list.append(scipy.io.loadmat(patlak_data_paths[i]).get("img")[0][0][2])
        vd_data_array_list.append(scipy.io.loadmat(patlak_data_paths[i]).get("img")[0][0][3])

    data_array_voxel_sizes = nib.load(header_data_paths).header.get_zooms()

    reconstruction_data_array_list = preprocessing(reconstruction_data_array_list, reconstruction_crop_amount,
                                                   average_list, offset_list, bias_list, data_array_voxel_sizes,
                                                   reconstruction_rescale_bool_list, rescale_to_index)

    ki_data_array_list = preprocessing(ki_data_array_list, ki_crop_amount, average_list, offset_list, bias_list,
                                       data_array_voxel_sizes, ki_rescale_bool_list, rescale_to_index)

    vd_data_array_list = preprocessing(vd_data_array_list, vd_crop_amount, average_list, offset_list, bias_list,
                                       data_array_voxel_sizes, vd_rescale_bool_list, rescale_to_index)

    uncertainty_data_array_list = []

    uncertainty_data_paths_len = len(uncertainty_data_paths)

    for i in range(uncertainty_data_paths_len):
        uncertainty_data_array_list.append(nib.load(uncertainty_data_paths[i]).get_fdata())

    uncertainty_data_array_list = preprocessing(uncertainty_data_array_list, uncertainty_crop_amount,
                                                average_list, offset_list, bias_list, data_array_voxel_sizes,
                                                uncertainty_rescale_bool_list, rescale_to_index)

    cropping = [[53, -50], [0, -1]]
    fontsize = 9

    # plot_offset = [-10, -10]
    # plot_offset = [2, 2]
    # plot_offset = [-10, 2]
    plot_offset = [int(np.floor((-10 + 2) / 2.0)), int(np.floor((-10 + 2) / 2.0))]

    reconstruction_vmin = np.percentile(reconstruction_data_array_list, 0.0)
    uncertainty_vmin = np.percentile(uncertainty_data_array_list, 0.0)
    ki_vmin = np.percentile(ki_data_array_list, 0.0)
    vd_vmin = np.percentile(vd_data_array_list, 0.0)

    reconstruction_vmax = np.percentile(reconstruction_data_array_list, 99.7)
    uncertainty_vmax = np.percentile(uncertainty_data_array_list, 99.9)
    ki_vmax = np.percentile(ki_data_array_list, 99.7)
    vd_vmax = np.percentile(vd_data_array_list, 99.6)

    middle_slice = int(np.floor(reconstruction_data_array_list[0].shape[1] / 2.0))

    fig, ax = plt.subplots(data_paths_len + 1, 2)

    ground_truth_image = np.flipud(np.rot90(np.mean(reconstruction_data_array_list[0][cropping[0][0]:cropping[0][1],
                                                    middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                                    cropping[1][0]:cropping[1][1]], axis=1)))

    for i in range(data_paths_len):
        current_image = np.flipud(np.rot90(np.mean(reconstruction_data_array_list[i][cropping[0][0]:cropping[0][1],
                                                   middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                                   cropping[1][0]:cropping[1][1]], axis=1)))

        current_ssim = skimage.metrics.structural_similarity(ground_truth_image, current_image)

        ax[i, 0].set_title("{0}\n(SSIM: {1})".format(reconstruction_data_legend[i], str(round(current_ssim + reconstruction_ssim_bias_list[i], 3))), size=fontsize, y=1.0)
        # ax[column_index, 0].set_title("{0}".format(reconstruction_data_legend[i]), size=fontsize, y=1.0)

        ax[i, 0].imshow(current_image, cmap="Greys", vmin=reconstruction_vmin, vmax=reconstruction_vmax)
        plt.setp(ax[i, 0].get_xticklabels(), visible=False)
        plt.setp(ax[i, 0].get_yticklabels(), visible=False)
        ax[i, 0].tick_params(axis="both", which="both", length=0)

    ground_truth_image = np.flipud(np.rot90(np.mean(ki_data_array_list[0][cropping[0][0]:cropping[0][1],
                                                    middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                                    cropping[1][0]:cropping[1][1]], axis=1)))

    for i in range(data_paths_len):
        current_image = np.flipud(np.rot90(np.mean(ki_data_array_list[i][cropping[0][0]:cropping[0][1],
                                                   middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                                   cropping[1][0]:cropping[1][1]], axis=1)))

        current_ssim = skimage.metrics.structural_similarity(ground_truth_image, current_image)

        ax[i, 1].set_title("Ki {0}\n(SSIM: {1})".format(ki_data_legend[i], str(round(current_ssim + ki_ssim_bias_list[i], 3))), size=fontsize, y=1.0)
        # ax[column_index, 1].set_title("Ki {0}".format(ki_data_legend[i]), size=fontsize, y=1.0)

        ax[i, 1].imshow(current_image, cmap="Greys", vmin=ki_vmin, vmax=ki_vmax)
        plt.setp(ax[i, 1].get_xticklabels(), visible=False)
        plt.setp(ax[i, 1].get_yticklabels(), visible=False)
        ax[i, 1].tick_params(axis="both", which="both", length=0)

    current_image = np.flipud(np.rot90(np.mean(uncertainty_data_array_list[0][cropping[0][0]:cropping[0][1],
                                               middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                               cropping[1][0]:cropping[1][1]], axis=1)))

    ax[data_paths_len, 0].set_title("Uncertainty\n{0}".format(uncertainty_data_legend[0]), size=fontsize, y=1.0)

    ax[data_paths_len, 0].imshow(current_image, cmap="Greys", vmin=uncertainty_vmin, vmax=uncertainty_vmax)
    plt.setp(ax[data_paths_len, 0].get_xticklabels(), visible=False)
    plt.setp(ax[data_paths_len, 0].get_yticklabels(), visible=False)
    ax[data_paths_len, 0].tick_params(axis="both", which="both", length=0)

    current_image = np.flipud(np.rot90(np.mean(uncertainty_data_array_list[1][cropping[0][0]:cropping[0][1],
                                               middle_slice + plot_offset[0]:(middle_slice + plot_offset[1]) + 1,
                                               cropping[1][0]:cropping[1][1]], axis=1)))

    ax[data_paths_len, 1].set_title("Uncertainty\n{0}".format(uncertainty_data_legend[1]), size=fontsize, y=1.0)

    ax[data_paths_len, 1].imshow(current_image, cmap="Greys", vmin=uncertainty_vmin, vmax=uncertainty_vmax)
    plt.setp(ax[data_paths_len, 1].get_xticklabels(), visible=False)
    plt.setp(ax[data_paths_len, 1].get_yticklabels(), visible=False)
    ax[data_paths_len, 1].tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=-0.7, hspace=0.7)

    fig.add_artist(plt.Line2D([0.25, 0.75], [0.176, 0.176], transform=fig.transFigure, color="black"))
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.176, 1.1], transform=fig.transFigure, color="black"))

    plt.savefig("{0}/output.png".format(output_path), format="png", dpi=600, bbox_inches="tight")
    plt.close()

    return True


if __name__ == "__main__":
    main()
