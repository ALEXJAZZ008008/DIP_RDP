# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os


data_path = "{0}/DIP_RDP_data/dynamic_thorax_simulation/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/dynamic_thorax_simulation/".format(os.getcwd())

noise_path = None

data_window_size = 128
data_window_bool = True

data_resample_power_of = 2

data_input_bool = True

data_gaussian_smooth_sigma_xy = 0.0
data_gaussian_smooth_sigma_z = 0.0

input_gaussian_weight = 1.0
input_poisson_weight = 0.0


model_path = "{0}/model.pkl".format(output_path)

layer_layers = [2, 2, 2, 2, 2, 2, 2, 2]
layer_depth = [1, 2, 4, 8, 16, 32, 64, 128]
layer_groups = [1, 1, 1, 1, 1, 1, 1, 1]
latent_depth = layer_depth[-1]


jitter_magnitude = 1

elastic_jitter_bool = False
elastic_jitter_sigma = 0.0
elastic_jitter_points_iterations = 4


input_gaussian_sigma = 0.0
skip_gaussian_sigma = 0.0
layer_gaussian_sigma = 0.0


dropout = 0.0

bayesian_test_bool = False
bayesian_output_bool = False
bayesian_iterations = 1

if bayesian_output_bool and bayesian_iterations < 2:
    bayesian_iterations = 2


epsilon = 1e-07

total_variation_bool = True
total_variation_weight = 0.0

covariance_weight = 0.0

scale_loss_weight = 0.0
scale_accuracy_scale = 1e00

uncertainty_weight = 0.0

kernel_regulariser_weight = 0.0
activity_regulariser_weight = 0.0


weight_decay = 0.0


backtracking_weight_percentage = None
backtracking_weight_perturbation = 1e-04

patience_smoothing_bool = False
patience_smoothing_magnitude = 9
patience = 10
plateau_cutoff = 1e-04
loss_scaling_patience_skip = 15
