# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os


# data_path = "{0}/DIP_RDP_data/static_thorax_simulation/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/static_thorax_simulation/".format(os.getcwd())

data_path = "{0}/DIP_RDP_data/dynamic_thorax_simulation/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/dynamic_thorax_simulation/".format(os.getcwd())

data_window_size = 64
data_window_bool = True


total_variation_bool = False
total_variation_weight = 1e-04


vanilla_max_iteration = 6062

patience = 10
plateau_cutoff = 1e-04
