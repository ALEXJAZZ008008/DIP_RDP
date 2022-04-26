# Copyright University College London 2021, 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import os


# data_path = "{0}/DIP_RDP_data/parameter_search/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/parameter_search/".format(os.getcwd())

# data_path = "{0}/DIP_RDP_data/static_thorax_simulation/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/static_thorax_simulation/".format(os.getcwd())

# data_path = "{0}/DIP_RDP_data/dynamic_thorax_simulation/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/dynamic_thorax_simulation/".format(os.getcwd())

# data_path = "{0}/DIP_RDP_data/kjell_parameter_search/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/kjell_parameter_search/".format(os.getcwd())

# data_path = "{0}/DIP_RDP_data/kjell_dynamic_lung_simulation/".format(os.path.dirname(os.getcwd()))
# output_path = "{0}/output/kjell_dynamic_lung_simulation/".format(os.getcwd())

data_path = "{0}/DIP_RDP_data/kjell_dynamic_liver_simulation/".format(os.path.dirname(os.getcwd()))
output_path = "{0}/output/kjell_dynamic_liver_simulation/".format(os.getcwd())

data_crop_bool = True
data_crop_amount = [256, 6, 0]

data_window_size = 64
data_window_bool = True

data_resample_power_of = 2


total_variation_weight = 1e00


patience = 10
plateau_cutoff = 1e-04
