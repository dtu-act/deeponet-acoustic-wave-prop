# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import argparse
import json
from deeponet_room_acoustics.end2end.eval1D2D import inference

# id = "spectral_sine_1D"
# input_dir = "/work3/nibor/data/input1D"
# output_path = "/work3/nibor/data/deeponet/output1D"
# custom_eval_data = os.path.join(input_dir, "rect3x3_freq_indep_ppw_2_4_2_from_ppw_dx5_srcs33_val.h5")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_settings", type=str, required=True)
    parser.add_argument("--h5_data", type=str, required=False)
    args = parser.parse_args()

    with open(args.path_settings, "r") as json_file:
         settings_train = json.load(json_file)
        
    custom_eval_data = args.h5_data if args.h5_data else None    
    inference(settings_train, custom_data=custom_eval_data, do_animate=False)

if __name__ == "__main__":
    main()