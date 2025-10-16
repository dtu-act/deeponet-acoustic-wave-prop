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
from deeponet_acoustics.end2end.eval1D2D import inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_settings", type=str, required=True)
    parser.add_argument("--h5_data", type=str, required=False)
    args = parser.parse_args()

    with open(args.path_settings, "r") as json_file:
         settings_train = json.load(json_file)
        
    custom_eval_data = args.h5_data if args.h5_data else None    
    inference(settings_train, custom_data_path=custom_eval_data, do_animate=False)

if __name__ == "__main__":
    main()