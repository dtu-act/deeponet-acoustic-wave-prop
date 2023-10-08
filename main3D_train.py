# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import argparse
from train3D import train

parser = argparse.ArgumentParser()
parser.add_argument("--path_settings", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
  settings_path = args.path_settings  
  train(settings_path)