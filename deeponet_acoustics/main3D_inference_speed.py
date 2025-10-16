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
from deeponet_acoustics.end2end.inference_speed import evaluate_inference_speed3D

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_settings", type=str, required=True)
  args = parser.parse_args()

  with open(args.path_settings, "r") as json_file:
    settings_train = json.load(json_file)

  # TODO: load eval settings for flexibility
     
  evaluate_inference_speed3D(settings_train)

if __name__ == "__main__":
  main()