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
from deeponet_room_acoustics.end2end.train1D2D import train

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_settings", type=str, required=True)
  args = parser.parse_args()

  with open(args.path_settings, "r") as json_file:
         settings_train = json.load(json_file)
  
  train(settings_train)

if __name__ == "__main__":
  main()