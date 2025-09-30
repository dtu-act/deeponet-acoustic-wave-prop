# ==============================================================================
# Copyright 2025 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import argparse
from train1D2D import train

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_settings", type=str, required=True)
  args = parser.parse_args()
  settings_path = args.path_settings  
  train(settings_path)

if __name__ == "__main__":
  main()