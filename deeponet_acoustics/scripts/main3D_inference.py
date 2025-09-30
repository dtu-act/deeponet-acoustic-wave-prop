# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import argparse
from eval3D import evaluate

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_settings", type=str, required=True)
  parser.add_argument("--path_eval_settings", type=str, required=True)
  args = parser.parse_args()
  settings_path = args.path_settings  
  eval_settings_path = args.path_eval_settings  
  evaluate(settings_path, eval_settings_path)

if __name__ == "__main__":
  main()