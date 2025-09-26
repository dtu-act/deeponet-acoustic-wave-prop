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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_settings", type=str, required=True)
    args = parser.parse_args()
    train(args.path_settings)

if __name__ == "__main__":
    main()