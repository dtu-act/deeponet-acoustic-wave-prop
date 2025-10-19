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

from deeponet_acoustics.end2end.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_settings", type=str, required=True)
    args = parser.parse_args()

    with open(args.path_settings, "r") as json_file:
        settings = json.load(json_file)

    train(settings)


if __name__ == "__main__":
    main()
