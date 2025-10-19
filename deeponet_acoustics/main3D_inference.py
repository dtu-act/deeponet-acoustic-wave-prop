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

from deeponet_acoustics.end2end.inference3D import inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_settings", type=str, required=True)
    parser.add_argument("--path_eval_settings", type=str, required=True)
    args = parser.parse_args()

    with open(args.path_settings, "r") as json_file:
        settings_train = json.load(json_file)

    with open(args.path_eval_settings, "r") as json_file:
        settings_eval = json.load(json_file)

    inference(settings_train, settings_eval)


if __name__ == "__main__":
    main()
