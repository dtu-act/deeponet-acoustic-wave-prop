# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen 
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import json

def parseSettings(path_to_json):    
    file_handle = open(path_to_json, "r")
    data = file_handle.read()
    json_obj = json.loads(data)
    file_handle.close()

    return json_obj