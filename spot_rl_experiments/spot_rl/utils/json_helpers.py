# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os


def load_json_files(directory_path):
    """
    Load the contents of all JSON files inside a directory into a list.

    Parameters:
    - directory_path (str): Path to the directory containing JSON files.

    Returns:
    - json_data_list (list): List containing the JSON data from all the files.
    """
    json_data_list = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            json_data = load_json_file(file_path)
            json_data_list.append(json_data)

    return json_data_list


def load_json_file(file_path):
    """
    Load the contents of a JSON file into a Python object.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - json_data (dict or list): JSON data loaded from the file.
    """
    with open(file_path, "r") as file:
        json_data = json.load(file)

    return json_data


def save_json_file(file_path, data):
    """
    Save the contents of a Python object into a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    - data : any Python object

    Returns:
    - json_data (dict or list): JSON data loaded from the file.
    """
    with open(file_path, "w") as output_file:
        json.dump(data, output_file)
        print(f"Data saved successfully to json file at {file_path}")
