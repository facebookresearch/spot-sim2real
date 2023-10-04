"""
Experimental script to detect objects in ADT episodes and dump results in a csv file.
The CSV file contains:
    - episode_id
    - timestamp
    - object_id
    - object_name
    - bbox (x1, y1, x2, y2)
    - confidence
    - object_location (x, y, z)
    - wearer_location (x, y, z)
"""

import click
import yaml
from adt_data_utils.adt_data_loader import ADTSequences

experiment_path = ""  # TODO: set this to the path of the experiment folder


@click.command()
@click.option("data-path", type=str)
@click.option("experiment-path", type=str)
def main(data_path, experiment_path):
    adt = ADTSequences(data_path, is_path=False)
    experiment = yaml.load(open(experiment_path, "r"), Loader=yaml.FullLoader)
