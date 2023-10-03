# Data loaders for Aria data (VRS, MPS)

Project aria already exposes [basic data
utilities](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities).
This package implements data-loaders on top of these utilities to make multi-modal data
access and processing easier.

## Installation

Create the conda env:

```bash
conda create -n aria_env -y python=3.9
conda activate aria_env
```

Install this package (will also install requirements)

```bash
# Make sure you are in the root of this repo
pip install -e .
```

## Quickstart

Ensure that you have at least the VRS file output from Aria (optionally MPS outputs too).

See [ipython notebook]() for examples on how to use the interface.
