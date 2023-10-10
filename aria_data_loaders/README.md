# Data loaders for Aria data (VRS, MPS)

Project aria already exposes [basic data
utilities](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities).
This package implements data-loaders on top of these utilities to make multi-modal data
access and processing easier.

## Installation

Create the conda env (*skip this step if you are installing this along with spot-sim2real repo*):

```bash
conda create -n aria_env -y python=3.9
conda activate aria_env
```

### Installing FairOtag for QR code detection
Run these commands from outside of spot-sim2real
```bash
git clone -b fairo_viz_subplots git@github.com:KavitShah1998/fairo.git
cd fairo/perception/fairotag/
pip install -e .
```

### Install aria_data_loaders package (will also install requirements)

```bash
# Make sure you are in the root of aria_data_loaders dir
pip install -e .

cd ../spot_rl_experiments/
pip install -e .
```

## Quickstart

Ensure that you have at least the VRS file output from Aria (optionally MPS outputs too).

See [ipython notebook]() for examples on how to use the interface.
