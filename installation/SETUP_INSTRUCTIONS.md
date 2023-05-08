# Setup Instructions

### Clone the repo

```bash
git clone git@github.com:facebookresearch/spot-sim2real.git
cd spot-sim2real/
git submodule update --init --recursive
```

### Update the system packages

```bash
sudo apt-get update
```

### Install Miniconda at /home/user

```bash
# Download miniconda
cd ~/ && wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh

# Run the script
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
```

### Follow these instructions while installing Miniconda

```bash
Do you accept the license terms? [yes|no]
[no] >>>
Please answer 'yes' or 'no':' -- <type yes>

Miniconda3 will now be installed into this location:
/home/<user>/miniconda3  -- <Press enter>

Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] -- <type yes?

```

### Source and initialize conda

```bash
source ~/.bashrc
conda init

# Export path (**Please update the path as per your setup configuration**)
echo 'export PATH=/home/<user>/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Check
conda --version
```

### Install mamba

```bash
# Install
conda install -c conda-forge mamba

# Check
mamba --version
```

### Create environment (Takes a while)

```bash
# cd into the cloned repository
cd ~/spot-sim2real/

# Use the yaml file to setup the environemnt
mamba env create -f installation/environment.yml
source ~/.bashrc
mamba init

# Update bashrc to activate this environment
echo 'mamba activate spot_ros' >> ~/.bashrc
source ~/.bashrc
```

### Install torch and cuda packages (**through the installation preview, ensure all of the following packages are installed as CUDA versions and not CPU versions**)

```bash
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# If this command fails for error "Could not solve for environment specs", run the following
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Add required channels

```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack
conda config --env --set channel_priority strict
```

### Setup pre-commit.  This allows you to run automatic pre-commit on running `git commit`

```bash
pre-commit install
```

### Creating an account on CircleCI

- Since we have integrated CircleCI tests on this repo, you would need to create and link your CircleCI account
- You can create your account from this link (https://app.circleci.com/). Once you have created the account, go to "Organization Settings", on the left tab click on "VCS"
- Finally click on "Manage GitHub Checks". CircleCI will request access to `facebookresearch` org owner.

### Setup bd_spot_wrapper

```bash
# Generate module
cd bd_spot_wrapper/ && python generate_executables.py
pip install -e . && cd ../
```

### Setup spot_rl_experiments

```bash
# Generate module
cd spot_rl_experiments/ && python generate_executables.py
pip install -e .

# Download weights (need only once)
git lfs install
git clone https://huggingface.co/spaces/jimmytyyang/spot-sim2real-data
unzip spot-sim2real-data/weight/weights.zip && rm -rf spot-sim2real-data && cd ../
```

### Setup MaskRCNN

```bash
# Generate module
cd third_party/mask_rcnn_detectron2/ && pip install -e .

# Setup detectron
git clone git@github.com:facebookresearch/detectron2.git
pip install -e detectron2 && cd ../../
```

### Setup DeblurGAN

```bash
# Generate module
cd third_party/DeblurGANv2/ && pip install -e . && cd ../../
```

### Setup Habitat-lab

```bash
cd third_party/habitat-lab/
mamba install -c aihabitat habitat-sim==0.2.1 -y
python setup.py develop --all
cd ../../
```

### Download inceptionresnet weights

```bash
# Create dir to store weights if it does not exist
mkdir -p ~/.cache/torch/hub/checkpoints

# Get weights (May take a while)
wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth -O ~/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth --no-check-certificate
```

### Ensure you have port-audio library for sounddevice (useful for connecting external microphones for speech-to-text)

```bash
sudo apt-get install libportaudio2
```

### Setup SPOT Robot

- Connect to robot's wifi, password for this wifi can be found in robot's belly after removing battery.
- Make sure that the robot is in access point mode (update to client mode in future). Refer to [this](https://support.bostondynamics.com/s/article/Spot-network-setup) page for information regarding Spot's network setup.

```bash
echo 'export SPOT_ADMIN_PW=<your-spot-admin-password>' >> ~/.bashrc
echo 'export SPOT_IP=<your-spot-ip>' >> ~/.bashrc
source ~/.bash_profile
```

### For Meta internal users (with Meta account), please check the following link for the ip and the password

[Link](https://docs.google.com/document/d/1u4x4ZMjHDQi33PB5V2aTZ3snUIV9UdkSOV1zFhRq1Do/edit)

### Mac Users

It is not recommended to run the code on a Mac machine, and we do not support this. However, it is possible to run the code on a Mac machine. Please reach out to Jimmy Yang (jimmytyyang@meta.com) for help.
