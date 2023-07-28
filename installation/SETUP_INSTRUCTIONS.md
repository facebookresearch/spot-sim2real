# Setup Instructions

### Clone the repo

```bash
git clone git@github.com:facebookresearch/spot-sim2real.git
cd spot-sim2real/
git submodule update --init --recursive
```

### Update the system packages and install required pkgs

```bash
sudo apt-get update
sudo apt-get install gcc
sudo apt-get install g++
sudo apt install tmux
```

### Install Miniconda at /home/<user>

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

# mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Add required channels

```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack
conda config --env --set channel_priority strict
```

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

# Get git lfs (large file system)
sudo apt-get install git-lfs
git lfs install

# Download weights (need only once)
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
If you face any issues in this step, refer to [this section in ISSUES.md](/installation/ISSUES.md#issues-while-running-setuppy-for-detectron2)

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
If you face any issues in this step, refer to [this section in ISSUES.md](/installation/ISSUES.md#issues-while-running-setuppy-for-habitat-lab)

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


### Setting ROS env variables
* If using **ROS on only 1 computer** (i.e. you don't need 2 or more machines in the ROS network), follow these steps
    ```bash
    echo 'export ROS_HOSTNAME=localhost' >> ~/.bashrc
    echo 'export ROS_MASTER_URI=http://localhost:11311' >> ~/.bashrc
    source ~/.bashrc
    ```
* If using **ROS across multiple computers**, follow these steps on each computer
    ```bash
    # your_local_ip = ip address of this computer in the network
    echo 'export ROS_IP=<your_local_ip>' >> ~/.bashrc
    # ros_masters_ip = ip address of the computer running roscore
    echo 'export ROS_MASTER_URI=http://<ros_masters_ip>:11311' >> ~/.bashrc
    source ~/.bashrc
    ```

For assistance with finding the right ip of your computer, [please follow these steps](/installation/ISSUES.md#how-to-find-ip-address-of-local-computer).

### Setup SPOT Robot
- Connect to robot's wifi, password for this wifi can be found in robot's belly after removing battery.
- Make sure that the robot is in access point mode (update to client mode in future). Refer to [this](https://support.bostondynamics.com/s/article/Spot-network-setup) page for information regarding Spot's network setup.

```bash
echo 'export SPOT_ADMIN_PW=<your-spot-admin-password>' >> ~/.bashrc
echo 'export SPOT_IP=<your-spot-ip>' >> ~/.bashrc
source ~/.bashrc
```

### Testing the setup by running simple navigation policy on robot
1. Create waypoints.yaml file using the following command
    ```bash
    spot_rl_waypoint_recorder -x
    ```
2. Follow Steps 1,2,3,4 from [README.md](/README.md#running-the-demo-asclscseq-experts)
3. Go to root of repo, and run simple command to move robot to a new waypoint using the navigation policy. This command will move robot 2.5m in front after undocking. **Ensure there is 2.5m space in front of dock**
    ```bash
    python spot_rl_experiments/spot_rl/envs/nav_env.py -w "test_receptacle"
    ```
4. Once the robot has moved, you can dock back the robot with the following command
    ```bash
    spot_rl_autodock
    ```

### For Meta internal users (with Meta account), please check the following link for the ip and the password

[Link](https://docs.google.com/document/d/1u4x4ZMjHDQi33PB5V2aTZ3snUIV9UdkSOV1zFhRq1Do/edit)

### Mac Users

It is not recommended to run the code on a Mac machine, and we do not support this. However, it is possible to run the code on a Mac machine. Please reach out to Jimmy Yang (jimmytyyang@meta.com) for help.


### For folks who are interested to contribute to this repo, you'll need to setup pre-commit.
The repo runs CI tests on each PR and the PRs are merged only when the all checks have passed.
Installing the pre-commit allows you to run automatic pre-commit while running `git commit`. 
```bash
pre-commit install
```

### Creating an account on CircleCI
- Since we have integrated CircleCI tests on this repo, you would need to create and link your CircleCI account
- You can create your account from this link (https://app.circleci.com/). Once you have created the account, go to "Organization Settings", on the left tab click on "VCS"
- Finally click on "Manage GitHub Checks". CircleCI will request access to `facebookresearch` org owner.
