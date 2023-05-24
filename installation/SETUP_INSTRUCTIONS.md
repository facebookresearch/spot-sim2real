# Setup Instructions

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

# Configure conda
source ~/.bashrc
conda init

# Check
conda --version
```

### Install mamba

```bash
# Install
conda install -c conda-forge mamba

# Export path
echo 'export PATH=/home/<user>/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Check
mamba --version
```

### Create environment (Takes a while)
```bash
# Use the yaml file to setup the environemnt
mamba env create -f installation/environment.yml
source ~/.bashrc
mamba init

# Update bashrc to activate this environment
echo 'mamba activate spot_ros' >> ~/.bashrc
source ~/.bashrc
```

### Install torch and cuda packages
```bash
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### Add required channels
```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-experimental
conda config --env --add channels robostack
conda config --env --set channel_priority strict
```

### Setup bd_spor_wrapper
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
pip install --upgrade --no-cache-dir gdown
gdown --fuzzy https://drive.google.com/file/d/1bZqRLCDv3_E9ijbKuqroWHCBt-jvjjvc/view
unzip weights.zip && rm weights.zip && cd ../
```

### Setup MaskRCNN (Unused in the demo?)
```bash
# Generate module
cd mask_rcnn_detectron2/ && pip install -e .

# Setup detectron
git clone git@github.com:facebookresearch/detectron2.git
pip install -e detectron2 && cd ../
```

### Setup DeblurGAN (Unused in the demo?)
```bash
# Generate module
cd DeblurGANv2/ && pip install -e . && cd ../
```

### Setup Habitat-lab (Unused in the demo?)
```bash
cd habitat-lab/
mamba install -c aihabitat habitat-sim==0.2.1 -y
python setup.py develop --all
cd ../
```

### Install these Packages (should be moved to environment.yaml in future)
```bash
pip install transformers
pip uninstall tensorflow
pip install openai
pip install whisper
```
### Download inceptionresnet weights
```bash
# Create dir to store weights if it does not exist
mkdir -p ~/.cache/torch/hub/checkpoints

# Get weights (May take a while)
wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth -O ~/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth --no-check-certificate
```

### Setup SPOT Robot
- Connect to robot's wifi
- wifi pwd for the *Purple* robot : 2qc6w9fjizk3
- wifi pwd for the *Blue* robot : UPDATE ME
- make sure that the robot is in access point mode (update to client mode in future)

```bash
# For purple spot (Using default Access Point Mode IP)
echo 'export SPOT_ADMIN_PW=i4fhwamvx5rf' >> ~/.bashrc
echo 'export SPOT_IP=192.168.80.3' >> ~/.bashrc

# For blue spot
echo 'export SPOT_ADMIN_PW=reou2fdfgsw7' >> ~/.bashrc
echo 'export SPOT_IP=192.168.80.3' >> ~/.bashrc
```





