# Simple Python API for Spot

## Installation

Create the conda env:

```bash
conda create -n spot_env -y python=3.6
conda activate spot_env
```
Install requirements
```bash
pip install -r requirements.txt
```
Install this package
```bash
# Make sure you are in the root of this repo
pip install -e .
```

## Quickstart
Ensure that you are connected to the robot's WiFi.

The following script allows you to move the robot without having to use tablet (which prompts you to enter a password once a day):
```
python -m spot_wrapper.keyboard_teleop
```
If you get an error about the e-stop, you just need to make sure that you run this script in another terminal:
```
python -m spot_wrapper.estop
```

Read through the `spot_wrapper/keyboard_teleop.py` to see most of what this repo offers in terms of actuating the Spot and its arm.

To receive/monitor data (vision/proprioception) from the robot, you can use these scripts:
```
python -m spot_wrapper.view_camera
python -m spot_wrapper.view_arm_proprioception
python -m spot_wrapper.monitor_nav_pose
```
