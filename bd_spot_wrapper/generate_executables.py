import os
import os.path as osp
import sys

this_dir = osp.dirname(osp.abspath(__file__))
base_dir = osp.join(this_dir, "spot_wrapper")
bin_dir = osp.join(os.environ["CONDA_PREFIX"], "bin")

orig_to_alias = {
    "estop": "spot_estop",
    "headless_estop": "spot_headless_estop",
    "home_robot": "spot_reset_home",
    "keyboard_teleop": "spot_keyboard_teleop",
    "monitor_nav_pose": "spot_monitor_nav_pose",
    "roll_over": "spot_roll_over",
    "selfright": "spot_selfright",
    "sit": "spot_sit",
    "stand": "spot_stand",
    "view_arm_proprioception": "spot_view_arm_proprioception",
    "view_camera": "spot_view_camera",
}


print("Generating executables...")
for orig, alias in orig_to_alias.items():
    exe_path = osp.join(bin_dir, alias)
    data = f"#!/usr/bin/env bash \n{sys.executable} -m spot_wrapper.{orig} $@\n"
    with open(exe_path, "w") as f:
        f.write(data)
    os.chmod(exe_path, 33277)
    print("Added:", alias)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("THESE EXECUTABLES ARE ONLY VISIBLE TO THE CURRENT CONDA ENV!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
