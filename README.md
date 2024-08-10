# :robot: Spot-Sim2Real
Spot-Sim2Real is a modular library for development of Spot for embodied AI tasks (e.g., [Language-guided Skill Coordination (LSC)](https://languageguidedskillcoordination.github.io/), [Adaptive Skill Coordination (ASC)](https://arxiv.org/pdf/2304.00410.pdf)) -- configuring Spot robots, controlling sensorimotor skills, and coordinating Large Language Models (LLMs).

## :memo: Setup instructions
Please refer to the [setup instructions page](/installation/SETUP_INSTRUCTIONS.md) for information on how to setup the repo. Note that this repo by-default does not track dirty status of submodules, if you're making any intentional changes within the third-party packages be sure to track them separately.

## :computer: Connecting to the robot
Computer can be connected to the robot in one of the following modes.
1. Ethernet (Gives best network speed, but it is cluttery :sad: )\
This mode can be used to create a wired connection with the robot. Useful for teleoperating the robot via computer
2. Access Point Mode\
This is a wireless mode where robot creates its wifi network. Connect robot to this mode for teleoperating it using controller over long distances. Robot is in Access Point mode if you see a wifi with name like `spot-BD-***********` (where * is a number)
3. Client Mode (Gives 2nd best network speed, we usually prefer this)\
This is a wireless mode where robot is connected to an external wifi network (from a nearby router). Computer should be connected to this same network, wired connection between router and computer will be faster than wireless connection.

**Follow the steps from [Spot's Network Setup](https://support.bostondynamics.com/s/article/Spot-network-setup) page by Boston Dynamics to connect to the robot.**

After setting up spot in correct network configuration, please add its IP inside bashrc
```bash
echo "export SPOT_IP=<spot's ip address>" >> ~/.bashrc
source ~/.bashrc
```

Test and ensure you can ping spot
```bash
ping $SPOT_IP
```

If you get response like this, then you are on right network
```bash
(spot_ros) user@linux-machine:~$ ping $SPOT_IP
PING 192.168.1.5 (192.168.1.5) 56(84) bytes of data.
64 bytes from 192.168.1.5: icmp_seq=1 ttl=64 time=8.87 ms
64 bytes from 192.168.1.5: icmp_seq=2 ttl=64 time=7.36 ms
```

Before starting to run the code, you need to ensure that all ROS env variables are setup properly inside bashrc. Please follow the steps from [Setting ROS env variables](/installation/SETUP_INSTRUCTIONS.md#setting-ros-env-variables) for proper ROS env var setup.

## :desktop_computer: Getting to the repo
Go to the repository
```bash
cd /path/to/spot-sim2real/
```

The code for the demo lies inside the `main` branch.
```bash
# Check your current git branch
git rev-parse --abbrev-ref HEAD

# If you are not in the `main` branch, then checkout to the `main` branch
git checkout main
```

## :light_rail: Try teleoperating the robot using keyboard
### :rotating_light: Running Emergency Stop
* Since we do not have a physical emergency stop button (like the large red push buttons), we need to run an e-stop node.
    ```bash
    python -m spot_wrapper.estop
    ```

- Keep this window open at all the times, if the robot starts misbehaving you should be able to quickly press `s` or `space_bar` to kill the robot

### :musical_keyboard: Running keyboard teleop
* Ensure you have the Estop up and running in one terminal. [Follow these instructions for e-stop](/README.md#rotating_light-running-emergency-stop)
* Run keyboard teleop with this command in a new terminal
    ```bash
    spot_keyboard_teleop
    ```

## :video_game: Instructions to record waypoints (use joystick to move robot around)
- Before running scripts on the robot, waypoints should be recorded. These waypoints exist inside file `spot-sim2real/spot_rl_experiments/configs/waypoints.yaml`

- Before recording receptacles, make the robot sit at home position then run following command
    ```bash
    spot_reset_home
    ```

- There are 2 types of waypoints that one can record,
    1. clutter - These only require the (x, y, theta) of the target receptacle
    2. place - These requre (x, y, theta) for target receptable as well as (x, y, z) for exact drop location on the receptacle

- To record a clutter target, teleoperate the robot to reach near the receptacle target (using joystick). Once robot is at a close distance to receptacle, run the following command
    ```bash
    spot_rl_waypoint_recorder -c <name_for_clutter_receptacle>
    ```

- To record a place target, teleoperate the robot to reach near the receptacle target (using joystick). Once robot is at a close distance to receptacle, use manipulation mode in the joystick to manipulate the end-effector at desired (x,y,z) position. Once you are satisfied with the end-effector position, run the following command
    ```bash
    spot_rl_waypoint_recorder -p <name_for_place_receptacle>
    ```


## :rocket: Running instructions
### Running the demo (ASC/LSC/Seq-Experts)
#### Step1. Run the local launch executable
- In a new terminal, run the executable as
    ```bash
    spot_rl_launch_local
    ```
    This command starts 4 tmux sessions\n

    1. roscore
    2. img_publishers
    3. proprioception
    4. tts

- You can run `tmux ls` in the terminal to ensure that all 4 tmux sessions are running.
    You need to ensure that all 4 sessions remain active until 70 seconds after running the `spot_rl_launch_local`. If anyone of them dies before 70 seconds, it means there is some issue and you should rerun `spot_rl_launch_local`.

- You should try re-running `spot_rl_launch_local` atleast 2-3 times to see if the issue still persists. Many times roscore takes a while to start due to which other nodes die, re-running can fix this issue.

- You can verify if all ros nodes are up and running as expected if the output of `rostopic list` looks like the following
    ```bash
    (spot_ros) user@linux-machine:~$ rostopic list
    /filtered_hand_depth
    /filtered_head_depth
    /hand_rgb
    /mask_rcnn_detections
    /mask_rcnn_visualizations
    /raw_hand_depth
    /raw_head_depth
    /robot_state
    /rosout
    /rosout_agg
    /text_to_speech
    ```
- If you don't get the output as follows, one of the tmux sessions might be failing. Follow [the debugging strategies](/installation/ISSUES.md#debugging-strategies-for-spot_rl_launch_local-if-any-one-of-the-4-sessions-are-dying-before-70-seconds) described in ISSUES.md for triaging and resolving these errors.

#### Step2. Run ROS image visualization
- This is the image visualization tool that helps to understand what robot is seeing and perceiving from the world
    ```bash
    spot_rl_ros_img_vis
    ```
- Running this command will open an image viewer and start printing image frequency from different rosotopics.

- If the image frequency corresponding to `mask_rcnn_visualizations` is too large and constant (like below), it means that the bounding box detector has not been fully initialized yet
    ```bash
    raw_head_depth: 9.33 raw_hand_depth: 9.33 hand_rgb: 9.33 filtered_head_depth: 11.20 filtered_hand_depth: 11.20 mask_rcnn_visualizations: 11.20
    raw_head_depth: 9.33 raw_hand_depth: 9.33 hand_rgb: 9.33 filtered_head_depth: 11.20 filtered_hand_depth: 8.57 mask_rcnn_visualizations: 11.20
    raw_head_depth: 9.33 raw_hand_depth: 9.33 hand_rgb: 9.33 filtered_head_depth: 8.34 filtered_hand_depth: 8.57 mask_rcnn_visualizations: 11.20
    ```

    Once the `mask_rcnn_visualizations` start becoming dynamic (like below), you can proceed with next steps
    ```bash
    raw_head_depth: 6.87 raw_hand_depth: 6.88 hand_rgb: 6.86 filtered_head_depth: 4.77 filtered_hand_depth: 5.01 mask_rcnn_visualizations: 6.14
    raw_head_depth: 6.87 raw_hand_depth: 6.88 hand_rgb: 6.86 filtered_head_depth: 4.77 filtered_hand_depth: 5.01 mask_rcnn_visualizations: 5.33
    raw_head_depth: 4.14 raw_hand_depth: 4.15 hand_rgb: 4.13 filtered_head_depth: 4.15 filtered_hand_depth: 4.12 mask_rcnn_visualizations: 4.03
    raw_head_depth: 4.11 raw_hand_depth: 4.12 hand_rgb: 4.10 filtered_head_depth: 4.15 filtered_hand_depth: 4.12 mask_rcnn_visualizations: 4.03
    ```

#### Step3. Reset home **in a new terminal**
- This is an important step. Ensure robot is at its start location and sitting, then run the following command in a new terminal
    ```bash
    spot_reset_home
    ```

- The waypoints that were recorded are w.r.t the home location. Since the odometry drifts while robot is moving, **it is necessary to reset home before start of every new run**

#### Step4. Emergency stop
- Follow the steps described in [e-stop section](/README.md#rotating_light-running-emergency-stop)


#### Step5. Main demo code **in a new terminal**
- Ensure you have correctly added the waypoints of interest by following the [intructions to record waypoints](/README.md#rocket-running-instructions)
- In a new terminal you can now run the code of your choice
    1. To run Sequencial experts
        ```bash
        spot_rl_mobile_manipulation_env
        ```

    2. To run Adaptive skill coordination
        ```bash
        spot_rl_mobile_manipulation_env -m
        ```

    3. To run Language instructions with Sequencial experts, *ensure the usb microphone is connected to the computer*
        ```bash
        python spot_rl_experiments/spot_rl/envs/lang_env.py
        ```


- If you are done with demo of one of the above code and want to run another code, you do not need to re-run other sessions and nodes. Running a new command in the same terminal will work just fine. But **make sure to bring robot at home location and reset its home** using `spot_reset_home` in the same terminal

#### Step6. [Optional] Pick with Pose estimation (uses NVIDIA's FoundationPose Model)
- Ensure [FoundationPoseForSpotSim2real](https://github.com/tusharsangam/FoundationPoseForSpotSim2Real) is setup as submodule in third_party folder please follow instructions in [SETUP_INSTRUCTIONS.mds](./installation/SETUP_INSTRUCTIONS.md)
- Currently we only support pose estimation for bottle, penguine plush toy & paper cup found in FB offices' microktichen
- New Meshes can be added using 360 video of the object from any camera (iphone, android), entire process will be described in the above repo's README
- Pose estimation model [FoundationPose](https://nvlabs.github.io/FoundationPose/) runs as a microservice & can be communicated through pyzmq socket
- The [Step 1](#step1-run-the-local-launch-executable) should also start the pose estimation service & no other step is required to start this microservice
- <b>How to use Pose Estimation ?</b>
    - You can pass two flags `enable_pose_estimation` & `enable_pose_correction` with `pick` skill as `skillmanager.pick(enable_pose_estimation=True, enable_pose_correction=True)`
    - If you enable pose correction, spot will first manually correct the object pose for eg. rotate horizontal object to be vertical etc & place the corrected the object at the same place.
    - Our `orientationsolver` can also correct the object to face the camera but it incurs additional pick attempt before place can be run thus is kept to be false by default
    - <b>Enabling pose estimation can help in two major way</b> - informs grasp api how to approach the object viz. topdown or side which increases the grasp success probability & correct object orientation before place is ran.


#### Step7. [Optional] Object detection with tracking (uses Meta's SAM2 Model)
- Follow the instruction to setup SAM2 in [SAM2 github page](https://github.com/facebookresearch/segment-anything-2). You can create a new conda environment, different from spot conda environment. The reason for not merging these two environments is that spot-sim2real currently use a lower python version.
- Once finishing the installation, run tracking service on its own conda environment by
```bash
python spot_rl_experiments/spot_rl/utils/tracking_service.py
```
- We support open/close skills using SAM2's tracking. It is done by setting
```python
import rospy
rospy.set_param("enable_tracking", True)
```


### Using Spot Data-logger
All logs will get stored inside `data/data_logs` directory

#### Logged keys
The logger will capture spot's data such that each timestamp's log packet is a dict with following keys:
```bash
"timestamp" : double, # UTC epoch time from time.time()
"datatime": str # human readable corresponding local time as "YY-MM-DD HH:MM:SS"
"camera_data" : [
                    {
                        "src_info" : str, # this is name of camera source as defined in SpotCamIds
                        "raw_image": np.ndarray, # this is spot's camera data as cv2 (see output of Spot.image_response_to_cv2() for more info)
                        "camera_intrinsics": np.ndarray, # this is 3x3 matrix holding camera intrinsics
                        "base_T_camera": np.ndarray, # this is 4x4 transformation matrix of camera w.r.t base frame of robot
                    },
                    ...
                ],
"vision_T_base": np.ndarray, # this is 4x4 transformation matrix of base frame w.r.t vision frame
"base_pose_xyt": np.ndarray, # this is 3 element array representing x,y,yaw w.r.t home frame
"arm_pose": np.array, # this is a 6 element array representing arm joint states (ordering : sh0, sh1, el0, el1, wr0, wr1)
"is_gripper_holding_item": bool, # whether gripper is holding something or not
"gripper_open_percentage": double, # how much is the gripper open
"gripper_force_in_hand": np.ndarray, # force estimate on end-effector in hand frame
```


#### Logging data
The data logger is designed to log the data provided [here](/README.md#logged-keys) at whatever rate sensor data becomes available (which depends on network setup).


To run the logger async, simply run the following command in a new terminal
```bash
python -m spot_wrapper.data_logger --log_data
```
This will record data in a while loop, press `Ctrl+c` to spot the logger. That will save the log file inside `data/data_logs/<YY,MM,DD-HH,MM,SS>.pkl` file

Warning : This logger will cause motion blur as camera data is logged while the robot moves. Currently we do not support Spot-Record-Go protocol to log

#### Log replay
It is also possible to replay the logged data (essentially the camera streams that have been logged) using the following command :
```bash
python -m spot_wrapper.data_logger --replay="<name_of_log_file>.pkl"
```
Caution : For replay, the log file SHOULD be a pkl file with the keys provided [here](/README.md#logged-keys)

Caution : Please ensure the log file is present inside `data/data_logs` dir.


## :wrench: Call skills (non-blocking) without installing spot-sim2real in your home environment
We provide an function that can call skills in seperate conda environment. And the calling of skill itself is a non-blocking call.
#### Step1. Follow Running instructions section to setup image client in spot_ros conda environment
#### Step2. Run ```skill_executor.py``` to listen to which skill to use. This will run on the background.
```bash
python spot_rl_experiments/spot_rl/utils/skill_executor.py
```
#### Step3. Use ROS to use skill in your application. Now you can call skills in non-blocking way.
```python
# In your application, you import rospy for calling which skill to use
import rospy # This is the only package you need to install in your environment
rospy.set_param("skill_name_input", "Navigate,desk") # Call navigation skills to navigate to the desk. This is a non-blocking call.
```

## :eyeglasses: Run Spot-Aria project code
Follow the steps in [the project documentation](./aria_data_loaders/README.md#scroll-steps-to-run-episodic-memory-robotic-fetch-demostration).

## :star: Convert pytorch weights to torchscript
To convert pytorch weights to torchscript, please follow [Torchscript Conversion Instructions.](./spot_rl_experiments/utils/README.md)

## :mega: Acknowledgement
We thank [Naoki Yokoyama](http://naoki.io/) for setting up the foundation of the codebase, and [Joanne Truong](https://www.joannetruong.com/) for polishing the codebase. Spot-Sim2Real is built upon Naoki's codebases: [bd_spot_wrapper](https://github.com/naokiyokoyama/bd_spot_wrapper) and [spot_rl_experiments
](https://github.com/naokiyokoyama/spot_rl_experiments), and with new features (LLMs, pytest) and improving robustness.


## :writing_hand: Citations
If you find this repository helpful, feel free to cite our papers: [Adaptive Skill Coordination (ASC)](https://arxiv.org/pdf/2304.00410.pdf) and [Language-guided Skill Coordination (LSC)](https://languageguidedskillcoordination.github.io/).
```
@article{yokoyama2023adaptive,
  title={Adaptive Skill Coordination for Robotic Mobile Manipulation},
  author={Yokoyama, Naoki and Clegg, Alexander William and Truong, Joanne and Undersander, Eric  and Yang, Tsung-Yen and Arnaud, Sergio and Ha, Sehoon and Batra, Dhruv and Rai, Akshara},
  journal={arXiv preprint arXiv:2304.00410},
  year={2023}
}

@misc{yang2023adaptive,
    title={LSC: Language-guided Skill Coordination for Open-Vocabulary Mobile Pick-and-Place},
    author={Yang, Tsung-Yen and Arnaud, Sergio and Shah, Kavit and Yokoyama, Naoki and  Clegg, Alexander William and Truong, Joanne and Undersander, Eric and Maksymets, Oleksandr and Ha, Sehoon and Kalakrishnan, Mrinal and Mottaghi, Roozbeh and Batra, Dhruv and Rai, Akshara},
    howpublished={\url{https://languageguidedskillcoordination.github.io/}}
}
```

## License
Spot-Sim2Real is MIT licensed. See the [LICENSE file](/LICENSE) for details.
