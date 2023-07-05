# :robot: Repo: spot_sim2real
Spot-Sim2Real is a modular library for development of Spot for embodied AI tasks (e.g., [Language-guided Skill Coordination (LSC)](https://languageguidedskillcoordination.github.io/), [Adaptive Skill Coordination (ASC)](https://arxiv.org/pdf/2304.00410.pdf)) -- configuring Spot robots, controlling sensorimotor skills, and coordinating Large Language Models (LLMs).

## Setup instructions - Please refer [here](/installation/SETUP_INSTRUCTIONS.md) for information on how to setup the repo.

## Connecting to the robot
Computer can be connected to the robot in one of the following modes
1. Ethernet\
This mode can be used to create a wired connection with the robot. Useful for teleoperating the robot via computer
2. Access Point Mode\
This is a wireless mode where robot creates its wifi network. Connect robot to this mode for teleoperating it using controller over long distances. Robot is in Access Point mode if you see a wifi with name like `spot-BD-***********` (where * is a number)
3. Client Mode (To be used for CVPR demos)\
This is a wireless mode where robot is connected to an external wifi network (from a nearby router). Computer should be connected to this same network, wired connection between router and computer will be faster than wireless connection.

### Ensure that the robot's IP is correct in Client mode
This can be checked by going into wifi settings on the computer and making sure you do not see any wifi with name `spot-BD-********` (where * is a number)

Once you have **connected to the correct wifi**, ensure you can ping spot properly
```bash
ping $SPOT_IP
```

If you get response like this, then you are on right network
```bash
(spot_ros) kavitshah@frerd001:~$ ping $SPOT_IP
PING 192.168.1.5 (192.168.1.5) 56(84) bytes of data.
64 bytes from 192.168.1.5: icmp_seq=1 ttl=64 time=8.87 ms
64 bytes from 192.168.1.5: icmp_seq=2 ttl=64 time=7.36 ms
```

If you don't get a responce however then probably the IP in client mode is incorrect. Follow the "Connect to Spot via Direct Ethernet" instructions [here](https://support.bostondynamics.com/s/article/Spot-network-setup) to open Spot's network manager, select "Client Mode" and check robot's ip address from that page

### Ensure that the local computer's IP is correct

Find local ip of the computer using `ifconfig`. Try to find the profile with flags `<UP,BROADCAST,RUNNING,MULTICAST>`, the *inet* corresponding to that profile is the ip of your computer.

```bash
(spot_ros) kavitshah@frerd001:~$ ifconfig
enp69s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500  <---------------------------- This is the profile we are looking at
        inet 192.168.1.6  netmask 255.255.255.0  broadcast 192.168.1.255
        ...
        ...

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        ...
        ...

```

Run the following commands to ensure correct local IP is set
```bash
echo $ROS_IP

# If output of this command does not match local ip from previous step, then update it in bashrc
echo 'export ROS_IP=<your_local_ip>' >> ~/.bashrc


echo $ROS_MASTER_URI

# The output of this command should be of the form - http://<your_local_ip>:11311
# If output of this command does not match local ip from previous step, then update it in bashrc
echo 'export ROS_MASTER_URI=http://<your_local_ip>:11311' >> ~/.bashrc
```
## Getting to the repo
Go to the repository
```bash
cd $SPOT_REPO
```

The code for the demo lies inside the `main` branch.
```bash
git branch
# OR
git rev-parse --abbrev-ref HEAD

# If you are not in the `main` branch, then checkout to the `main` branch
git checkout main
```

## Instructions to Record waypoints (use joystick to move robot around)
- Before running scripts on the robot, waypoints should be recording. These waypoints exist inside file `spot-sim2real/spot_rl_experiments/configs/waypoints.yaml`

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


## Running Instructions
### Perform the following steps
#### Step1. Run executable command **In a New Terminal**
- Run the executable as
    ```bash
    spot_rl_launch_local
    ```
    This command starts 4 tmux sessions\n

    1. roscore
    2. img_publishers
    3. proprioception
    4. tts

- You can run `tmux ls` in the terminal to ensure that all 4 tmux sessions are running.
    You need to ensure that all 4 sessions remain active until 50-70 seconds after running the `spot_rl_launch_local`. If anyone of them dies before 70 seconds, it means there is some issue and you should rerun `spot_rl_launch_local`.

- You should try re-running `spot_rl_launch_local` atleast 2-3 times to see if the issue still persists. Many times roscore takes a while to start due to which other nodes die, re-running can fix this issue.

- **Debugging strategies** for each session if any one of the 4 sessions is dying before 70 seconds
    1. `roscore`
        1. If you see that roscore is dying before 70 seconds, it means that the ip from `ROS_IP` and/or `ROS_MASTER_URI` is not matching local ip if your computer, in other words the local ip of your computer has changed.
        Follow the instructions regarding ip described above to update the local IP.
        2. Try running following command to ensure roscore is up and running.
            ```bash
            rostopic list
            ```
    2. `img_publishers` This is an important node. If this node dies, there could be several reasons.
        1. Roscore is not running. If roscore dies, then img_publishers die too. Fixing `roscore` will resolve this particular root-cause too.
        2. Computer is not connected to robot. You can clarify if this is the case by tring to ping the robot `ping $SPOT_IP`
        3. Code specific failure. To debug this, you should try running the following command in the terminal to find out the root cause
            ```bash
            $CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local
            ```
            Once you have fixed the issue, you need to kill all `img_publishers` nodes that are running, this can be done using `htop`
    3. `proprioception`
        1. This node dies sometimes due to roscore taking quite a while to start up. Re-running `spot_rl_launch_local` should fix this in most cases.
        2. If it still does not get fixed, run this command on a new terminal
            ```bash
            $CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception
            ```
            Once you have fixed the issue, you need to kill all `proprioception` nodes that are running, this can be done using `htop`
    4. `tts` If this node dies, we would be surprised too. In that case, try re-running `spot_rl_launch_local`. If it still fails, don't bother fixing this one :p

        If the output of `rostopic list` looks like the following, then all sessions and ros-nodes are running as expected.
        ```bash
        (spot_ros) kavitshah@frerd001:~$ rostopic list
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

### Step2. Run ROS image visualization
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

### Step3. Reset Home **In a New Terminal**
- This is an important step. Ensure robot is at its start location and sitting, then run the following command in a new terminal
    ```bash
    spot_reset_home
    ```

- The waypoints that were recorded are w.r.t the home location. Since the odometry drifts while robot is moving, **it is necessary to reset home before start of every new run**

### Step4. Emergency Stop
- Since we do not have a physical emergency stop button (like the large red push buttons), we need to run an e-stop node.
    ```bash
    python -m spot_wrapper.estop
    ```

- Keep this window open at all the times, if the robot starts misbehaving you should be able to quickly press `s` or `space_bar` to kill the robot


### Step5. Main Demo code **In a New Terminal**
- In a new window you can now run the code of your choice

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
