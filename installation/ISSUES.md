# Common Issues

The following are some of the most commonly seen issues

## If you face an issue saying "The detected CUDA version (12.1) mismatches the version that was used to compile" :
```bash
RuntimeError:
    The detected CUDA version (12.1) mismatches the version that was used to compile
    PyTorch (11.3). Please make sure to use the same CUDA versions.

    [end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
```
Root cause: Your system has an nvidia-driver (with CUDA=12.1 in my case). But we used a different CUDA(=11.3) to compile pytorch. Installation of detectron2 python package does not like this and will complain.

Tried solution : Delete all nvidia drivers from system (root) and install a new one. It would be better to use an **11.x** driver. This is a solution that we use, but you can use a different method.

## How to find IP address of local computer
Find local ip of the computer using `ifconfig`. Try to find the profile with flags `<UP,BROADCAST,RUNNING,MULTICAST>`, the *inet* corresponding to that profile is the ip of your computer.

```bash
(spot_ros) user@linux-machine:~$ ifconfig
enp69s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500  <---------------------------- This is the profile we are looking at
        inet 192.168.1.6  netmask 255.255.255.0  broadcast 192.168.1.255
        ...
        ...

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        ...
        ...

```

## Issues with `spot_rl_launch_local`:

 If you are seeing the following error while running `spot_rl_launch_local` then you are missing `tmux` package.
  ```bash
  (spot_ros) user@linux-machine:~/spot-sim2real$ spot_rl_launch_local 
  Killing all tmux sessions...
  /path/to/local_only.sh: line 2: tmux: command not found
  /path/to/local_only.sh: line 3: tmux: command not found
  /path/to/local_only.sh: line 4: tmux: command not found
  /path/to/local_only.sh: line 5: tmux: command not found
  Starting roscore tmux...
  /path/to/local_only.sh: line 8: tmux: command not found
  Starting other tmux nodes..
  /path/to/local_only.sh: line 10: tmux: command not found
  /path/to/local_only.sh: line 11: tmux: command not found
  /path/to/local_only.sh: line 12: tmux: command not found
  /path/to/local_only.sh: line 14: tmux: command not found
  ```
  Easy fix :
  1. Install tmux using `sudo apt install tmux`

### Debugging strategies for `spot_rl_launch_local` if any one of the 4 sessions are dying before 70 seconds
  1. `roscore`
      1. If you see that roscore is dying before 70 seconds, it means that the ip from `ROS_IP`/`ROS_HOSTNAME` and/or `ROS_MASTER_URI` is not matching local ip if your computer, in other words the local ip of your computer has changed.
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
          Once you have fixed the issue, you need to kill all `img_publishers` nodes that are running `spot_rl_launch_local`, this can be done using `htop`
      4. If failure is due to missing `waypoints.yaml` file, then [follow these steps to generate the `waypoints.yaml` file](/README.md#video_game-instructions-to-record-waypoints-use-joystick-to-move-robot-around)
      5. If you face an issue regarding `"Block8 has no module relu"`, [follow these steps described in ISSUES.md](/installation/ISSUES.md#if-you-face-an-issue-saying-block8-has-no-module-relu)
      6. If facing this bug -- `"KeyError: 'Neither weights/CUTOUT_WT_True_SD_200_ckpt.99.pvp.pth nor /home/$USER/spot-sim2real/spot_rl_experiments/weights/CUTOUT_WT_True_SD_200_ckpt.99.pvp.pth exist!'"` [Follow these steps](https://github.com/facebookresearch/spot-sim2real/blob/main/installation/SETUP_INSTRUCTIONS.md#setup-spot_rl_experiments)
  3. `proprioception`
      1. This node dies sometimes due to roscore taking quite a while to start up. Re-running `spot_rl_launch_local` should fix this in most cases.
      2. If it still does not get fixed, run this command on a new terminal
          ```bash
          $CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception
          ```
          Once you have fixed the issue, you need to kill all `proprioception` nodes that are running before running `spot_rl_launch_local`, this can be done using `htop`
      3. If failure is due to missing `waypoints.yaml` file, then [follow these steps to generate the `waypoints.yaml` file](/README.md#video_game-instructions-to-record-waypoints-use-joystick-to-move-robot-around)
  4. `tts` If this node dies, we would be surprised too. In that case, try re-running `spot_rl_launch_local`. If it still fails, don't bother fixing this one :p


## If you face an issue saying "Block8 has no module relu":

Issue:
```
(spot_ros) user@linux-machine:~/spot-sim2real$ $CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local

RuntimeError:
Module 'Block8' has no attribute 'relu' :
  File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/pretrainedmodels/models/inceptionresnetv2.py", line 231
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
                  ~~~~~~~~~ <--- HERE
        return out

```
Soln : Then open the file `inceptionresnetv2.py` at `/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/pretrainedmodels/models/inceptionresnetv2.py` in editor of your choice and make it look like the following.

Before the change, the code should look like follows:
```bash
204 class Block8(nn.Module):
205 
206     def __init__(self, scale=1.0, noReLU=False):
207         super(Block8, self).__init__()
208 
209         self.scale = scale
210         self.noReLU = noReLU
211 
212         self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
213 
214         self.branch1 = nn.Sequential(
215             BasicConv2d(2080, 192, kernel_size=1, stride=1),
216             BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
217             BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
218         )
219 
220         self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
221         if not self.noReLU:
222           self.relu = nn.ReLU(inplace=False)
223
```

After the change, the code should look like follows:
```bash
204 class Block8(nn.Module):
205 
206     def __init__(self, scale=1.0, noReLU=False):
207         super(Block8, self).__init__()
208 
209         self.scale = scale
210         self.noReLU = noReLU
211 
212         self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
213 
214         self.branch1 = nn.Sequential(
215             BasicConv2d(2080, 192, kernel_size=1, stride=1),
216             BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
217             BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
218         )
219 
220         self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
221         # if not self.noReLU:
222         self.relu = nn.ReLU(inplace=False)
223 
```

**Please only change code on line `221` & `222` inside `inceptionresnetv2.py`**

  ## Issues while downloading weights
  If you see issues like
  ```bash
  Archive:  spot-sim2real-data/weight/weights.zip   End-of-central-directory signature not found.  Either this file is not   a zipfile, or it constitutes one disk of a multi-part archive.  In the   latter case the central directory and zipfile comment will be found on   the last disk(s) of this archive. unzip:  cannot find zipfile directory in one of spot-sim2real-data/weight/weights.zip or         spot-sim2real-data/weight/weights.zip.zip, and cannot find spot-sim2real-data/weight/weights.zip.ZIP, period
  ```

  OR
  ```bash
  Ick! 0x73726576
  ```

  It means git-lfs has not been installed properly on your system. You can install git-lfs by the following commands
  ```bash
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
  git-lfs install
  ```

  ## Issues while running setup.py for Detectron2
  
  If you face the following issues while running setup.py for Detectron2, it indicates either gcc & g++ are missing from your system or are not linked properly.

  ### Missing compilers

  ```bash
error: command 'gcc' failed: No such file or directory
    [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pycocotools
Failed to build pycocotools
ERROR: Could not build wheels for pycocotools, which is required to install pyproject.toml-based projects
```

 Easy fix : 
 1. Remove gcc & g++ versions if they exist,
 2. Install gcc -> `sudo apt-get install gcc`
 3. Install g++ -> `sudo apt-get install g++`
 4. Then retry running the same setup.py command

### Improperly linked compilers

 However if you see an error along the lines of 
 ```bash
  /path/to/spot_ros/env/compiler_compat/ld: skipping incompatible /lib64/libm.so.6 when searching for /lib64/libm.so.6
 ```

Then this may mean your python is accessing different versions of `ld` and `gcc/g++` creating a linking problem. We need to ensure `ld` and compiler binaries are being used from the same place, i.e. system or from within the environment. First run the following command: 

```bash
which ld
```

If the `ld` being linked is from within the environment, then make sure `which gcc` and `which g++` return the system binaries, i.e. `/usr/bin/gcc` and `/usr/bin/g++`. If all the conditions are met then you are facing an improper linking issue. Easy fix would be to install C/C++ compilers for your environment through: 
```bash
mamba install -c conda-forge cxx-xompiler
```

Rerun `pip install -e detectron2` from the correct place in the repository (`repo_root/third_party/mask_rcnn_detector2`).

  ## Issues while running setup.py for Habitat-lab
  If you face the following issues while running setup.py for Habitat-lab, it is because a running the setup script tried to install a newer version of `tensorflow` which depends on newer version of `numpy` which conflicts with already existing `numpy` version in our v-env.
  ```bash
  Installed /home/user/miniconda3/envs/spot_ros/lib/python3.8/site-packages/tensorflow-2.13.0-py3.8-linux-x86_64.egg
  Searching for tensorflow-estimator<2.14,>=2.13.0
  Reading https://pypi.org/simple/tensorflow-estimator/
  Downloading https://files.pythonhosted.org/packages/72/5c/c318268d96791c6222ad7df1651bbd1b2409139afeb6f468c0f327177016/tensorflow_estimator-2.13.0-py2.py3-none-any.whl#sha256=6f868284eaa654ae3aa7cacdbef2175d0909df9fcf11374f5166f8bf475952aa
  Best match: tensorflow-estimator 2.13.0
  Processing tensorflow_estimator-2.13.0-py2.py3-none-any.whl
  Installing tensorflow_estimator-2.13.0-py2.py3-none-any.whl to /home/user/miniconda3/envs/spot_ros/lib/python3.8/site-packages
  Adding tensorflow-estimator 2.13.0 to easy-install.pth file

  Installed /home/user/miniconda3/envs/spot_ros/lib/python3.8/site-packages/tensorflow_estimator-2.13.0-py3.8.egg
  Searching for tensorboard<2.14,>=2.13
  Reading https://pypi.org/simple/tensorboard/
  Downloading https://files.pythonhosted.org/packages/67/f2/e8be5599634ff063fa2c59b7b51636815909d5140a26df9f02ce5d99b81a/tensorboard-2.13.0-py3-none-any.whl#sha256=ab69961ebddbddc83f5fa2ff9233572bdad5b883778c35e4fe94bf1798bd8481
  Best match: tensorboard 2.13.0
  Processing tensorboard-2.13.0-py3-none-any.whl
  Installing tensorboard-2.13.0-py3-none-any.whl to /home/user/miniconda3/envs/spot_ros/lib/python3.8/site-packages
  Adding tensorboard 2.13.0 to easy-install.pth file
  Installing tensorboard script to /home/user/miniconda3/envs/spot_ros/bin

  Installed /home/user/miniconda3/envs/spot_ros/lib/python3.8/site-packages/tensorboard-2.13.0-py3.8.egg
  error: numpy 1.21.6 is installed but numpy<=1.24.3,>=1.22 is required by {'tensorflow'}
  ```

  Easy fix :
  1. Clear all caches from mamba - `mamba clean -f -a`
  2. Install spefic tensorflow version - `pip install tensorflow==2.9.0`
  3. Then retry running the same setup.py command
