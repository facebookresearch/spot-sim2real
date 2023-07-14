# Common Issues

The following are some of the most commonly seen issues

<!-- ### ASC is installed. Run basic ASC commands. -->
<!-- AFTER BD_SPOT_WRAPPER
# Take a pause, run something on the robot -->


## If you face an issue saying "Block8 has no module relu":

Issue:
```
(spot_ros) test@robodev111:~/spot-sim2real$ $CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local

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

## Debugging strategies for `spot_rl_launch_local` if any one of the 4 sessions are dying before 70 seconds
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
  3. `proprioception`
      1. This node dies sometimes due to roscore taking quite a while to start up. Re-running `spot_rl_launch_local` should fix this in most cases.
      2. If it still does not get fixed, run this command on a new terminal
          ```bash
          $CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception
          ```
          Once you have fixed the issue, you need to kill all `proprioception` nodes that are running before running `spot_rl_launch_local`, this can be done using `htop`
  4. `tts` If this node dies, we would be surprised too. In that case, try re-running `spot_rl_launch_local`. If it still fails, don't bother fixing this one :p


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