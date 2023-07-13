# Common Issues

The following are some of the most commonly seen issues

<!-- AFTER BD_SPOT_WRAPPER
# Take a pause, run something on the robot -->



### ASC is installed. Run basic ASC commands.

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
Then open the `inceptionresnetv2.py` in Vim
```bash
vi /home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/pretrainedmodels/models/inceptionresnetv2.py
```

Once VIM opens, run following commands
1. ESC then `:set number` then ENTER      				 <--- This will enable line numbers
2. ESC then `:221` then ENTER              				 <--- This will bring you to the line where correction needs to be made
3. ESC then `i` then `# ` then ESC         				 <--- This will get you into INSERT mode, comment the if condition on this line
4. ESC then `:222` then ENTER then `i` then BACKSPACE (x1) then ESC	 <--- This will bring you to the following line, remove the indentation for that line
5. ESC then 'wq' then ENTER						 <--- This will save and exit

This should resolve the issue

## If you face an issue saying "The detected CUDA version (12.1) mismatches the version that was used to compile" :
```
    PyTorch:
## How to resolve this
× python setup.py develop did not run successfully.
│ exit code: 1
╰─> [55 lines of output]
    running develop
    running egg_info
    creating detectron2.egg-info
    writing detectron2.egg-info/PKG-INFO
    writing dependency_links to detectron2.egg-info/dependency_links.txt
    writing requirements to detectron2.egg-info/requires.txt
    writing top-level names to detectron2.egg-info/top_level.txt
    writing manifest file 'detectron2.egg-info/SOURCES.txt'
    reading manifest file 'detectron2.egg-info/SOURCES.txt'
    adding license file 'LICENSE'
    writing manifest file 'detectron2.egg-info/SOURCES.txt'
    running build_ext
    /home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/command/easy_install.py:144: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
      warnings.warn(
    /home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
      warnings.warn(
    Traceback (most recent call last):
      File "<string>", line 2, in <module>
      File "<pip-setuptools-caller>", line 34, in <module>
      File "/home/test/test_fair/spot-sim2real/mask_rcnn_detectron2/detectron2/setup.py", line 151, in <module>
        setup(
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/__init__.py", line 87, in setup
        return distutils.core.setup(**attrs)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 148, in setup
        return run_commands(dist)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 163, in run_commands
        dist.run_commands()
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 967, in run_commands
        self.run_command(cmd)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/dist.py", line 1214, in run_command
        super().run_command(command)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 986, in run_command
        cmd_obj.run()
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/command/develop.py", line 34, in run
        self.install_for_development()
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/command/develop.py", line 114, in install_for_development
        self.run_command('build_ext')
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 313, in run_command
        self.distribution.run_command(command)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/dist.py", line 1214, in run_command
        super().run_command(command)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 986, in run_command
        cmd_obj.run()
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 79, in run
        _build_ext.run(self)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 339, in run
        self.build_extensions()
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 434, in build_extensions
        self._check_cuda_version(compiler_name, compiler_version)
      File "/home/test/miniconda3/envs/spot_ros/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 812, in _check_cuda_version
        raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
    RuntimeError:
    The detected CUDA version (12.1) mismatches the version that was used to compile
    PyTorch (11.3). Please make sure to use the same CUDA versions.

    [end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
```
Root cause: Your system has an nvidia-driver (with CUDA=12.1 in my case). But we used a different CUDA(=11.3) to compile pytorch. Installation of detectron2 python package does not like this and will complain.

Tried solution : Delete all nvidia drivers from system (root) and install a new one. It would be better to use an **11.x** driver. This is a solution that we use, but you can use a different method.
