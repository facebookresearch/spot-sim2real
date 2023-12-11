## :eyeglasses: Exporting habitat-lab policies/networks in torchscript to load into spotsim2real environment without libraries and environment conflicts 
You might want to train new policies or networks in [habitat-lab](https://github.com/facebookresearch/habitat-lab). However, habitat-lab conda environment packages and spot-ros (used for spot-sim2real) environment packages might create version incompatabilities.
Thus, we export a model in intermediate representation (IR) using torchscript module provided in Pytorch. Disentangling the deployment and development environment of a model, provides freedom to the model developer. We made a [conversion script](https://github.com/facebookresearch/spot-sim2real/blob/d200deef1ca3f4608cb3f84b43672bda63a3ce0b/spot_rl_experiments/utils/hab3_policy_conversion.py) to convert mobile-gaze policy that was trained in new version of habitat-lab to torchscript model.

In general, these are the steps you can follow for conversion.

1. Load the pytorch model with class files, <b>transfer the model on cuda</b>
2. Pass some random input tensor to the model and trace it's forward pass using ```torch.jit.trace```, usage example can be found [here](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
3. Save the traced model as modelXX.torchscript, replace modelXX with desired name
4. To load the model in spotsim2real use ```torch.jit.load(path/to/saved/torchscript/model, map_location="cuda:0/cpu")```

## Solving Pytorch & CUDA error in recent Habitat-lab version setup
We encountered cuda error when setting up the recent habitat-lab version, it installs a recent version (2.2.1) of pytorch and CUDA 11.8. However, the hardware driver was older than 11.8 thus ```torch.cuda.is_available()``` was ```False``` and showing driver old error. 
To fix that first uninstall pytorch using ```pip uninstall pytorch torchvision torchaudio ``` in your habitat-lab conda env then run the following ```conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch``` in same habitat-lab env (this is the pytorch and cuda version we use for spot-sim2real/spot-ros env)