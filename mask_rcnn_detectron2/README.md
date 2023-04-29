# mask_rcnn_detectron2
## Installation
Run the following in your Python 3 conda env to install detectron2
```
git clone https://github.com/facebookresearch/detectron2.git
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install opencv-python
python -m pip install -e detectron2
```
If you get issues about CUDA versions, you can determine your system's CUDA version using `nvidia-smi` and looking at the top right corner

