import os.path as osp
import subprocess

this_dir = osp.dirname(osp.abspath(__file__))
depth_node_script = osp.join(this_dir, "depth_filter_node.py")
mask_rcnn_node_script = osp.join(this_dir, "mask_rcnn_utils.py")

cmds = [
    f"python {depth_node_script}",
    f"python {depth_node_script} --head",
    f"python {mask_rcnn_node_script}",
]

processes = [subprocess.Popen(cmd, shell=True) for cmd in cmds]
try:
    while any([p.poll() is None for p in processes]):
        pass
finally:
    [p.kill() for p in processes]
