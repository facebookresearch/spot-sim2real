import os.path as osp
import subprocess

this_dir = osp.dirname(osp.abspath(__file__))
local_parallel_inference = osp.join(this_dir, "img_publishers.py")

cmds = [
    f"python {local_parallel_inference}",
    f"python {local_parallel_inference} --nav",
]

processes = [subprocess.Popen(cmd, shell=True) for cmd in cmds]
try:
    while any([p.poll() is None for p in processes]):
        pass
finally:
    [p.kill() for p in processes]
