# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
import sys

subprocess.check_call(
    f"{sys.executable} -m spot_rl.envs.nav_env -w dock -d", shell=True
)
