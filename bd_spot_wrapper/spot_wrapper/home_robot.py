# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from spot_wrapper.spot import Spot

if __name__ == "__main__":
    spot = Spot("NavPoseMonitor")
    spot.home_robot(write_to_file=True)
