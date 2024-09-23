# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import rospy

if __name__ == "__main__":
    # Put the bbox center/extent info here
    bbox_extent = [1.8, 0.7, 0.4]
    bbox_center = [2.1, 1.5, 0.2]
    bbox_info = (
        [str(v) for v in bbox_center]
        + [str(v) for v in bbox_extent]
        + ["pillow", "couch", "sofa"]
    )
    skill_input = ";".join(bbox_info)
    rospy.set_param(
        "/skill_name_input",
        f"{str(time.time())},nav_path_planning_with_view_poses,{skill_input}",
    )
    print("Listen to the skill execution...")
    while True:
        msg = rospy.get_param(
            "/skill_name_suc_msg", f"{str(time.time())},None,None,None"
        )
        suc_fail = msg.split(",")[1]
        if suc_fail != "None":
            print(msg)
            break
    print("Done!")
