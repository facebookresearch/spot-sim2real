# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

if __name__ == "__main__":
    # Put the bbox center/extent info here
    bbox_center = [8.2, 6.0, 0.1]
    bbox_extent = [1.3, 1.0, 0.8]
    bbox_info = [str(v) for v in bbox_center] + [str(v) for v in bbox_extent]
    skill_input = ";".join(bbox_info)
    rospy.set_param(
        "/skill_name_input", f"nav_path_planning_with_view_poses,{skill_input}"
    )
    print("Listen to the skill execution...")
    while True:
        msg = rospy.get_param("/skill_name_suc_msg", "None,None,None")
        suc_fail = msg.split(",")[1]
        if suc_fail != "None":
            print(msg)
            break
    print("Done!")
