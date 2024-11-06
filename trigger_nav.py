import time

import rospy

obj_tag = "nightstand on left"
bbox_extent = [0.8, 0.6, 0.3]
bbox_center = [10.4, 2.4, 0.0]


bbox_center = [str(v) for v in bbox_center]
bbox_center = ";".join(bbox_center)


bbox_extent = [str(v) for v in bbox_extent]
bbox_extent = ";".join(bbox_extent)

skill_param_processed = bbox_center + ";" + bbox_extent + ";" + obj_tag + "|"


rospy.set_param(
    "skill_name_input",
    f"{str(time.time())},nav_path_planning_with_view_poses,{skill_param_processed}",
)
