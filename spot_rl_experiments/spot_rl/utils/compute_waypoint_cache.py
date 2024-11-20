import json
import os.path as osp
import pickle

import numpy as np
from spot_rl.utils.retrieve_robot_poses_from_cg import ROOT_PATH as CG_ROOT_PATH
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)


# This file computes the view poses and save the cache in the disk
def main():
    cg_json_path = osp.join(CG_ROOT_PATH, "sg_cache", "cfslam_object_relations.json")
    assert osp.exists(cg_json_path), f"{cg_json_path} doesn't exists please check path"
    waypoint_compute_cache_path = osp.join(
        CG_ROOT_PATH, "sg_cache", "map", "waypoint_compute_cache.pkl"
    )

    with open(cg_json_path, "r") as infile:
        cg_relations_data = json.load(infile)
        waypoint_cache = {}
        for cg_relation_node in cg_relations_data:
            for objectkey in ["object1", "object2"]:
                object = cg_relation_node.get(objectkey, None)
                if object is None:
                    continue
                bbox_center = np.array(object.get("bbox_center"))
                bbox_extent = np.array(object.get("bbox_extent"))
                object_tag = [object.get("object_tag")]
                view_poses, category_tag = get_view_poses(
                    bbox_center, bbox_extent, object_tag, True
                )
                x, y = 1, 0
                nav_pts = get_navigation_points(
                    robot_view_pose_data=view_poses,
                    bbox_centers=bbox_center,
                    bbox_extents=bbox_extent,
                    cur_robot_xy=[x, y],
                    goal_xy_yaw_from_cache=None,
                    visualize=False,
                    savefigname="pathplanning.png",
                )
                bbox_center_str = [str(v) for v in bbox_center.tolist()]
                bbox_extent_str = [str(v) for v in bbox_extent.tolist()]
                unique_key = ",".join(bbox_center_str + bbox_extent_str)
                final_waypoint = nav_pts[-1]
                final_waypoint[-1] = np.rad2deg(final_waypoint[-1])
                waypoint_cache[unique_key] = (final_waypoint, category_tag)
                print(f"{object_tag} : final waypoint {final_waypoint}")
        with open(waypoint_compute_cache_path, "wb") as outfile:
            pickle.dump(waypoint_cache, outfile)


if __name__ == "__main__":
    main()
