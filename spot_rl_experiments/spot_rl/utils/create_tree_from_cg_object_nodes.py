# mypy: ignore-errors
import gzip
import os.path as osp
import pickle
from pprint import pp

import numpy as np
import quads
import yaml
from spot_rl.utils.construct_configs import load_config
from spot_rl.utils.path_planning import get_xyzxyz

PATH_TO_CONFIG_FILE = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
    "configs",
    "cg_config.yaml",
)
assert osp.exists(PATH_TO_CONFIG_FILE), "cg_config.yaml wasn't found"
cg_config = load_config(PATH_TO_CONFIG_FILE)

ROOT_PATH = cg_config["CG_ROOT_PATH"]

PATH_TO_CACHE_FILE = osp.join(
    ROOT_PATH, "sg_cache", "map", "scene_map_cfslam_pruned.pkl.gz"
)


def populate_quad_tree():
    if osp.exists(PATH_TO_CACHE_FILE):
        with gzip.open(PATH_TO_CACHE_FILE, "rb") as f:
            cache_file = pickle.load(f)
            max_x, max_y = -np.inf, -np.inf
            min_x, min_y = np.inf, np.inf
            for i, object_item in enumerate(cache_file):
                boxMin, boxMax = get_xyzxyz(
                    np.array(object_item["caption_dict"]["bbox_center"]),
                    np.array(object_item["caption_dict"]["bbox_extent"]),
                )
                boxMin, boxMax = (
                    min(boxMin[0], boxMax[0]),
                    min(boxMin[1], boxMax[1]),
                ), (max(boxMin[0], boxMax[0]), max(boxMin[1], boxMax[1]))
                min_x = min(boxMin[0], min_x)
                min_y = min(boxMin[1], min_y)
                max_x = max(boxMax[0], max_x)
                max_y = max(boxMax[1], max_y)
            width = max_y - min_y + 1 + 10
            height = max_x - min_x + 1 + 10
            print(f"Y extent {width}, X extent {height}")
            tree = quads.QuadTree((0, 0), width, height)
            tree.insert(quads.Point(0.0, 0.0, data="Dock"))

            data_dict = {}
            error = 0
            for i, object_item in enumerate(cache_file):
                object_tag = object_item["caption_dict"]["response"]["object_tag"]
                id = object_item["caption_dict"]["id"]
                boxcenter = object_item["caption_dict"]["bbox_center"][:2]

                x, y = -1 * boxcenter[1], boxcenter[0]

                caption_data = object_item["caption_dict"]
                data = {
                    "id": caption_data["id"],
                    "object_tag": caption_data["response"]["object_tag"],
                    "bbox_extent": caption_data["bbox_extent"],
                    "bbox_center": caption_data["bbox_center"],
                }
                data_dict[f"{x:.1f}, {y:.1f}"] = data
                try:
                    assert tree.insert((x, y), data=f"{id}_{object_tag}")
                except Exception:
                    error += 1
            print(f"Number of errors while creating the tree {error}")
            # quads.visualize(tree)
            return tree, data_dict


def query_quad_tree(x_incg, y_incg, tree: quads.QuadTree, data_dic):
    xin_query = -1 * y_incg
    yin_query = x_incg
    region = 1
    bb = quads.BoundingBox(
        min_x=xin_query - region,
        min_y=yin_query - region,
        max_x=xin_query + region,
        max_y=yin_query + region,
    )
    points = tree.within_bb(bb)

    converted_points = []

    for point in points:
        key = f"{point.x:.1f}, {point.y:.1f}"
        x_in_cg = point.y
        y_in_cg = -1 * point.x
        pp(data_dic.get(key, "NotFound"))
        converted_points.append((x_in_cg, y_in_cg, data_dic.get(key, "NotFound")))
    return converted_points


if __name__ == "__main__":
    cache_file_for_quad_tree = "quad_tree.pkl"
    if osp.exists(cache_file_for_quad_tree):
        with open(cache_file_for_quad_tree, "rb") as f:
            tree, data = pickle.load(f)
    else:
        tree, data = populate_quad_tree()
        with open(cache_file_for_quad_tree, "wb") as f:
            pickle.dump((tree, data), f)
    # This script is used to create a quad tree from the CG objects to load the objects in the
    # graph.
    nodes = query_quad_tree(3.2, -1.2, tree, data)
