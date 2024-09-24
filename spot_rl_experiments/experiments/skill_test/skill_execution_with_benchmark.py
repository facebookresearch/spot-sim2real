from abc import ABC

import magnum as mn
import numpy as np
from spot_wrapper.utils import get_angle_between_two_vectors


class SpotSkillExecuterWithBenchmark(ABC):

    """This class runs spot skills & gives average benchmark"""

    def __init__(self):
        self.spotskillmanager = None
        self._cur_skill_name_input = None

    def compute_metrics(self, traj, target_point, name_key="_nav1"):
        """Compute the metrics"""
        num_steps = len(traj)
        final_point = np.array(traj[-1]["pose"][0:2])
        distance = np.linalg.norm(target_point - final_point)
        # Compute the angle
        vector_robot_to_target = target_point - final_point
        vector_robot_to_target = vector_robot_to_target / np.linalg.norm(
            vector_robot_to_target
        )
        vector_forward_robot = np.array(
            self.spotskillmanager.get_env().curr_transform.transform_vector(
                mn.Vector3(1, 0, 0)
            )
        )[[0, 1]]
        vector_forward_robot = vector_forward_robot / np.linalg.norm(
            vector_forward_robot
        )
        dot_product_facing_target = abs(
            np.dot(vector_robot_to_target, vector_forward_robot)
        )
        angle_facing_target = abs(
            get_angle_between_two_vectors(vector_robot_to_target, vector_forward_robot)
        )
        return {
            f"num_steps{name_key}": num_steps,
            f"distance{name_key}": distance,
            f"dot_product_facing_target_{name_key}": dot_product_facing_target,
            f"angle_facing_target_{name_key}": angle_facing_target,
        }

    def benchmark(self):
        pass
