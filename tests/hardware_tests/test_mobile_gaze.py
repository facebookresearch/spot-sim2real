import numpy as np
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    from spot_rl.utils.utils import map_user_input_to_boolean

    # Pick Targets from Aria with +- 10 degree angle change
    pick_targets = {
        "kitchen": (
            3.8482142244527835,
            -3.4519528625906206,
            np.deg2rad(-89.14307672622927),
        ),
        "kitchen_-10": (
            3.8482142244527835,
            -3.4519528625906206,
            np.deg2rad(-89.14307672622927 - 10.0),
        ),
        "kitchen_+10": (
            3.8482142244527835,
            -3.4519528625906206,
            np.deg2rad(-89.14307672622927 + 10.0),
        ),
        "table": (4.979398852803741, 3.535594946585519, -0.008869974097951427),
        "table_+10": (
            4.979398852803741,
            3.535594946585519,
            np.deg2rad(np.rad2deg(-0.008869974097951427) + 15.0),
        ),
        "table_-10": (
            4.979398852803741,
            3.535594946585519,
            np.deg2rad(np.rad2deg(-0.008869974097951427) - 15.0),
        ),
    }
    contnue = True
    i: int = 0
    n: int = len(pick_targets)
    pick_targets_keys = list(pick_targets.keys())
    while contnue and i < 0:
        pick_from = pick_targets_keys[i]
        i += 1
        spotskillmanager = SpotSkillManager(use_mobile_pick=True)
        spotskillmanager.nav(*pick_targets[pick_from])
        x = input(f"Press Enter to continue to mobile gaze from {pick_from}")
        spotskillmanager.pick("cereal_box")
        spotskillmanager.spot.open_gripper()
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

    # Navigate to dock and shutdown
    spotskillmanager.dock()
