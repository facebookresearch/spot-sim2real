# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import json
import os.path as osp
import signal
import sys

import yaml
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager

controller_dict = {
    "nav": "nav_controller",
    "pick": "gaze_controller",
    "place": "place_controller",
}
config_dict = {"nav": "nav_config", "pick": "pick_config"}
episode_name, episode_log, final_success, total_time, total_steps = (
    None,
    None,
    None,
    None,
    None,
)


def save_logs_as_json(arg1=None, arg2=None):
    global final_success, total_time, total_steps, episode_log, episode_name
    if episode_name is not None and episode_log is not None:
        episode_log["total_time"] = total_time
        episode_log["final_success"] = final_success
        episode_log["total_steps"] = total_steps
        file_path = f"logs/integration_test/8-15-24/{episode_name}.json"
        with open(file_path, "w") as file:
            json.dump(episode_log, file, indent=4)
            print(f"Saved log: {file_path}")


atexit.register(save_logs_as_json)
# Register the signal handler for termination signals
signal.signal(signal.SIGINT, save_logs_as_json)
signal.signal(signal.SIGTERM, save_logs_as_json)


def parse_yaml(file_path):
    global final_success, total_time, total_steps, episode_log, episode_name
    with open(file_path, "r") as file:
        episodes = yaml.safe_load(file)

    for episode, details in episodes.items():
        print(f"Executing {episode}")
        # Instantiate ActionTaker with provided args
        action_taker_args = details.pop("spotskillmanagerargs")
        action_taker = SpotSkillManager(**action_taker_args)
        episode_name, episode_log = episode, {"actions": []}
        actions = details["actions"]
        final_success, total_time, total_steps = True, 0, 0
        terminate_episode = False
        for action_dict in actions:
            if terminate_episode:
                break
            for action, args in action_dict.items():
                method = getattr(action_taker, action, None)
                if method:
                    # breakpoint()
                    status, _ = method(*args)
                    controller = getattr(action_taker, controller_dict[action], None)
                    if controller:
                        skill_log = controller.skill_result_log
                        final_success = final_success and skill_log["success"]
                        # check if episode failed. If so, terminate the episode
                        max_episode_steps = getattr(
                            action_taker, f"{action}_config"
                        ).MAX_EPISODE_STEPS
                        max_episode_steps = (
                            75 if action == "place" else max_episode_steps
                        )
                        if (
                            skill_log["num_steps"] >= max_episode_steps
                            and not skill_log["success"]
                        ):
                            terminate_episode = True
                        print("status: ", status, skill_log["success"])
                        print(skill_log["time_taken"], total_time)
                        total_time += skill_log["time_taken"]
                        total_steps += skill_log.get("num_steps", 0)
                        episode_log["actions"].append({f"{action}": skill_log})
                    else:
                        print(
                            f"{controller_dict[action]} not found in SpotSkillManager"
                        )
                else:
                    print(f"Action {action} not found in SpotSkillManager")
        save_logs_as_json()
        contnu = map_user_input_to_boolean("Should I dock y/n ?")
        if contnu:
            action_taker.dock()
        else:
            # spotskillmanager.spot. reset arm
            action_taker.spot.sit()
        contnu = map_user_input_to_boolean(
            f"Finished {episode}. Press y/n to continue to the next episode..."
        )
        if not contnu:
            break


if __name__ == "__main__":

    current_script_path = osp.dirname(osp.abspath(__file__))
    yaml_file_root = osp.join(current_script_path, "episode_configs")
    if len(sys.argv) != 2:
        print("Usage: python run_episode.py episode1.yaml")
        sys.exit(1)
    yaml_file_path = osp.join(yaml_file_root, sys.argv[1])
    assert osp.exists(
        yaml_file_path
    ), f"{yaml_file_path} doesn't exists please recheck the path"
    continu = map_user_input_to_boolean(
        f"Using the yaml file {yaml_file_path} should I continue"
    )
    if continu:
        parse_yaml(yaml_file_path)
