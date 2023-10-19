from spot_rl.envs.gaze_env import construct_config_for_gaze
from spot_rl.envs.nav_env import construct_config_for_nav
from spot_rl.envs.place_env import construct_config_for_place
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import map_user_input_to_boolean

if __name__ == "__main__":
    run_the_loop = True  # Don't break the test loop
    is_in_position = False  # is it before the receptacle
    # Construct all three configs
    nav_config, pick_config, place_config = (
        construct_config_for_nav(),
        construct_config_for_gaze(),
        construct_config_for_place(),
    )
    # Modify any values in the config
    nav_config.SUCCESS_DISTANCE = 0.10
    place_config.SUCCESS_DISTANCE = 0.10
    pick_config.SUCCESS_DISTANCE = 0.10
    pick_config.MAX_EPISODE_STEPS = 350

    while run_the_loop:
        # Send the modified configs in SpotSkillManager
        spotskillmanager = SpotSkillManager(nav_config, pick_config, place_config)
        if not is_in_position:
            spotskillmanager.nav("pick_table_05_45")
        # Reset Arm before gazing starts
        spotskillmanager.get_env().reset_arm()
        spotskillmanager.gaze_controller.reset_env_and_policy("ball")
        pick_stats = spotskillmanager.pick("ball")
        # print(pick_stats)
        spotskillmanager.get_env().reset_arm()
        spotskillmanager.gaze_controller.reset_env_and_policy("ball")
        is_in_position = False
        # Navigate to Test Receotacle
        spotskillmanager.nav("pick_table_05_45")
        is_in_position = True

        run_the_loop = map_user_input_to_boolean(
            "Do you want to continue to next test or dock & exit ?"
        )
        if not run_the_loop:
            spotskillmanager.dock()
