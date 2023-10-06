from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import map_user_input_to_boolean


if __name__ == "__main__":
    
    #print(spotskillmanager.pick_config)
    # Nav-Pick-Nav-Place sequence 1
    run_the_loop = True
    while run_the_loop:
        spotskillmanager = SpotSkillManager()
        spotskillmanager.nav_config.SUCCESS_DISTANCE = 0.10
        spotskillmanager.place_config.SUCCESS_DISTANCE = 0.10
        spotskillmanager.pick_config.SUCCESS_DISTANCE = 0.10
        spotskillmanager.pick_config.MAX_EPISODE_STEPS = 350
        max_steps = 350
        #spotskillmanager.nav("starting_point")
        spotskillmanager.nav("pick_table_01")
        pick_stats = spotskillmanager.pick("ball")
        print(pick_stats)
        #spotskillmanager.nav("starting_point")
        spotskillmanager.nav("pick_table_01")
        
        run_the_loop = map_user_input_to_boolean(
                    "Do you want to continue to next test or dock & exit ?"
                )
        if not run_the_loop:
            spotskillmanager.dock()