import argparse
import json


def parse_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return None


def format_value(key, value):
    if isinstance(value, (int, float)):
        value = round(value, 3)
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Parse a JSON file")
    parser.add_argument("file", help="Path to the JSON file to parse")
    args = parser.parse_args()

    parsed_data = parse_json_file(args.file)

    results_dict = {
        "overall_success": " ",
        "total_time": parsed_data["total_time"],
        "total_num_steps": parsed_data["total_steps"],
        "0_nav_success": " ",
        "0_nav_time_taken": " ",
        "0_nav_num_steps": " ",
        "0_nav_distance_travelled": " ",
        "0_nav_distance_to_goal_linear": " ",
        "0_nav_distance_to_goal_angular": " ",
        "1_pick_success": " ",
        "1_pick_time_taken": " ",
        "1_pick_num_steps": " ",
        "2_nav_success": " ",
        "2_nav_time_taken": " ",
        "2_nav_num_steps": " ",
        "2_nav_distance_travelled": " ",
        "2_nav_distance_to_goal_linear": " ",
        "2_nav_distance_to_goal_angular": " ",
        "3_place_success": " ",
        "3_place_time_taken": " ",
        "3_place_num_steps": " ",
    }

    success_ctr = 0
    for idx, action in enumerate(parsed_data["actions"]):
        for action_type in action.keys():
            for k, v in action[action_type].items():
                if (
                    action_type == "pick" or action_type == "place"
                ) and k == "distance_travelled":
                    pass
                elif k == "robot_trajectory":
                    pass
                elif k == "distance_to_goal":
                    for bk, bv in action[action_type][k].items():
                        match_str = f"{idx}_{action_type}_{k}_{bk}"
                        results_dict[match_str] = bv
                else:
                    match_str = f"{idx}_{action_type}_{k}"
                    results_dict[match_str] = v
                    if "success" in k and v == True:
                        success_ctr += 1
    results_dict["overall_success"] = success_ctr / 4

    # Print keys
    print("\t".join(results_dict.keys()))

    # Print values
    print("\t".join(format_value(key, value) for key, value in results_dict.items()))


if __name__ == "__main__":
    main()
