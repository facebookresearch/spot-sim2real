import argparse
import csv
import json
import os


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


def export_to_csv(results_dict, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header (keys of the dictionary)
        writer.writerow(results_dict.keys())
        # Write data (values of the dictionary)
        writer.writerow(results_dict.values())


def process_json_file(input_file, output_file):
    parsed_data = parse_json_file(input_file)
    if parsed_data is None:
        return

    results_dict = {
        "overall_success": " ",
        "total_time": parsed_data.get("total_time", " "),
        "total_num_steps": parsed_data.get("total_steps", " "),
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
    for idx, action in enumerate(parsed_data.get("actions", [])):
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
                    if "success" in k and v is True:
                        success_ctr += 1
    results_dict["overall_success"] = success_ctr / 4

    export_to_csv(results_dict, output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert JSON files in a folder to CSV")
    parser.add_argument("input_folder", help="Path to the folder containing JSON files")
    parser.add_argument("output_folder", help="Path to the folder where CSV files will be saved")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in os.listdir(args.input_folder):
        if filename.endswith(".json"):
            input_file = os.path.join(args.input_folder, filename)
            output_file = os.path.join(args.output_folder, f"{os.path.splitext(filename)[0]}.csv")
            process_json_file(input_file, output_file)
            print(f"Processed {filename} into {output_file}")


if __name__ == "__main__":
    main()
