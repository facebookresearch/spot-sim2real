import json


def room_state_yaml_serialize(waypoints_yaml):
    print(f"room_state_yaml_serialize")
    waypoints_json = {
        "message_type": "room_state",
        "waypoints": {
            "place_targets": [],
            "clutter_targets": [],
            "nav_targets": [],
            "object_targets": [],
        },
    }  # json.dumps()
    for place_target in waypoints_yaml["place_targets"]:
        x = waypoints_yaml["place_targets"][place_target][0]
        y = waypoints_yaml["place_targets"][place_target][1]
        z = waypoints_yaml["place_targets"][place_target][2]
        waypoints_json["waypoints"]["place_targets"].append(
            {"name": place_target, "place_position_for_arm": {"x": x, "y": y, "z": z}}
        )
    for clutter_amount in waypoints_yaml["clutter_amounts"]:
        amount = waypoints_yaml["clutter_amounts"][clutter_amount]
        waypoints_json["waypoints"]["clutter_targets"].append(
            {"name": clutter_amount, "number_objects": amount}
        )
    for nav_target in waypoints_yaml["nav_targets"]:
        print(
            f"nav target : {nav_target} | {waypoints_yaml['nav_targets'][nav_target]}"
        )
        x = waypoints_yaml["nav_targets"][nav_target][0]
        y = waypoints_yaml["nav_targets"][nav_target][1]
        yaw = waypoints_yaml["nav_targets"][nav_target][2]
        waypoints_json["waypoints"]["nav_targets"].append(
            {"name": nav_target, "position": {"x": x, "y": y}, "yaw": yaw}
        )
    for object_target in waypoints_yaml["object_targets"]:
        name = waypoints_yaml["object_targets"][object_target][0]
        placement = waypoints_yaml["object_targets"][object_target][1]
        waypoints_json["waypoints"]["object_targets"].append(
            {"name": name, "suggested_placement": placement}
        )
        return json.dumps(waypoints_json)
