import logging
import threading

import siro

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
)

# Game State:


class GameState:
    def __init__(self, game_state_lock):
        self.game_state_lock = game_state_lock
        self.connections = set()
        self.users = set()

    async def add_user(self, user, websocket):
        async with self.game_state_lock:
            user_wrapper = UserServerWrapper(user, websocket)
            self.users.add(user_wrapper)
            self.connections.add(websocket)

    async def remove_user(self, user):
        async with self.game_state_lock:
            self.connections.remove(user.websocket)
            self.users.remove(user)

    def get_user_by_websocket(self, websocket):
        for user_wrapper in self.users:
            if user_wrapper.websocket is websocket:
                return user_wrapper
        return None

    def serialize(self):
        serialized_users = []
        try:
            serialized_users = [p.user.serialize() for p in self.users]
        finally:
            return {"connected_users": serialized_users}


class UserServerWrapper:
    def __init__(self, user, websocket):
        self.websocket = websocket
        self.user = user
        self.websocketId = websocket.id


# User and Commands:


class RobotState:
    def __init__(self, position, yaw, connection_status):
        self.position = position
        self.yaw = yaw
        self.connection_status = connection_status
        self.lock = threading.Lock()

    def serialize(self):
        with self.lock:
            blob = {
                "message_type": "robot_state",
                "position": self.position.serialize(),
                "yaw": self.yaw,
                "connection_status": self.connection_status,
            }
        return blob

    def set_x_y_yaw(self, x, y, yaw):
        with self.lock:
            self.position.x = x
            self.position.y = y
            self.yaw = yaw


class User:
    def __init__(
        self,
        user_id,
        username,
        display_name,
        color_index,
        color,
        is_master_user,
        network_user_id,
    ):
        self.user_id = user_id
        self.username = username
        self.display_name = display_name
        self.color_index = color_index
        self.color = color
        self.is_master_user = is_master_user
        self.network_user_id = network_user_id

    @staticmethod
    def deserialize(data_dict):
        user_id = data_dict["id"]
        user_name = data_dict["username"]
        user_display_name = data_dict["display_name"]
        network_user_id = data_dict["network_user_id"]
        user_color_index = data_dict["color_index"]
        user_color_dict = data_dict["color"]
        user_is_master_user = data_dict["is_master_user"]
        user_color = siro.Color.deserialize(user_color_dict)
        return User(
            user_id=user_id,
            username=user_name,
            display_name=user_display_name,
            color_index=user_color_index,
            color=user_color,
            is_master_user=user_is_master_user,
            network_user_id=network_user_id,
        )

    def serialize(self):
        return {
            "id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "network_user_id": self.network_user_id,
            "is_master_user": self.is_master_user,
            "color_index": self.color_index,
            "color": self.color.serialize(),
        }


class ClientCommand:
    def __init__(self, command_id, started_by, tasks):
        self.command_id = command_id
        self.started_by = started_by
        self.tasks = tasks
        self.command_state = siro.CommandState.NoState
        self.error_message = ""
        self.current_task_index = 0

    @staticmethod
    def deserialize(data_dict):
        command_id = data_dict["id"]
        started_by = User.deserialize(data_dict["started_by"])
        tasks = []

        for task_data in data_dict["tasks"]:
            task_type = task_data["task_type"]
            task_specific_data = task_data["data"]
            if task_type == siro.ROBOT_TASK_FIND_OBJECT:
                tasks.append(FindObjectData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_PLACE_OBJECT:
                tasks.append(PlaceObjectData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_NAVIGATE_TO:
                tasks.append(NavigateToData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_SIT:
                tasks.append(SitData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_STAND:
                tasks.append(StandData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_DOCK:
                tasks.append(DockData.deserialize(task_specific_data))
            elif task_type == siro.ROBOT_TASK_UNDOCK:
                tasks.append(UndockData.deserialize(task_specific_data))
            else:
                logging.info(f"Error::TASK NOT RECOGNISED {task_type}")

        return ClientCommand(command_id, started_by, tasks)

    @staticmethod
    def serialize(self):
        serialized_tasks_statuses = []
        for task_data in self.tasks:
            task_state = task_data.state
            serialized_tasks_statuses.append(CommandTaskStatus.serialize(task_state))

        return {
            "message_type": siro.SERVER_MESSAGE_TYPE_COMMAND_STATE,
            "id": self.command_id,
            "started_by": User.serialize(self.started_by),
            "command_state": self.command_state,
            "tasks": serialized_tasks_statuses,
        }


class CommandTaskStatus:
    def __init__(self, task_state, task_type, info):
        self.task_state = task_state
        self.task_type = task_type
        self.info = info

    @staticmethod
    def serialize(self):
        return {
            "task_state": self.task_state,
            "task_type": self.task_type,
            "info": self.info,
        }


class CommandTaskData(object):
    def __init__(self, command_type):
        self.lock = threading.Lock()
        self.state = CommandTaskStatus(siro.TaskState.NoState, command_type, "")
        self.robot_command_id = -1

    def set_state(self, state):
        with self.lock:
            self.state.task_state = state

    def get_state(self):
        with self.lock:
            return self.state.task_state

    def create_empty(self):
        pass


class NavigateToData(CommandTaskData):
    def __init__(self, name, position, yaw):
        super().__init__(siro.ROBOT_TASK_NAVIGATE_TO)
        self.name = name
        self.position = position
        self.yaw = yaw

    @staticmethod
    def deserialize(data_dict):
        position = siro.Vector2.deserialize(data_dict["position"])
        return NavigateToData(data_dict["name"], position, data_dict["yaw"])

    def serialize(self):
        return {
            "name": self.name,
            "position": self.position.serialize(),
            "yaw": self.yaw,
        }

    def create_empty(self):
        return NavigateToData(self.name, self.position, self.yaw)

    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#robotcommandfeedbackstatus-status#
    def set_status_from_robot_feedback(self, robot_feedback):
        robot_state = (
            robot_feedback.feedback.synchronized_feedback.mobility_command_feedback
        ).se2_trajectory_feedback.status
        if robot_state == 0:
            self.set_state(siro.TaskState.NoState)
        elif robot_state == 1:
            self.set_state(siro.TaskState.Success)
        elif robot_state == 2:
            self.set_state(siro.TaskState.InProgress)
        elif robot_state == 3:
            self.set_state(siro.TaskState.InProgress)


class FindObjectData(CommandTaskData):
    def __init__(self, objects_to_find):
        super().__init__(siro.ROBOT_TASK_FIND_OBJECT)
        self.objects_to_find = objects_to_find

    @staticmethod
    def deserialize(data_dict):
        return FindObjectData(data_dict["objects_to_find"])

    @staticmethod
    def serialize(self):
        return {"objects_to_find": self.objects_to_find}

    def create_empty(self):
        return FindObjectData(self.objects_to_find)


class PlaceObjectData(CommandTaskData):
    def __init__(self, arm_placement_position):
        super().__init__(siro.ROBOT_TASK_PLACE_OBJECT)
        self.arm_placement_position = arm_placement_position

    @staticmethod
    def deserialize(data_dict):
        position = siro.Vector3.deserialize(data_dict["arm_placement_position"])
        return PlaceObjectData(position)

    def serialize(self):
        return {"arm_placement_position": self.arm_placement_position.serialize()}

    def create_empty(self):
        return PlaceObjectData(self.arm_placement_position)


class ShutdownData(CommandTaskData):
    def __init__(self):
        super().__init__(siro.ROBOT_TASK_SHUTDOWN)

    @staticmethod
    def deserialize(data_dict):
        return ShutdownData()

    def serialize(self):
        return {}

    def create_empty(self):
        return ShutdownData()


class SitData(CommandTaskData):
    def __init__(self):
        super().__init__(siro.ROBOT_TASK_SIT)

    @staticmethod
    def deserialize(data_dict):
        return SitData()

    def serialize(self):
        return {}

    def create_empty(self):
        return SitData()


class DockData(CommandTaskData):
    def __init__(self):
        super().__init__(siro.ROBOT_TASK_DOCK)

    @staticmethod
    def deserialize(data_dict):
        return DockData()

    def serialize(self):
        return {}

    def create_empty(self):
        return DockData()


class UndockData(CommandTaskData):
    def __init__(self):
        super().__init__(siro.ROBOT_TASK_UNDOCK)

    @staticmethod
    def deserialize(data_dict):
        return UndockData()

    def serialize(self):
        return {}

    def create_empty(self):
        return UndockData()


class StandData(CommandTaskData):
    def __init__(self):
        super().__init__(siro.ROBOT_TASK_STAND)

    @staticmethod
    def deserialize(data_dict):
        return StandData()

    def serialize(self):
        return {}

    def create_empty(self):
        return StandData()


# Vectors & Colors:
class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def deserialize(data_dict):
        return Vector2(data_dict["x"], data_dict["y"])

    def serialize(self):
        return {"x": self.x, "y": self.y}


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def deserialize(data_dict):
        return Vector3(data_dict["x"], data_dict["y"], data_dict["z"])

    def serialize(self):
        return {"x": self.x, "y": self.y, "z": self.z}


class Color:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    @staticmethod
    def deserialize(data_dict):
        logging.info(data_dict)
        return Color(data_dict["r"], data_dict["g"], data_dict["b"], data_dict["a"])

    def serialize(self):
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}


class Fiducial:
    def __init__(self, tag_id: str, position: Vector3):
        self.tag_id = tag_id
        self.position = position

    @staticmethod
    def deserialize(data_dict):
        position = Vector3.deserialize(data_dict["position"])
        return Fiducial(data_dict["tag_id"], position)

    def serialize(self):
        return {"id": self.tag_id, "position": self.position.serialize()}


class NavigationTarget:
    def __init__(self, name, position, yaw):
        self.name = name
        self.position = position
        self.yaw = yaw

    def serialize(self):
        return {
            "name": self.name,
            "position": self.position.serialize(),
            "yaw": self.yaw,
        }

    @staticmethod
    def deserialize(data_dict):
        position = Vector2.deserialize(data_dict["position"])
        return NavigationTarget(data_dict["name"], position, data_dict["yaw"])


class PlaceTarget:
    def __init__(self, name, place_position_for_arm):
        self.name = name
        self.place_position_for_arm = place_position_for_arm

    def serialize(self):
        return {
            "name": self.name,
            "place_position_for_arm": self.place_position_for_arm.serialize(),
        }

    @staticmethod
    def deserialize(data_dict):
        position = Vector3.deserialize(data_dict["place_position_for_arm"])
        return PlaceTarget(data_dict["name"], position)


class ObjectTarget:
    def __init__(self, name, suggested_placement):
        self.name = name
        self.suggested_placement = suggested_placement

    def serialize(self):
        return {"name": self.name, "suggested_placement": self.suggested_placement}

    @staticmethod
    def deserialize(data_dict):
        return ObjectTarget(data_dict["name"], data_dict["suggested_placement"])


class ClutterTarget:
    def __init__(self, name, clutter_amount):
        self.name = name
        self.clutter_amount = clutter_amount

    def serialize(self):
        return {"name": self.name, "number_objects": self.clutter_amount}

    @staticmethod
    def deserialize(data_dict):
        return ClutterTarget(data_dict["name"], data_dict["number_objects"])


class WaypointData:
    def __init__(self):
        self.place_targets = []
        self.clutter_targets = []
        self.object_targets = []
        self.nav_targets = []

    def reset(self):
        self.place_targets.clear()
        self.clutter_targets.clear()
        self.object_targets.clear()
        self.nav_targets.clear()

    def deserialize_from_yaml(self, data_dict):
        self.reset()
        for name, coords in data_dict["nav_targets"].items():
            self.nav_targets.append(
                NavigationTarget(name, Vector2(coords[0], coords[1]), coords[2])
            )
        for place_target in data_dict["place_targets"]:
            x = data_dict["place_targets"][place_target][0]
            y = data_dict["place_targets"][place_target][1]
            z = data_dict["place_targets"][place_target][2]
            self.place_targets.append(PlaceTarget(place_target, Vector3(x, y, z)))
        for object_target in data_dict["object_targets"]:
            name = data_dict["object_targets"][object_target][0]
            placement = data_dict["object_targets"][object_target][1]
            self.object_targets.append(ObjectTarget(name, placement))
        for clutter_name in data_dict["clutter_amounts"]:
            amount = data_dict["clutter_amounts"][clutter_name]
            self.clutter_targets.append(ClutterTarget(clutter_name, amount))

    def deserialize_from_json(self, data_dict):
        self.reset()
        for nav_target in data_dict["nav_targets"]:
            self.nav_targets.append(NavigationTarget.deserialize(nav_target))
        for place_target in data_dict["place_targets"]:
            self.place_targets.append(PlaceTarget.deserialize(place_target))
        for object_target in data_dict["object_targets"]:
            self.object_targets.append(ObjectTarget.deserialize(object_target))
        for clutter_target in data_dict["clutter_targets"]:
            self.clutter_targets.append(ClutterTarget.deserialize(clutter_target))

    def serialize(self):
        return {
            "clutter_targets": [p.serialize() for p in self.clutter_targets],
            "nav_targets": [p.serialize() for p in self.nav_targets],
            "object_targets": [p.serialize() for p in self.object_targets],
            "place_targets": [p.serialize() for p in self.place_targets],
        }


class RoomState:
    def __init__(self):
        self.waypoints = WaypointData()
        self.fiducials = []

    def serialize(self):
        return {
            "message_type": siro.SERVER_MESSAGE_TYPE_ROOM_STATE,
            "waypoints": self.waypoints.serialize(),
            "fiducials": [p.serialize() for p in self.fiducials],
        }


class MobileRobotDummyData:
    def __init__(self, connection_status):
        self.websocket = None
        self.position = Vector2(0, 0)
        self.yaw = 0
        self.connection_status = connection_status
        self.fiducials = set()
        self.targetPosition = Vector2(0, 0)
        self.targetYaw = 0

    def set_base_position(self, x, y, yaw):
        self.targetPosition = Vector2(x, y)
        self.targetYaw = yaw
