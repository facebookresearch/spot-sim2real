import os
from queue import Queue

from robot_command_runner import CommandTaskRunner
from siro_keyboard_command_builder import KeyboardCommandProcessor
from siro_robot_connection.mobile_robot_dummy_bridge import MobileRobotDummyBridge
from siro_tasks.dock_task import DockTask
from siro_tasks.sit_task import SitTask
from siro_tasks.stand_task import StandTask
from siro_tasks.undock_task import UndockTask
from websockets.exceptions import ConnectionClosed

is_stub_mode = os.environ.get("IS_STUB_MODE", "False").lower() in ("true", "1", "t")
is_use_mobile_manipulation = os.environ.get(
    "USE_MOBILE_MANIPULATION", "True"
).lower() in ("true", "1", "t")
is_external_sim_mode = os.environ.get("IS_EXTERNAL_SIM_MODE", "False").lower() in (
    "true",
    "1",
    "t",
)
keyboard_command_mode = os.environ.get("IS_COMMAND_INTERFACE", "True").lower() in (
    "true",
    "1",
    "t",
)
is_use_spot_fiducials = os.environ.get("IS_USE_SPOT_FIDUCIALS", "False").lower() in (
    "true",
    "1",
    "t",
)


import asyncio
import json
import logging
import socket
import struct
import sys

import siro
import websockets
from siro import (
    ClientCommand,
    ConnectionStatus,
    Fiducial,
    MobileRobotDummyData,
    RoomState,
)
from siro_tasks.base_command_task import BaseCommandTask
from siro_tasks.find_object_task import FindObjectTask
from siro_tasks.navigate_to_task import NavigateToTask
from siro_tasks.place_object_task import PlaceObjectTask
from spot_rl.utils.utils import get_waypoint_yaml

if is_stub_mode or is_external_sim_mode:
    from siro_robot_connection.simulation_robot_bridge import SimulationRobotBridge
else:
    from siro_robot_connection.spot_robot_bridge import SpotRobotBridge

### LOGGER SETUP ###

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filename="server_log.log",
    level=logging.INFO,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.CRITICAL)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler("server_log.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


### SERVER MESSAGES ###


def command_state_json(_current_command):
    return json.dumps(siro.ClientCommand.serialize(_current_command))


def enter_user_message(user):
    json_payload = json.dumps(
        {
            "message_type": siro.SERVER_MESSAGE_TYPE_USER_STATE,
            "state": "enter",
            "user": user.serialize(),
        }
    )
    return json_payload


def exit_user_message(user):
    return json.dumps(
        {
            "message_type": siro.SERVER_MESSAGE_TYPE_USER_STATE,
            "state": "exit",
            "user": user.serialize(),
        }
    )


def room_state_update_message():
    return json.dumps(room_state.serialize())


def get_robot_state_message():
    return json.dumps(robot_bridge.robot_state.serialize())


def connected_users(users):
    serialized_users = "[]" if users is None else [p.user.serialize() for p in users]
    message = json.dumps(
        {"message_type": siro.SERVER_MESSAGE_ACTIVE_USERS, "users": serialized_users}
    )
    print(f"connected_users: {message}")
    return message


def game_state_message():
    return json.dumps(
        {
            "message_type": siro.SERVER_MESSAGE_TYPE_APP_STATE,
            "state": game_state.serialize(),
        }
    )


def shutdown():
    global is_shutdown
    global run_robot_service

    if is_shutdown:
        return
    print("Shutting Down")
    run_robot_service = False
    if robot_bridge is not None:
        robot_bridge.shutdown()
    if keyboard_command is not None:
        keyboard_command.dispose()
    if command_task_runner is not None:
        command_task_runner.dispose()
    print("Shut down complete. Please restart server")
    is_shutdown = True


def start_command_task(active_command: siro.ClientCommand):
    global current_command_task

    if (
        robot_bridge.robot_state.connection_status
        is not siro.ConnectionStatus.Connected
    ):
        print("Robot no connected and cannot complete task")
    else:
        if active_command.current_task_index >= len(active_command.tasks):
            active_command.command_state = siro.CommandState.Complete
            return

        task_state_data = active_command.tasks[active_command.current_task_index]
        current_command.command_state = siro.CommandState.InProgress
        active_task_type = task_state_data.state.task_type
        if active_task_type == siro.ROBOT_TASK_NAVIGATE_TO:
            current_command_task = NavigateToTask(
                robot_bridge, task_state_data, use_policy
            )
        elif active_task_type == siro.ROBOT_TASK_PLACE_OBJECT:
            current_command_task = PlaceObjectTask(
                robot_bridge, task_state_data, use_policy
            )
        elif active_task_type == siro.ROBOT_TASK_FIND_OBJECT:
            current_command_task = FindObjectTask(
                robot_bridge, task_state_data, use_policy
            )
        elif active_task_type == siro.ROBOT_TASK_SIT:
            current_command_task = SitTask(robot_bridge, task_state_data, use_policy)
        elif active_task_type == siro.ROBOT_TASK_STAND:
            current_command_task = StandTask(robot_bridge, task_state_data, use_policy)
        elif active_task_type == siro.ROBOT_TASK_DOCK:
            current_command_task = DockTask(robot_bridge, task_state_data, use_policy)
        elif active_task_type == siro.ROBOT_TASK_UNDOCK:
            current_command_task = UndockTask(robot_bridge, task_state_data, use_policy)
        elif active_task_type == siro.ROBOT_TASK_SHUTDOWN:
            shutdown()
        if current_command_task is not None:
            command_task_queue.put(current_command_task)


async def route_message(original_message, websocket):

    message = json.loads(original_message)

    message_type_to_handler = {
        siro.CLIENT_MESSAGE_GET_CONNECTED_USERS: get_connected_users,
        siro.CLIENT_MESSAGE_TYPE_PLAYER_ENTER: handle_user_enter,
        siro.CLIENT_MESSAGE_TYPE_PLAYER_EXIT: handle_user_exit,
        siro.CLIENT_MESSAGE_ROBOT_COMMAND: handle_user_command_task_list,
    }

    relay_message_type_to_handler = {
        siro.CLIENT_MESSAGE_COMMAND_BUILDER_UPDATE: broadcast_command_builder_update
    }
    if is_external_sim_mode:
        message_type_to_handler[siro.FAKE_ROBOT_ENTER] = handle_fake_robot_enter
        message_type_to_handler[
            siro.FAKE_ROBOT_POSITION_YAW_UPDATE
        ] = handle_fake_robot_position_yaw
        message_type_to_handler[
            siro.FAKE_ROBOT_WAYPOINTS_UPDATE
        ] = handle_fake_robot_waypoints
        message_type_to_handler[
            siro.FAKE_ROBOT_FIDUCIALS_UPDATE
        ] = handle_fake_robot_fiducials
    else:
        logging.info(f"Message Received : {message}")

    if "message_type" in message:
        message_type = message["message_type"]
        if message_type in message_type_to_handler:
            await message_type_to_handler[message_type](message, websocket)
        elif message_type in relay_message_type_to_handler:
            await relay_message_type_to_handler[message_type](
                original_message, websocket
            )


async def broadcast_command_builder_update(message, websocket):
    websockets.broadcast(game_state.connections, message)


async def get_connected_users(message, websocket):
    json_message = connected_users(game_state.users)
    logging.info(f"get_connected_users : {json_message}")
    await websocket.send(json_message)
    logging.info(f"get_connected_users sent")


async def handle_fake_robot_enter(message, websocket):
    print(f"FAKE ROBOT ENTERED!!!")
    fake_robot_data.websocket = websocket
    fake_robot_data.connection_state = ConnectionStatus.Connected


async def handle_fake_robot_position_yaw(message, websocket):
    fake_robot_data.position.x = message["x"]
    fake_robot_data.position.y = message["y"]
    fake_robot_data.yaw = message["yaw"]


async def handle_fake_robot_waypoints(message, websocket):
    room_state.waypoints.deserialize_from_json(message["waypoints"])
    websockets.broadcast(game_state.connections, room_state_update_message())
    if keyboard_command is not None:
        keyboard_command.set_waypoints(room_state.waypoints)


async def handle_fake_robot_fiducials(message, websocket):
    fiducials = message["fiducials"]
    room_state.fiducials.clear()
    for fiducial in fiducials:
        room_state.fiducials.append(Fiducial.deserialize(fiducial))
    websockets.broadcast(game_state.connections, room_state_update_message())


async def handle_user_enter(message, websocket):
    logging.info(f"USER ENTER!!!PLEASE SEND USER DATA HERE!!! {message}")
    data_dict = message["user"]
    user = siro.User.deserialize(data_dict)
    await game_state.add_user(user=user, websocket=websocket)
    logging.info("Server>>>>>>>User Entered")
    logging.info(f"Server>>>>>>>User count: {len(game_state.connections)}")
    websockets.broadcast(game_state.connections, get_robot_state_message())


def reset_server():
    global current_command
    global current_command_task
    robot_bridge.reset()

    current_command = None
    current_command_task = None
    game_state.users.clear()


def quit_server():
    if robot_bridge is not None:
        robot_bridge.shutdown()
    exit()


async def handle_user_exit(message, websocket):
    user_wrapper = game_state.get_user_by_websocket(websocket)
    if user_wrapper is not None:
        await game_state.remove_user(user_wrapper)
        websockets.broadcast(
            game_state.connections, exit_user_message(user_wrapper.user)
        )
    else:
        if fake_robot_data is not None and websocket is fake_robot_data.websocket:
            fake_robot_data.connection_status = ConnectionStatus.NotConnected
            fake_robot_data.websocket = None
            logging.warning(f"Fake robot disconnected")
        else:
            logging.warning(f"user not found by websocket id: {str(websocket.id)}")


async def handle_user_command_task_list(message, websocket):
    global current_command

    if current_command is not None:
        logging.info(
            "SIRO Warning>>>>>Attempt to start a new command when there is one in progress"
        )
    elif len(message["tasks"]) < 1:
        logging.info("SIRO Warning>>>>>You have sent an empty command with no tasks!")
    else:
        client_command = siro.ClientCommand.deserialize(message)
        pending_command_queue.put(client_command)


# RUN SERVER #


async def handle_websocket(websocket, path):
    try:
        await websocket.send(room_state_update_message())
        logging.info("Sent room_state to new websocket connection!")
        # sync server game state to newly connected game client
        await websocket.send(game_state_message())
        # route and handle messages for duration of websocket connection
        async for message in websocket:
            await route_message(message, websocket)
    except ConnectionClosed:
        logging.info("Received unexpected connection close.")
    finally:
        logging.info("User connection closed")
        # upon websocket disconnect remove client's user
        if fake_robot_data is not None:
            if websocket == fake_robot_data.websocket:
                fake_robot_data.websocket = None
                fake_robot_data.connection_state = ConnectionStatus.NotConnected
        await handle_user_exit(None, websocket)
        if len(game_state.connections) == 0:
            reset_server()


def process_current_task():
    global current_command

    if current_command_task is not None:
        task_state = current_command_task.task.get_state()
        if task_state == siro.TaskState.Success:
            print(f"Task success {current_command_task.task.state.task_type}")
            current_command.current_task_index = current_command.current_task_index + 1
            start_command_task(current_command)
            websockets.broadcast(
                game_state.connections, command_state_json(current_command)
            )
        elif task_state == siro.TaskState.Error or task_state == siro.TaskState.Fail:
            print(f"Task failure {current_command_task.task.state.task_type}")
            current_command.command_state = siro.CommandState.Error


def process_command_queue():
    global current_command
    if current_command is None:
        if pending_command_queue.qsize() > 0:
            current_command = pending_command_queue.get()
            current_command.command_state = siro.CommandState.Starting
            start_command_task(current_command)
            websockets.broadcast(
                game_state.connections, command_state_json(current_command)
            )


async def server():
    async with (
        websockets.serve(
            handle_websocket, host=os.environ["SIRO_HOST"], port=os.environ["SIRO_PORT"]
        )
    ):
        # Network discovery related stuff:
        current_time = 0
        discovery_send_every_seconds = 1
        last_discovery_send_time = 0
        hostname = socket.gethostname()
        discovery_message = (
            "SERVER:" + socket.gethostbyname(hostname) + ":" + os.environ["SIRO_PORT"]
        ).encode()
        multicast_group = ("224.10.10.10", 7475)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        ttl = struct.pack("b", 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

        while run_robot_service:
            global current_command
            global current_command_task
            process_command_queue()

            if multicast_discovery and (
                last_discovery_send_time < (current_time - discovery_send_every_seconds)
            ):
                sock.sendto(discovery_message, multicast_group)
                last_discovery_send_time = current_time

            if current_command is not None:
                robot_bridge.update()
                if current_command.command_state == siro.CommandState.Complete:
                    logging.info(f"Command Complete: {current_command.command_id}")
                    websockets.broadcast(
                        game_state.connections, command_state_json(current_command)
                    )
                    current_command_task = None
                    current_command = None
                    if keyboard_command is not None:
                        keyboard_command.enable_input()
                elif current_command.command_state == siro.CommandState.Error:
                    logging.info(
                        f"Command ended with error: {current_command.error_message}"
                    )
                    websockets.broadcast(
                        game_state.connections, command_state_json(current_command)
                    )
                    current_command_task = None
                    current_command = None
                    if keyboard_command is not None:
                        keyboard_command.enable_input()
                elif current_command.command_state == siro.CommandState.InProgress:
                    process_current_task()
            websockets.broadcast(game_state.connections, get_robot_state_message())
            await asyncio.sleep(tick_sleep)
            current_time = current_time + tick_sleep
        await asyncio.Future()


def print_init_mode_message():
    if is_stub_mode:
        print("SIRO>>>>>>You're currently running in Stub mode!")
    if is_external_sim_mode:
        print(
            "SIRO>>>>>>>>>>>>You are running in External Sim Mode. Please connect the mobile_dummy_robot app"
        )


def initialise_waypoints():
    waypoints_yaml_dict = get_waypoint_yaml()
    room_state.waypoints.deserialize_from_yaml(waypoints_yaml_dict)


def print_waypoints():
    logging.info("------------------WAYPOINTS------------------")
    logging.info(room_state.waypoints.serialize())
    logging.info("---------------------------------------------")


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    host = s.getsockname()[0]
    s.close()
    return host


print(f"Running at {get_ip_address()}")


def is_networking_on():
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


if not is_networking_on():
    print("YOU ARE CURRENTLY NOT CONNECTED TO A NETWORK!")

# UDP Multicast for client discovery
pending_command_queue: Queue = Queue()
command_task_queue: Queue = Queue()
multicast_discovery = True
keyboard_command: KeyboardCommandProcessor = None
fake_robot_data: MobileRobotDummyData = None
room_state: RoomState = RoomState()
current_command: ClientCommand = None
current_command_task: BaseCommandTask = None
run_robot_service = True
tick_sleep = 0.0333  # 1 sec / 30 fps
is_shutdown = False

# Spot Init:
print_init_mode_message()
use_policy = (
    not is_stub_mode and not is_external_sim_mode and is_use_mobile_manipulation
)

print("about to init bridge")
if is_external_sim_mode:
    fake_robot_data = MobileRobotDummyData(ConnectionStatus.NotConnected)
    robot_bridge = MobileRobotDummyBridge(fake_robot_data)
    print("Init as External Sim")
elif is_stub_mode:
    robot_bridge = SimulationRobotBridge()
    print("Init as Stub")
else:
    robot_bridge = SpotRobotBridge()
    print("Init as Robot bridge")
    print(robot_bridge.get_nav_policy())
    print(robot_bridge.get_gaze_policy())
    print(robot_bridge.get_place_policy())

game_state = siro.GameState(game_state_lock=asyncio.Lock())
initialise_waypoints()
robot_bridge.startup()
print_waypoints()


if is_use_spot_fiducials:
    room_state.fiducials = robot_bridge.get_fiducial_world_objects()


if keyboard_command_mode:
    keyboard_command = KeyboardCommandProcessor(
        room_state.waypoints, pending_command_queue
    )

command_task_runner = CommandTaskRunner(
    command_queue=command_task_queue, spot=robot_bridge
)

try:
    asyncio.run(server())
except KeyboardInterrupt:
    print("Keyboard Interaction Closed")
finally:
    shutdown()
    print("Server Shutdown")
