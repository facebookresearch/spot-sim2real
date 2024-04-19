import threading
from queue import Queue

import siro
from siro import (
    ROBOT_TASK_DOCK,
    ROBOT_TASK_SHUTDOWN,
    ROBOT_TASK_UNDOCK,
    ClientCommand,
    CommandTaskData,
)
from siro.data_types import (
    DockData,
    NavigateToData,
    PlaceObjectData,
    ShutdownData,
    UndockData,
    Vector2,
    WaypointData,
)


class KeyboardThread(threading.Thread):
    stop_thread = False

    def __init__(self, input_cbk=None, name="keyboard-input-thread"):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)

        self.start()

    def run(self):
        while not self.stop_thread:
            self.input_cbk(input())  # waits to get input + Return

    def shutdown(self):
        self.stop_thread = True


class KeyboardCommandProcessor:

    started_issuing_command = False
    current_task_creation = -1

    def __init__(self, waypoints: WaypointData, command_queue: Queue):
        self.waypoints = waypoints
        self.current_command: ClientCommand = None
        self.current_task_list = []
        self.command_queue = command_queue
        self.last_command_sent: ClientCommand = None
        self.kthread = KeyboardThread(self.process_input)
        print(self.waypoints)
        self.create_command()
        self.show_task_input_choices()
        self.show_extra_menu_choices()
        self.show_end_menu_line()

    def dispose(self):
        if self.kthread is not None:
            self.kthread.shutdown()
            self.kthread = None

    def create_command(self):
        print("---------------------------------------------")
        print("Currently running in keyboard command mode.")
        print("---------------------------------------------")
        self.current_task_list.clear()
        command_id = "NA"
        started_by = siro.User(
            "NA",
            "SERVER",
            "SERVER",
            0,
            siro.Color(1, 0, 0, 1),
            False,
            "dummy_network_id",
        )
        self.current_command = ClientCommand(
            command_id, started_by, self.current_task_list
        )
        self.started_issuing_command = True
        self.current_task_creation = -1

    @staticmethod
    def show_task_input_choices():
        print("------------------------")
        print("----Command Builder-----")
        print("------------------------")
        print("Type one of the following + 'Enter' to continue:")
        print("1 - Navigate to target")
        print("2 - Pick Up Object")
        print("3 - Place Object")
        print("4 - Navigate to and Place Object")
        print("5 - Stand")
        print("6 - Sit")
        print("----------OR------------")
        print("y - Run the command")
        print("u - Start command again")

    @staticmethod
    def show_extra_menu_choices():
        print("------------------------")
        print("----Standalone Tasks----")
        print("------------------------")
        print("d - Dock Robot")
        print("f - Undock Robot")
        print("z - Go to Robot Home")
        print("q - Shutdown Server")

    @staticmethod
    def show_end_menu_line():
        print("------------------------")
        print("")

    def add_task(self, task_data: CommandTaskData):
        self.current_command.tasks.append(task_data)
        self.show_task_input_choices()
        self.show_end_menu_line()
        self.current_task_creation = -1

    def send_command(self, command: ClientCommand):
        print("send_command")
        command.command_state = siro.CommandState.Starting
        self.command_queue.put(command)
        self.current_task_creation = -1
        self.started_issuing_command = False

    def process_input(self, inp):

        if self.kthread is None:
            return

        if self.started_issuing_command:
            if self.current_task_creation == -1:
                self.process_main_menu_selection(inp)
            elif inp.isdigit():
                self.process_task_sub_decision(inp)

    def enable_input(self):
        self.create_command()
        self.show_task_input_choices()
        self.show_extra_menu_choices()
        self.show_end_menu_line()

    def on_command_failed(self, command: ClientCommand):
        # find the task that failed
        # add as an option to try again or reset the robot
        self.create_command()
        self.show_task_input_choices()
        self.show_extra_menu_choices()
        self.show_end_menu_line()

    def set_waypoints(self, waypoints: WaypointData):
        self.waypoints = waypoints

    @staticmethod
    def copy_command(current_command):
        tasks = []
        for t in current_command.tasks:
            tasks.append(t.create_empty())
        return ClientCommand(
            current_command.command_id, current_command.started_by, tasks
        )

    def process_task_sub_decision(self, inp):
        selection = int(inp) - 1
        if self.current_task_creation == 1:
            count = len(self.waypoints.nav_targets)
            if selection < count:
                print("------------")
                selected_nav_target = self.waypoints.nav_targets[selection]
                print("Added task - navigate to target: ", selected_nav_target.name)
                print(
                    f"{selected_nav_target.position.x},{selected_nav_target.position.y},{selected_nav_target.yaw}"
                )
                self.add_task(
                    siro.NavigateToData(
                        selected_nav_target.name,
                        selected_nav_target.position,
                        selected_nav_target.yaw,
                    )
                )
        elif self.current_task_creation == 2:
            count = len(self.waypoints.object_targets)
            if selection < count:
                recognised_objects = self.waypoints.object_targets
                pick_up_selection = recognised_objects[selection]
                print("------------")
                print("Sent command - pick up: ", pick_up_selection.name)
                pick_up_items = [pick_up_selection.name]
                print("Added task - picking up items: ", pick_up_items)
                self.add_task(siro.FindObjectData(pick_up_items))
        elif self.current_task_creation == 3:
            count = len(self.waypoints.place_targets)
            if selection < count:
                place_selection = self.waypoints.place_targets[selection]
                arm_placement = place_selection.place_position_for_arm
                print(
                    f"adding place object task: {place_selection.name} | {arm_placement.x} | {arm_placement.y} | {arm_placement.z}"
                )
                self.add_task(PlaceObjectData(arm_placement))
        elif self.current_task_creation == 4:
            count = len(self.waypoints.place_targets)
            if selection < count:
                place_selection = self.waypoints.place_targets[selection]
                arm_placement = place_selection.place_position_for_arm
                for nav_target in self.waypoints.nav_targets:
                    if nav_target.name == place_selection.name:
                        print(f"adding nav task: {nav_target.name}")
                        self.current_command.tasks.append(
                            NavigateToData(
                                nav_target.name, nav_target.position, nav_target.yaw
                            )
                        )
                        break
                print(
                    f"adding place object task: {place_selection.name} | {arm_placement.x} | {arm_placement.y} | {arm_placement.z}"
                )
                self.add_task(PlaceObjectData(arm_placement))

    def process_main_menu_selection(self, inp):
        if inp == "1":
            if (
                self.waypoints.nav_targets is None
                or len(self.waypoints.nav_targets) == 0
            ):
                print("No nav targets to select from. Please select another option")
            else:
                self.current_task_creation = 1
                print("------------")
                print("Please select a nav_target to move to: ")
                index = 1
                for nav_target in self.waypoints.nav_targets:
                    print(
                        index,
                        " - ",
                        nav_target.name,
                        nav_target.position.x,
                        nav_target.position.y,
                    )
                    index = index + 1
        elif inp == "2":
            if (
                self.waypoints.object_targets is None
                or len(self.waypoints.object_targets) == 0
            ):
                print("No object targets to select from. Please select another option")
            else:
                self.current_task_creation = 2
                print("------------")
                print("Please select an item to pick up: ")
                index = 1
                for object_target in self.waypoints.object_targets:
                    name = object_target.name
                    print(index, " - ", name)
                    index = index + 1
        elif inp == "3":
            if (
                self.waypoints.place_targets is None
                or len(self.waypoints.place_targets) == 0
            ):
                print("No place targets to select from. Please select another option")
            else:
                self.current_task_creation = 3
                print("------------")
                print("Please select a location to place the item: ")
                index = 1
                for place_target in self.waypoints.place_targets:
                    print(index, " - ", place_target.name)
                    index = index + 1
        elif inp == "4":
            if (
                self.waypoints.place_targets is None
                or len(self.waypoints.place_targets) == 0
            ):
                print("No place targets to select from. Please select another option")
            else:
                self.current_task_creation = 4
                print("------------")
                print("Please select a location to place the item: ")
                index = 1
                for place_target in self.waypoints.place_targets:
                    print(index, " - ", place_target.name)
                    index = index + 1
        elif inp == "5":
            self.add_task(siro.StandData())
        elif inp == "6":
            self.add_task(siro.SitData())
        elif inp == "c" or inp == "u":
            print("Resetting Command")
            self.create_command()
            self.show_task_input_choices()
            self.show_extra_menu_choices()
            self.show_end_menu_line()
        elif inp == "y":
            if len(self.current_command.tasks) > 0:
                self.last_command_sent = self.copy_command(self.current_command)
                self.send_command(self.current_command)
            else:
                print("Please add tasks to the command before running.")
                self.show_task_input_choices()
                self.show_extra_menu_choices()
                self.show_end_menu_line()
        elif inp == "d":
            self.current_task_list.clear()
            command_id = ROBOT_TASK_DOCK
            started_by = siro.User(
                "NA",
                "SERVER",
                "SERVER",
                0,
                siro.Color(1, 0, 0, 1),
                False,
                "dummy_network_id",
            )
            self.current_command = ClientCommand(
                command_id, started_by, self.current_task_list
            )
            for nav_target in self.waypoints.nav_targets:
                if nav_target.name == "dock":
                    nav_data = NavigateToData(
                        name=nav_target.name,
                        position=nav_target.position,
                        yaw=nav_target.yaw,
                    )
                    self.current_command.tasks.append(nav_data)
                    self.current_command.tasks.append(DockData())
                    self.send_command(self.current_command)
                    break
        elif inp == "f":
            self.current_task_list.clear()
            command_id = ROBOT_TASK_UNDOCK
            started_by = siro.User(
                "NA",
                "SERVER",
                "SERVER",
                0,
                siro.Color(1, 0, 0, 1),
                False,
                "dummy_network_id",
            )
            self.current_command = ClientCommand(
                command_id, started_by, self.current_task_list
            )
            self.current_command.tasks.append(UndockData())
            self.send_command(self.current_command)
        elif inp == "z":
            self.current_task_list.clear()
            command_id = "go_to_zero"
            started_by = siro.User(
                "NA",
                "SERVER",
                "SERVER",
                0,
                siro.Color(1, 0, 0, 1),
                False,
                "dummy_network_id",
            )
            self.current_command = ClientCommand(
                command_id, started_by, self.current_task_list
            )
            nav_data = NavigateToData(name="zero_point", position=Vector2(0, 0), yaw=0)
            self.current_command.tasks.append(nav_data)
            self.send_command(self.current_command)
        elif inp == "q":
            self.current_task_list.clear()
            command_id = ROBOT_TASK_SHUTDOWN
            started_by = siro.User(
                "NA",
                "SERVER",
                "SERVER",
                0,
                siro.Color(1, 0, 0, 1),
                False,
                "dummy_network_id",
            )
            self.current_command = ClientCommand(
                command_id, started_by, self.current_task_list
            )
            self.current_command.tasks.append(ShutdownData())
            self.send_command(self.current_command)
