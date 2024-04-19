from enum import IntEnum


class ConnectionStatus(IntEnum):
    NoStatus = 0
    NotConnected = 1
    Connecting = 2
    Connected = 3
    Disconnecting = 4
    Error = 5


class CommandState(IntEnum):
    NoState = 0
    Starting = 1
    InProgress = 2
    Complete = 3
    Error = 4


class TaskState(IntEnum):
    NoState = 0
    Starting = 1
    InProgress = 2
    Success = 3
    Fail = 4
    Error = 5


# https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-SE2TrajectoryCommand-Feedback-Status
class RobotTrajectoryStatus(IntEnum):
    Unknown = (
        0,
    )  # STATUS_UNKNOWN should never be used. If used, an internal error has happened.
    AtGoal = (1,)  # The robot has arrived and is standing at the goal.
    GoingToGoal = (
        2,
    )  # The robot has arrived at the goal and is doing final positioning.
    NearGoal = 3  # The robot is attempting to go to a goal.
