import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

spot = None


def get_arm_joints_from_spot(rename=True):
    global spot
    assert spot is not None, "Spot object not initialized"
    joint_states = (
        spot.robot_state_client.get_robot_state().kinematic_state.joint_states
    )
    arm_joint_names, arm_torque_values = [], []
    i = 1
    for jstate in joint_states:
        if "arm0" in str(jstate.name):
            jointname = str(jstate.name).split(".")[-1]
            arm_joint_names.append("J_" + str(i)) if rename else arm_joint_names.append(
                jointname
            )
            arm_torque_values.append(float(jstate.load.value))
            i += 1
    assert len(arm_joint_names) > 0, "No arm joints were found in given joint_states"
    return arm_joint_names, arm_torque_values


if __name__ == "__main__":
    from spot_wrapper.spot import Spot

    spot = Spot("arm collision detector")

    joint_names_original, torques = get_arm_joints_from_spot(False)
    print(joint_names_original)
    # hardware_link_info = robot_state_client.get_hardware_config_with_link_info()
    # metrics = robot_state_client.get_robot_metrics()

    # Create the figure and axis objects
    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Initialize the plot with some default data
    joint_names, torques = get_arm_joints_from_spot(False)
    bars = ax.bar(joint_names, torques)

    # Set the labels
    ax.set_xlabel("Joint Name")
    ax.set_ylabel("Torque (N·m)")
    ax.set_ylim(0, 20)  # Set an appropriate limit for your torques

    # Function to update the plot
    def update(frame):
        joint_names, torques = get_arm_joints_from_spot(False)
        # print(torques)
        # torques = np.random.rand(8) * 10
        for bar, torque in zip(bars, torques):
            bar.set_height(torque)
        return bars
        ax.clear()  # Clear previous bars
        ax.bar(joint_names, torques)
        ax.set_xlabel("Joint Name")
        ax.set_ylabel("Torque (N·m)")
        ax.set_ylim(0, 20)
        return bars

    # Create an animation
    ani = FuncAnimation(fig, update, frames=range(100), blit=True, interval=100)

    # Display the plot
    plt.show()
