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

    # Create the figure and axis objects
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12, 12))

    # Initialize the plot with some default data
    joint_names, torques = get_arm_joints_from_spot(False)
    bars = axes[0].bar(joint_names, torques)

    # Set the labels
    axes[0].set_xlabel("Joint Name")
    axes[0].set_ylabel("Torque (N·m)")
    axes[0].set_ylim(0, 20)  # Set an appropriate limit for your torques
    torque_data = {i: [] for i in range(8)}  # type: ignore

    max_len = 1000

    for i in range(1, 8):
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel(f"{joint_names[i-1]}")

    # Function to update the plot
    def update(frame):
        # Plot the instant torque
        _, torques = get_arm_joints_from_spot(False)
        for bar, torque in zip(bars, torques):
            bar.set_height(torque)

        # Plot the history of the troque
        for i in range(7):
            axes[i + 1].cla()
            torque_data[i].append(torques[i])
            torque_data[i] = torque_data[i][-max_len:]
            index = [i for i in range(len(torque_data[i]))]
            axes[i + 1].plot(index, torque_data[i])
            axes[i + 1].set_xlim([0, max_len])

        fig.tight_layout()

    # Create an animation
    ani = FuncAnimation(fig, update, frames=range(100), blit=False, interval=100)

    # Display the plot
    plt.show()
