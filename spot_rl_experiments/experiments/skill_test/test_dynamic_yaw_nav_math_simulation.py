import math

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from svgpath2mpl import parse_path
from svgpathtools import svg2paths

axis_size = 10


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in rad
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0], [0.1, 0.3]])  # arrow shape
    angle = rot
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [
        mpl.path.Path.MOVETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.CLOSEPOLY,
    ]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale


class robot_simulation:
    def __init__(self, goal_xy):
        self.x, self.y, self.yaw = [0, 0, 0]  # x, y, theta
        self._goal_xy = goal_xy
        self._goal_heading_list = []
        self._xyyaw_list = []

    def robot_new_xyyaw(self, xyyaw):
        self.x, self.y, self.yaw = xyyaw

    def get_current_angle_for_target_facing(self):
        vector_robot_to_target = self._goal_xy - np.array([self.x, self.y])
        vector_robot_to_target = vector_robot_to_target / np.linalg.norm(
            vector_robot_to_target
        )
        vector_forward_robot = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        vector_forward_robot = vector_forward_robot / np.linalg.norm(
            vector_forward_robot
        )

        return vector_robot_to_target, vector_forward_robot

    def compute_angle(self):
        (
            vector_robot_to_target,
            vector_forward_robot,
        ) = self.get_current_angle_for_target_facing()
        x1 = (
            vector_robot_to_target[1] * vector_forward_robot[0]
            - vector_robot_to_target[0] * vector_forward_robot[1]
        )
        x2 = (
            vector_robot_to_target[0] * vector_forward_robot[0]
            + vector_robot_to_target[1] * vector_forward_robot[1]
        )
        rotation_delta = np.arctan2(x1, x2)
        goal_heading = wrap_heading(self.yaw + rotation_delta)
        return goal_heading

    def animate(self, xyyaw_list, save_name):
        self._goal_heading_list = []
        self._xyyaw_list = xyyaw_list
        for xyyaw in xyyaw_list:
            self.robot_new_xyyaw(xyyaw)
            self._goal_heading_list.append(self.compute_angle())

        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)

        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        self.ax.spines["left"].set_position("center")
        self.ax.spines["bottom"].set_position("center")

        # Eliminate upper and right axes
        self.ax.spines["right"].set_color("none")
        self.ax.spines["top"].set_color("none")

        # Show ticks in the left and lower axes only
        self.ax.xaxis.set_ticks_position("bottom")
        self.ax.yaxis.set_ticks_position("left")
        self.ax.set_xlim([-axis_size, axis_size])
        self.ax.set_ylim([-axis_size, axis_size])
        plt.gca().invert_xaxis()

        # make arrows and x,y labels
        self.ax.plot(
            (0),
            (0),
            ls="",
            marker="<",
            ms=10,
            color="k",
            transform=self.ax.get_yaxis_transform(),
            clip_on=False,
        )
        self.ax.plot(
            (0),
            (1),
            ls="",
            marker="^",
            ms=10,
            color="k",
            transform=self.ax.get_xaxis_transform(),
            clip_on=False,
        )
        self.ax.set_xlabel("$Y$", size=14, labelpad=-24, x=-0.02)
        self.ax.set_ylabel("$X$", size=14, labelpad=-21, y=1.02, rotation=0)

        ani = animation.FuncAnimation(
            fig=fig,
            func=self.animation_update,
            frames=len(self._goal_heading_list) + 1,
            interval=1,
        )
        ani.save(filename=save_name, writer="pillow")

    def animation_update(self, frame):
        # for each frame, update the data stored on each artist.
        if frame == 0:
            return
        goal_headings = self._goal_heading_list[:frame]
        xyyaws = self._xyyaw_list[:frame]
        # update the scatter plot:
        x_data = [v[1] for v in xyyaws]
        y_data = [v[0] for v in xyyaws]
        yaw_data = [v[2] for v in xyyaws]
        for i in range(len(x_data)):
            marker, scale = gen_arrow_head_marker(
                yaw_data[i] + np.pi / 2
            )  # for ploting, add 90 deg
            self.ax.scatter(
                x_data[i],
                y_data[i],
                marker=marker,
                s=(25 * scale) ** 1.5,
                c="b",
                label="robot pose",
            )
            self.ax.annotate(
                str(int(np.rad2deg(goal_headings[i]))), (x_data[i] - 0.35, y_data[i])
            )

        # Plot target
        self.ax.scatter(
            self._goal_xy[1],
            self._goal_xy[0],
            marker="*",
            s=25,
            c="red",
            label="target goal",
        )
        self.ax.legend(loc="upper right")


if __name__ == "__main__":
    """
                 X
                 0
                 ^
                 |
                 |
                 |
    Y pi/2<-------------- -pi/2
                 |
                 |
                 |
    """
    # Robot rotation in place
    print("======Robot ratates in place========")

    for i, goal_xy in enumerate([[1, -1], [-1, 1], [-1, -1], [1, 1]]):
        robot_sim = robot_simulation(goal_xy)
        animation_list = []
        for x in range(-4, 4, 2):
            for y in range(-4, 4, 2):
                for yaw in range(-180, 180, 45):
                    animation_list.append([x, y, np.deg2rad(yaw)])
        robot_sim.animate(animation_list, f"debug_{i}.gif")
