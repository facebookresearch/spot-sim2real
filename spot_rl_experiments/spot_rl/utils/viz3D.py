import ast
import collections
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_object_data():
    """Read and parse object data from a text file."""
    data = []
    with open("/home/achuthan/Downloads/fremont_furniture.json", "r") as f:
        furniture_list = json.load(f)
    return furniture_list


def plot_objects_3d(data):
    """Plot the 3D positions of labeled objects."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    label_colors = {}
    color_cycle = plt.cm.tab20.colors  # 20 distinct colors
    # color_map = collections.defaultdict(
    #     lambda: color_cycle[len(label_colors) % len(color_cycle)]
    # )

    for i, (label, (x, y, z)) in enumerate(data):
        # Assign a unique color per label type
        if label not in label_colors:
            label_colors[label] = color_cycle[len(label_colors) % len(color_cycle)]

        ax.scatter(x, y, z, color=label_colors[label], s=40)
        ax.text(x, y, z, f"{i}: {label}", fontsize=7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Object Layout by Coordinates")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# === Run ===
data = read_object_data()
plot_objects_3d(data)
