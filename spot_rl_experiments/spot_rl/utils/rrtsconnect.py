import math
import random

import numpy as np


# Creating a node class initializing position and assigning parent as None. Like Dijkstra, we set the initial cost to be positive infinity
class Node:
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.parent = None
        self.cost = float("inf")


# def heuristic(a, b,grid):
#     return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
def heuristic(a, b, grid):
    """
    In this heurestic , we will get nodes A being parent and B being the child and compute the L2 distance along with cost
    The heuristic considers a penalty for obstacles by calling the obstacle_cost function,
    which assesses how close the child node b is to any obstacles in the grid.This forms the informed heurestic along with distance
    """
    # Calculate the Euclidean distance
    distance = np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    # calculate penalty
    penalty = obstacle_cost(grid, (b.x, b.y))
    return distance + penalty


def is_valid_position(grid, position, radius=2.0):
    """
    Checking if the current position is valid - check using square proximity. For each point in the square area, checking
    whether the point is within the bounds of the grid and if the point is free.
    """
    x, y = position
    for dx in np.arange(-radius, radius + 0.5, 0.5):
        for dy in np.arange(-radius, radius + 0.5, 0.5):
            if np.sqrt(dx**2 + dy**2) <= radius:
                nx, ny = int(x + dx), int(y + dy)
                if (
                    not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1])
                    or grid[nx, ny] != 0
                ):
                    return False
    return True


def get_new_position(current, random_point, step_size):
    direction = np.array([random_point.x, random_point.y]) - np.array(
        [current.x, current.y]
    )
    if np.linalg.norm(direction) > step_size:
        direction = direction / np.linalg.norm(direction) * step_size
    return Node((current.x + direction[0], current.y + direction[1]))


def obstacle_cost(grid, position):
    """
    Constructing a 7x7 grid , which size can be customizable (used 7x7 toget better processing speed not trading off the accuracy)
    """
    x, y = position
    penalty = 0
    for i in range(-2, 5):
        for j in range(-2, 5):
            # getting position on grid wrt child point
            cx, cy = int(x + i), int(y + j)
            if 0 <= cx < grid.shape[0] and 0 <= cy < grid.shape[1]:
                if grid[cx, cy] != 0:
                    distance = np.sqrt(i**2 + j**2)
                    penalty += 1 / (
                        distance + 1.5
                    )  # distance to an obstacle decreases, the penalty increases
    return penalty


def rrt_connect_informed(
    grid,
    start,
    goal,
    max_iterations=1000,
    step_size=5.0,
    expand_range=5,
    goal_sample_rate=0.2,
):
    """
    This algorithm will build two trees, one starting from the start and the other from the goal,
    and attempts to connect them bi-directionally. The node expansion is informed by a heuristic that accounts
    for the cost associated with obstacles.

    max_iterations: Setting maximum number of times search needs to be done bi-directionally
    step_size : Max distance to extend a node in the tree during each iteration
    goal_sample_rate : Probablilty of random sampled node is the Goal node
    """
    # Define start amd goal
    start_tree = [Node(start)]
    goal_node = Node(goal)
    goal_tree = [goal_node]

    # Set intial cost as 0
    start_tree[0].cost = 0

    for i in range(max_iterations):
        # get a random point and check with sample rate
        if random.random() < goal_sample_rate:
            r_point = goal_node
        # get random point in the node near goal as Node((x,y))
        else:
            r_point = Node(
                (
                    random.randint(0, grid.shape[0] - 1),
                    random.randint(0, grid.shape[1] - 1),
                )
            )
        # Calling tree expansion from start node: if return true then we build path from start to goal
        if expand_tree(
            grid, start_tree, r_point, goal_node, expand_range, step_size, goal_tree
        ):
            return build_full_path(start_tree[-1], goal_tree[-1]), start_tree, goal_tree

        # Calling tree expansion from Goal node: if return true then we build path from start to goal
        if expand_tree(
            grid,
            goal_tree,
            r_point,
            start_tree[-1],
            expand_range,
            step_size,
            start_tree,
        ):
            return build_full_path(start_tree[-1], goal_tree[-1]), start_tree, goal_tree

    print(f"No path found after {max_iterations} iterations.")
    return [], [], []


def expand_tree(
    grid, tree, random_point, other_goal, expand_range, step_size, other_tree
):
    """
    Function calls to expand the tree by adding new node based on nearest node and heurestic.
    The it will generate a new position based on the nearest node and the specified step size, then
    it will create a new node , update its cost based on its path from parent.It will check whether
    new node is within range of the other goal and attempting to connect the two trees
    """
    near_node = None
    nearest_distance = float("inf")

    # Find the nearest node to the random_point
    for node in tree:
        distance = heuristic(node, random_point, grid)
        if distance < nearest_distance:
            nearest_distance = distance
            near_node = node

    # getting new position using the node obtained after heurestic check
    new_position = get_new_position(near_node, random_point, step_size)

    if is_valid_position(grid, (new_position.x, new_position.y)):
        new_node = Node((new_position.x, new_position.y))

        cost_t_come = heuristic(near_node, new_node, grid)
        new_node.parent = near_node
        new_node.cost = near_node.cost + cost_t_come  # Update cost
        tree.append(new_node)

        if is_goal_within_local_range(
            new_node, other_goal, expand_range, grid.shape[0], grid.shape[1]
        ):
            if connect_trees(new_node, other_tree, step_size, grid):
                return True

    return False


def is_goal_within_local_range(current, goal, expand_range, height, width):
    local_top = current.x - expand_range
    local_bottom = current.x + expand_range
    local_left = current.y - expand_range
    local_right = current.y + expand_range

    is_local_area_within_bounds = (
        0 <= local_top < height
        and 0 <= local_bottom < height
        and 0 <= local_left < width
        and 0 <= local_right < width
    )
    is_goal_in_local_area = (
        local_top <= goal.x <= local_bottom and local_left <= goal.y <= local_right
    )
    return is_goal_in_local_area and is_local_area_within_bounds


def connect_trees(node, tree, step_size, grid):
    """
    Trying to onnect two trees by extending the current node in random directions
    and checking for valid connections within the given step size.
    """
    for _ in range(20):  # Restricting to 20 iterations
        rand_dir = (random.uniform(-1, 1), random.uniform(-1, 1))
        check_pose = get_new_position(node, Node((rand_dir[0], rand_dir[1])), step_size)

        if is_valid_position(grid, (check_pose.x, check_pose.y)):
            n_node = None
            nearest_distance = float("inf")

            # Find the nearest node in the tree to the new position
            for n in tree:
                distance = heuristic(n, check_pose, grid)
                if distance < nearest_distance:
                    nearest_distance = distance
                    n_node = n

            if nearest_distance < step_size:
                check_pose.parent = n_node
                return True

    return False


# Helper function to calculate the Euclidean distance
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def add_point(current_point, path, distance_threshold=15.0):
    # Function to add points to the path while checking the distance
    if not path:
        path.append(current_point)
    else:
        last_point = path[-1]
        if distance(current_point, last_point) > distance_threshold:  # Check distance
            path.append(current_point)


def build_full_path(start_node, goal_node, distance_threshold=15.0):
    path = []
    visited = set()
    node = start_node
    # Logic to prevent looping which occurs in RRT often - so we track visited nodes
    while node is not None:
        current_point = (node.x, node.y)
        if current_point not in visited:
            add_point(current_point, path)
            visited.add(current_point)
        node = node.parent

    path.reverse()

    # Now check the goal_node path
    node = goal_node
    while node is not None:
        current_point = (node.x, node.y)
        # Logic to prevent looping which occurs in RRT often - so we track visited nodes
        if current_point not in visited:
            add_point(current_point, path)
            visited.add(current_point)
        node = node.parent

    refined_path = []
    for point in path:
        if not refined_path or distance(point, refined_path[-1]) > distance_threshold:
            refined_path.append(point)

    return refined_path
