import heapq

import numpy as np


# Heuristic function: Using Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_goal_within_local_range(current, goal, expand_range, height, width):
    local_top = current[0] - expand_range
    local_bottom = current[0] + expand_range
    local_left = current[1] - expand_range
    local_right = current[1] + expand_range
    is_local_area_within_bounds = (
        0 <= local_top < height
        and 0 <= local_bottom < height
        and 0 <= local_left < width
        and 0 <= local_right < width
    )
    is_goal_in_local_area = (
        local_top <= goal[0] <= local_bottom and local_left <= goal[1] <= local_right
    )
    return is_goal_in_local_area and is_local_area_within_bounds


# A* algorithm implementation
def astar(grid, start, goal):
    # Priority queue
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Dictionary to store the most efficient path to each node
    came_from = {}

    # g_score stores the cost of getting to each node from the start
    g_score = {start: 0}

    # f_score stores the estimated cost from start to goal through each node
    f_score = {start: heuristic(start, goal)}

    # Directions for moving in the grid: up, down, left, right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    expand_range = (
        2  # almost 0.6 m length & width, donnot change works well emperically
    )
    while open_set:
        current = heapq.heappop(open_set)[1]

        # If the goal is reached, reconstruct the path
        if is_goal_within_local_range(current, goal, expand_range, *grid.shape[:2]):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        # Explore neighbors
        for direction in directions:
            neighbor_center = (current[0] + direction[0], current[1] + direction[1])

            # each neightbour check expand_range X expand_range pixel around it for occupancy
            # Skip if the neighbor is out of bounds or occupied
            is_area_free = grid[neighbor_center[0], neighbor_center[1]] == 0
            is_area_within_bounds = (
                0 <= neighbor_center[0] < grid.shape[0]
                and 0 <= neighbor_center[1] < grid.shape[1]
            )

            if is_area_free and is_area_within_bounds:
                local_top = neighbor_center[0] - expand_range
                local_bottom = neighbor_center[0] + expand_range
                local_left = neighbor_center[1] - expand_range
                local_right = neighbor_center[1] + expand_range
                is_local_area_within_bounds = (
                    0 <= local_top < grid.shape[0]
                    and 0 <= local_bottom < grid.shape[0]
                    and 0 <= local_left < grid.shape[1]
                    and 0 <= local_right < grid.shape[1]
                )
                is_local_area_free = (
                    grid[
                        local_top : local_bottom + 1, local_left : local_right + 1
                    ].sum()
                    == 0
                    if is_local_area_within_bounds
                    else False
                )
                is_area_free = is_area_free and is_local_area_free
                is_area_within_bounds = (
                    is_area_within_bounds and is_local_area_within_bounds
                )
            neighbor = neighbor_center
            if is_area_within_bounds and is_area_free:

                # Tentative g_score for neighbor
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return empty path if no path found


if __name__ == "__main__":
    # Example usage
    grid = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    start = (0, 0)  # Starting point A
    goal = (4, 4)  # Goal point B

    path = astar(grid, start, goal)

    print("Path from A to B:", path)
