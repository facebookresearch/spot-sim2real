import heapq

import numpy as np


# Heuristic function: Using Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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

    while open_set:
        current = heapq.heappop(open_set)[1]

        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        # Explore neighbors
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Skip if the neighbor is out of bounds or occupied
            if (
                0 <= neighbor[0] < grid.shape[0]
                and 0 <= neighbor[1] < grid.shape[1]
                and grid[neighbor[0], neighbor[1]] == 0
            ):

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
