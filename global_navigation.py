import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop


def heuristic(a, b):
    """
    Chebyshev distance between two cells a and b.
    Used as h(n) function in A*.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dy if dx <= dy else dx


def display_map(map_grid, path, start, goal, explored, keypoints=None):
    """
    Displays the grid with:
      - black: obstacles (-1)
      - white: free (0)
      - grey: explored cells
      - blue: path
      - green: start
      - red: goal
    """
    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red', 'grey', 'yellow'])
    map_display = np.zeros_like(map_grid, dtype=object)

    # Free space / obstacles
    map_display[map_grid == -1] = 'black'
    map_display[map_grid == 0] = 'white'

    # Explored cells
    for position in explored:
        if map_display[tuple(position)] == 'white':
            map_display[tuple(position)] = 'grey'

    # Complete path (blue)
    for position in path:
        if map_display[position[0], position[1]] in ['white', 'grey']:
            map_display[position[0], position[1]] = 'blue'

    # Simplified keypoints (yellow) over path
    if keypoints is not None:
        for r, c in keypoints:
            map_display[r, c] = 'yellow'

    # Start and goal
    map_display[start[0], start[1]] = 'green'
    map_display[goal[0], goal[1]] = 'red'

    color_mapping = {
        'white': 0,
        'black': 1,
        'blue': 2,
        'green': 3,
        'red': 4,
        'grey': 5,
        'yellow': 6,
    }
    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(map_numeric_display, cmap=cmap)
    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    plt.show()


def grid_search(map_grid, S, G):
    """
    A* on a grid with Chebyshev distance and penalty for diagonals.

    Args:
        map_grid: np.ndarray with -1 for obstacle, 0 (or >0) for free/cost.
        S: tuple (row, col) start.
        G: tuple (row, col) goal.

    Returns:
        path: list of cells (row, col) from start to goal (or None).
        explored: set of explored cells.
        operation_count: number of operations performed.
    """
    came_from = {}
    g_costs = {S: 0}
    explored = set()
    operation_count = 0

    # (f_cost, g_cost, position)
    open_set = [(heuristic(S, G), 0, S)]
    current_pos = S

    while open_set:
        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        explored.add(current_pos)

        if current_pos == G:
            break

        # 8 neighbors (4 directions + diagonals)
        neighbors = [
            (current_pos[0] - 1, current_pos[1]),      # Up
            (current_pos[0] + 1, current_pos[1]),      # Down
            (current_pos[0],     current_pos[1] - 1),  # Left
            (current_pos[0],     current_pos[1] + 1),  # Right
            (current_pos[0] - 1, current_pos[1] - 1),
            (current_pos[0] - 1, current_pos[1] + 1),
            (current_pos[0] + 1, current_pos[1] + 1),
            (current_pos[0] + 1, current_pos[1] - 1),
        ]

        diagonal_neighbors = [
            (current_pos[0] - 1, current_pos[1] - 1),
            (current_pos[0] - 1, current_pos[1] + 1),
            (current_pos[0] + 1, current_pos[1] + 1),
            (current_pos[0] + 1, current_pos[1] - 1),
        ]

        for neighbor in neighbors:
            # Grid bounds
            if 0 <= neighbor[0] < map_grid.shape[0] and 0 <= neighbor[1] < map_grid.shape[1]:
                # Not obstacle
                if map_grid[neighbor[0], neighbor[1]] != -1:
                    tentative_g_cost = current_g_cost + 1 + map_grid[neighbor[0], neighbor[1]]

                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g_cost
                        came_from[neighbor] = current_pos
                        operation_count += 1

                        f_cost = tentative_g_cost + heuristic(neighbor, G)

                        # Penalty for diagonals
                        if neighbor in diagonal_neighbors:
                            f_cost += 2

                        heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # Path reconstruction
    if current_pos == G:
        path = []
        while current_pos != S:
            path.append(current_pos)
            current_pos = came_from[current_pos]
        path.append(S)
        path.reverse()
        return path, explored, operation_count

    # No path found
    return None, explored, operation_count