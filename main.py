import math
import asyncio
import numpy as np

from vision_recognition import run_recognition
from global_navigation import grid_search, display_map
from motion_control import follow_path_with_kalman


def build_navigation_map(occ_grid_coarse: np.ndarray) -> np.ndarray:
    """
    Builds a navigation map from the occupancy grid.
    Converts 1 (obstacle) to -1 and keeps 0 (free) as 0.
    
    Args:
        occ_grid_coarse: Occupancy grid (1 = obstacle, 0 = free)
    
    Returns:
        Navigation map (-1 = obstacle, 0 = free)
    """
    map_grid = occ_grid_coarse.copy().astype(np.int32)
    # Convert 1 (obstacle) to -1, keep 0 as 0
    map_grid[map_grid == 1] = -1
    return map_grid


def find_goal_position(map_grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    """
    Finds a valid goal position (opposite corner or free point).
    Used as fallback if the goal is not detected.
    
    Args:
        map_grid: Occupancy grid (-1 = obstacle, 0 = free)
        start: Start position (row, col)
    
    Returns:
        Goal position (row, col)
    """
    h, w = map_grid.shape
    # Try opposite corner first
    goal_candidates = [
        (h - 2, w - 2),  # Bottom-right corner
        (1, w - 2),      # Top-right corner
        (h - 2, 1),      # Bottom-left corner
        (h // 2, w - 2), # Middle right
        (h - 2, w // 2), # Middle bottom
    ]
    
    for goal in goal_candidates:
        if (0 <= goal[0] < h and 0 <= goal[1] < w and 
            map_grid[goal[0], goal[1]] != -1 and 
            goal != start):
            return goal
    
    # If no candidate works, search for any free point
    free_cells = np.argwhere(map_grid == 0)
    if len(free_cells) > 0:
        for cell in free_cells:
            goal = (int(cell[0]), int(cell[1]))
            if goal != start:
                return goal
    
    # Default: return opposite corner even if occupied
    return (max(1, h - 2), max(1, w - 2))


def main() -> None:
    # Step 1: Computer vision to detect Thymio, goal and obstacles
    result = run_recognition(cam_index=0, display=True)
    
    # Check if Thymio was detected
    if result.thymio_cell is None:
        print("Error: Thymio not detected. Cannot calculate path.")
        return
    
    # Step 2: Build navigation map from occupancy grid
    map_grid = build_navigation_map(result.occ_grid_coarse)
    
    # Step 3: Define start point (Thymio position)
    start = result.thymio_cell  # (row, col)
    
    # Step 4: Use detected goal (yellow square) or fallback
    if result.goal_cell is not None:
        goal = result.goal_cell  # (row, col)
    else:
        print("Warning: Goal (yellow square) not detected. Using automatic fallback goal.")
        goal = find_goal_position(map_grid, start)
    
    # Step 5: Calculate shortest path using A*
    path, explored, operation_count = grid_search(map_grid, start, goal)

    if path is None or len(path) == 0:
        print("No path found between start and goal.")
        display_map(map_grid, [], start, goal, explored)
        return

    print(f"Path found! Length: {len(path)} cells")
    # Step 6: Display map with complete path
    display_map(map_grid, path, start, goal, explored, keypoints=None)

    # Step 7: Follow path with Kalman filter + motion control
    resolution_m = 0.01  # Consistent with vision_recognition.run_recognition
    initial_theta = result.thymio_orientation
    board_height_m = 0.89
    
    # Path recalculation function for obstacle avoidance (uses current map)
    def recalculate_path(new_start: tuple[int, int], goal_cell: tuple[int, int], map_grid: np.ndarray) -> list | None:
        """Recalculates path from new start position with current map."""
        new_path, explored_local, operation_count_local = grid_search(map_grid, new_start, goal_cell)
        if new_path is None or len(new_path) == 0:
            display_map(map_grid, [], new_start, goal_cell, explored_local)
            return None
        return new_path
    
    print("Starting path following with Kalman filter and pure pursuit...")
    asyncio.run(
        follow_path_with_kalman(
            path_cells=path,
            start_cell=start,
            resolution_m=resolution_m,
            dt=0.05,
            base_speed=0.04,
            initial_theta=initial_theta,
            board_height_m=board_height_m,
            map_grid=map_grid,
            goal_cell=goal,
            recalculate_path_callback=recalculate_path,
        )
    )


if __name__ == "__main__":
    main()

