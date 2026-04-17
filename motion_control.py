import math
import time
from typing import List, Tuple, Callable

import numpy as np
from tdmclient import ClientAsync

from kalmanpy import kalman_filter
from vision_tracker import ThymioPoseTracker
from pure_pursuit import pure_pursuit_step
from global_navigation import grid_search

# Real-time visualization (map + estimated pose)
import matplotlib.pyplot as plt


Pose = Tuple[float, float, float]   # (x, y, theta) in meters / radians
GridCell = Tuple[int, int]         # (row, col) in grid


def grid_path_to_world_xy(
    path: List[GridCell], 
    cell_size_m: float,
    grid_height: int | None = None,
) -> List[Tuple[float, float]]:
    """
    Converts a path of cells (row, col) to points (x, y) in meters.

    Axis convention:
      - origin at bottom-left,
      - x-axis to the right,
      - y-axis upward.
    
    Args:
        path: List of cells (row, col)
        cell_size_m: Size of a cell in meters
        grid_height: Grid height in pixels (if None, estimate from path)
    """
    if not path:
        return []

    # Use actual image height if provided, otherwise estimate from path
    if grid_height is not None:
        grid_h = grid_height
    else:
        # Fallback: estimate from path (old behavior)
        max_row = max(r for r, _ in path)
        grid_h = max_row + 1

    world_path_xy: List[Tuple[float, float]] = []
    for r, c in path:
        x = (c + 0.5) * cell_size_m
        # Invert vertical axis to have origin at bottom-left
        y = (grid_h - 1 - r + 0.5) * cell_size_m
        world_path_xy.append((x, y))
    return world_path_xy


async def follow_path_with_kalman(
    path_cells: List[GridCell],
    start_cell: GridCell,
    resolution_m: float,
    dt: float = 0.1,
    base_speed: float = 0.03,
    wheel_radius: float = 0.023,
    axle_length: float = 0.093,
    initial_theta: float | None = None,
    cam_index: int = 0,
    lookahead_dist: float | None = None,
    board_height_m: float = 0.89,
    debug: bool = True,
    map_grid: np.ndarray | None = None,
    goal_cell: GridCell | None = None,
    recalculate_path_callback: Callable[[Tuple[int, int], Tuple[int, int], np.ndarray], List[Tuple[int, int]] | None] | None = None,
) -> None:
    """
    Path following based on:
      - complete path (all cells) converted to (x, y) metric coordinates,
      - pure pursuit algorithm,
      - pose measured by vision at each iteration.
    """
    if not path_cells:
        print("Error: Empty path, nothing to follow.")
        return

    # Convert path to metric coordinates
    # Use actual image height for consistency with vision_tracker
    height_px = max(2, int(board_height_m / resolution_m))
    path_xy = grid_path_to_world_xy(
        path_cells, 
        cell_size_m=resolution_m,
        grid_height=height_px,
    )

    # Default lookahead distance: several cells to aim further ahead
    if lookahead_dist is None:
        lookahead_dist = 9.0 * resolution_m

    # End tolerance: stop when close to last point
    goal_x, goal_y = path_xy[-1]
    goal_tolerance = 5.0 * resolution_m
    # Store initial goal position in metric coordinates
    initial_goal_m = (goal_x, goal_y)
    # Threshold for goal movement to trigger path recalculation (pixels -> m)
    goal_move_threshold_px = 8
    goal_move_threshold_m = goal_move_threshold_px * resolution_m

    def _build_navigation_map(occ_grid_coarse: np.ndarray) -> np.ndarray:
        """Converts occupancy grid (1=obstacle, 0=free) to navigation map (-1=obstacle, 0=free)."""
        nav = occ_grid_coarse.copy().astype(np.int32)
        nav[nav == 1] = -1
        return nav
    
    # Index of last reached point for pure pursuit
    lf_index = 0

    # Single connection to Thymio for entire loop
    print("Connecting to Thymio...")
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    print("Connected to Thymio.")

    # Vision tracker: camera open continuously
    tracker = ThymioPoseTracker(
        cam_index=cam_index,
        board_width_m=0.84,
        board_height_m=0.89,
        resolution_m=resolution_m,
    )

    # --- Kalman: state and covariances ---
    # State [x, y, theta]^T initialized at CENTER of start cell
    # (geometrically consistent with path), with orientation from vision
    first_pose = tracker.get_pose_m()
    if first_pose is None:
        print("Error: Cannot initialize pose from vision.")
        await node.unlock()
        tracker.release()
        client.close()
        return

    x = np.zeros((3, 1), dtype=float)
    # Position = center of start cell (same as path_xy[0])
    x[0, 0] = path_xy[0][0]
    x[1, 0] = path_xy[0][1]
    # Orientation = angle measured by vision
    x[2, 0] = first_pose[2]
    
    # Initial Kalman filter covariances (pose-only, 3x3)
    P = np.eye(3) * 1e-3
    dt = 0.05
    sigma_v2 = 1.514387409522633e-4  # Measured speed variance in mm
    sigma_omega = 2*sigma_v2/0.093**2
    q_theta = sigma_omega*dt**2
    q_pos = sigma_v2 * dt**2
    Q = np.zeros((3, 3))
    Q[0, 0] = q_pos
    Q[1, 1] = q_pos 
    Q[2, 2] = q_theta
    # Optional small noise on v, w if you want them to drift
    # Measurement covariance: trust vision more on theta
    R = np.diag(
        [
            0.1e-6, 
            0.1e-6,    
            (0.3 * math.pi / 180.0) ** 2,  # ~2° typical error on angle
        ]
    )

    # Control mode: first ALIGN (rotate in place), then TRACK (pure pursuit), or AVOID (obstacle avoidance)
    mode = "ALIGN"

    # Obstacle avoidance parameters
    obstThrL = 10  # Low threshold to exit avoidance mode
    obstThrH = 40  # High threshold to enter avoidance mode

    # Kidnapping detection via prox.ground.delta:
    # - on GROUND, prox.ground.delta is HIGH
    # - when LIFTING robot, this value becomes very LOW
    kidnapped = False
    # Thresholds on SUM of both sensors (delta[0] + delta[1])
    ground_lift_threshold = 50    # Below this -> robot considered lifted
    ground_place_threshold = 400  # Above this -> robot placed (hysteresis)

    # Parameters for improved avoidance maneuver
    avoid_forward_time = 4.5  # Time (s) to advance after turning
    avoid_state = "TURN_AWAY"  # States: TURN_AWAY, FORWARD, CHECK
    avoid_state_timer = 0.0  # Timer for avoidance phases
    avoid_direction = 0  # -1 = left, +1 = right
    prev_loop_time = None  # Previous loop time to calculate dt

    # Real-time visualization (map + estimated pose + pure pursuit target)
    viz_enabled = debug
    if viz_enabled:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # Plot global path (dynamically modifiable line)
        path_xs, path_ys = zip(*path_xy)
        path_line, = ax.plot(path_xs, path_ys, "b.-", label="Path")

        # Map limits slightly expanded around path
        margin = 0.05  # 5 cm margin
        ax.set_xlim(min(path_xs) - margin, max(path_xs) + margin)
        ax.set_ylim(min(path_ys) - margin, max(path_ys) + margin)

        # Estimated Thymio point
        robot_point, = ax.plot([], [], "ro", label="Thymio (EKF)")
        # Orientation vector
        heading_line, = ax.plot([], [], "r-", linewidth=2, label="Orientation")
        # Pure pursuit target point
        target_point, = ax.plot([], [], "gx", markersize=8, label="PP Target")

        ax.legend(loc="best")

    try:
        while True:
            loop_start = time.time()
            
            # Calculate dt for this iteration (use previous loop time)
            if prev_loop_time is not None:
                dt_iteration = time.time() - prev_loop_time
            else:
                dt_iteration = dt  # For first iteration, use nominal dt
            prev_loop_time = time.time()
            
            # Update avoidance timer BEFORE calculating commands
            if mode == "AVOID":
                avoid_state_timer += dt_iteration

            # --- Read horizontal proximity sensors ---
            # Sensors are in "prox.horizontal" as a list of 5 values [0,1,2,3,4]
            obst = [0, 0, 0, 0, 0]
            try:
                await node.wait_for_variables({"prox.horizontal"})
                prox_horizontal_val = node["prox.horizontal"]
                
                # ArrayCache can be accessed as list with [index] or converted to list
                if hasattr(prox_horizontal_val, '__getitem__'):
                    try:
                        for i in range(5):
                            obst[i] = int(prox_horizontal_val[i])
                    except (IndexError, TypeError):
                        try:
                            prox_list = list(prox_horizontal_val)
                            for i in range(min(5, len(prox_list))):
                                obst[i] = int(prox_list[i])
                        except Exception:
                            pass
                elif isinstance(prox_horizontal_val, (list, tuple)):
                    for i in range(min(5, len(prox_horizontal_val))):
                        obst[i] = int(prox_horizontal_val[i])
            except Exception:
                # If error, keep values at 0
                pass

            # --- Read ground proximity sensors (prox.ground.delta) for kidnapping ---
            ground_delta = [0, 0]
            try:
                await node.wait_for_variables({"prox.ground.delta"})
                ground_val = node["prox.ground.delta"]
                # Robust conversion (ArrayCache, list, etc.)
                if hasattr(ground_val, "__getitem__"):
                    try:
                        for i in range(min(2, len(ground_val))):
                            ground_delta[i] = int(ground_val[i])
                    except Exception:
                        try:
                            g_list = list(ground_val)
                            for i in range(min(2, len(g_list))):
                                ground_delta[i] = int(g_list[i])
                        except Exception:
                            pass
                elif isinstance(ground_val, (list, tuple)):
                    for i in range(min(2, len(ground_val))):
                        ground_delta[i] = int(ground_val[i])
            except Exception:
                pass

            # Obstacle detection: switch to AVOID mode if obstacle detected
            sum_left = obst[0] + obst[1]  # Left sensors (0 and 1)
            sum_right = obst[3] + obst[4]  # Right sensors (3 and 4)
            
            if mode in ["ALIGN", "TRACK"]:
                if sum_left > obstThrH or sum_right > obstThrH:
                    mode = "AVOID"
                    # Determine obstacle direction for maneuver
                    if sum_left > sum_right:
                        avoid_direction = -1  # Obstacle on left, turn right
                    else:
                        avoid_direction = +1  # Obstacle on right, turn left
                    avoid_state = "TURN_AWAY"
                    avoid_state_timer = 0.0
            
            # Exit AVOID mode: when obstacle disappears (detected in CHECK phase), recalculate path
            elif mode == "AVOID" and avoid_state == "CHECK":
                if sum_left < obstThrL and sum_right < obstThrL:
                    # Get current position from vision
                    pose_vis_recalc = tracker.get_pose_m()
                    if pose_vis_recalc is not None and recalculate_path_callback is not None and goal_cell is not None and map_grid is not None:
                        x_vis_recalc, y_vis_recalc, th_vis_recalc = pose_vis_recalc
                        # Convert position to grid cell
                        height_px = max(2, int(board_height_m / resolution_m))
                        grid_h = height_px
                        new_start_col = int((x_vis_recalc / resolution_m) - 0.5)
                        new_start_row = int(grid_h - 1 - (y_vis_recalc / resolution_m) + 0.5)
                        new_start_cell = (new_start_row, new_start_col)
                        
                        # Recalculate path
                        new_path = recalculate_path_callback(new_start_cell, goal_cell, map_grid)
                        if new_path is not None and len(new_path) > 0:
                            path_cells = new_path
                            path_xy = grid_path_to_world_xy(
                                path_cells,
                                cell_size_m=resolution_m,
                                grid_height=height_px,
                            )
                            goal_x, goal_y = path_xy[-1]
                            lf_index = 0
                            # Reset Kalman state with vision (pose-only)
                            x[0, 0] = x_vis_recalc
                            x[1, 0] = y_vis_recalc
                            x[2, 0] = th_vis_recalc
                            P = np.eye(3) * 1e-3
                            # Restart in ALIGN mode
                            mode = "ALIGN"
                            # Reset avoidance variables to allow new avoidance
                            avoid_state = "TURN_AWAY"
                            avoid_state_timer = 0.0
                            avoid_direction = 0
                        else:
                            # Stay in AVOID mode but return to TURN_AWAY
                            avoid_state = "TURN_AWAY"
                            avoid_state_timer = 0.0
                    else:
                        # If no callback or no vision, just return to TRACK
                        mode = "TRACK"
                        # Reset avoidance variables to allow new avoidance
                        avoid_state = "TURN_AWAY"
                        avoid_state_timer = 0.0
                        avoid_direction = 0

            # --- Current estimated pose (Kalman output) used by controller ---
            x_m = x[0, 0]
            y_m = x[1, 0]
            theta_rad = x[2, 0]

            # --- Mobile goal: if goal square moves, recalculate path AND obstacles ---
            if hasattr(tracker, "last_goal_m") and tracker.last_goal_m is not None:
                gx_m, gy_m = tracker.last_goal_m
                dist_goal_motion = math.hypot(gx_m - initial_goal_m[0], gy_m - initial_goal_m[1])
                if dist_goal_motion > goal_move_threshold_m:
                    snap = tracker.get_navigation_snapshot(red_cell_size_px=1)
                    if snap is not None:
                        occ_grid_coarse_new, thymio_cell_new, goal_cell_new = snap
                        # Update occupancy map and navigation map
                        if occ_grid_coarse_new is not None:
                            map_grid = _build_navigation_map(occ_grid_coarse_new)
                            # Update goal_cell if we have new goal cell
                            if goal_cell_new is not None:
                                goal_cell = goal_cell_new
                            # Start = Thymio cell if available, otherwise from EKF state
                            if thymio_cell_new is not None:
                                new_start_cell = thymio_cell_new
                            else:
                                grid_h = occ_grid_coarse_new.shape[0]
                                new_start_col = int((x_m / resolution_m) - 0.5)
                                new_start_row = int(grid_h - 1 - (y_m / resolution_m) + 0.5)
                                new_start_cell = (new_start_row, new_start_col)
                            # Calculate new path with updated obstacles
                            new_path, _, op_count_local = grid_search(map_grid, new_start_cell, goal_cell)
                            if new_path is not None and len(new_path) > 0:
                                path_cells = new_path
                                # Recalculate path_xy with new map (height possibly same)
                                height_px = max(2, int(board_height_m / resolution_m))
                                path_xy = grid_path_to_world_xy(
                                    path_cells,
                                    cell_size_m=resolution_m,
                                    grid_height=height_px,
                                )
                                goal_x, goal_y = path_xy[-1]
                                initial_goal_m = (goal_x, goal_y)
                                lf_index = 0
                                # Update path line in visualization
                                if viz_enabled:
                                    new_xs, new_ys = zip(*path_xy)
                                    path_line.set_data(new_xs, new_ys)
                                    ax.set_xlim(min(new_xs) - margin, max(new_xs) + margin)
                                    ax.set_ylim(min(new_ys) - margin, max(new_ys) + margin)

            # --- Pure pursuit: determine target point and angle error ---
            # (except in AVOID mode where we don't use path)
            if mode != "AVOID":
                heading_deg = math.degrees(theta_rad)
                
                goal_pt, lf_index, turn_error_deg = pure_pursuit_step(
                    path=path_xy,
                    current_pos=(x_m, y_m, heading_deg),
                    lookAheadDis=lookahead_dist,
                    LFindex=lf_index,
                )
                heading_error_pp = math.radians(turn_error_deg)

                # Current distance to goal (in estimated state)
                dist_to_goal = math.hypot(goal_x - x_m, goal_y - y_m)
            else:
                # In AVOID mode, don't use pure pursuit
                turn_error_deg = 0.0
                heading_error_pp = 0.0
                dist_to_goal = float('inf')  # No goal in avoidance mode
                goal_pt = None

            # --- Controller (v, w) with alignment phase, avoidance and kidnapping ---
            # max_w: angular velocity saturation in path following
            # Keep moderate saturation to avoid too abrupt rotations
            max_w = 0.5  # rad/s

            desired_angle = None

            # If robot is considered "kidnapped" (lifted), stop all movement
            # and wait to be placed to recalculate path (without restarting camera)
            ground_sum = sum(ground_delta)
            # Here, ground_sum is HIGH when seeing ground, LOW when lifting
            if not kidnapped and ground_sum < ground_lift_threshold:
                kidnapped = True
                print("Robot lifted, stopping motors.")
                v = 0.0
                w = 0.0
                v_l_cmd = 0.0
                v_r_cmd = 0.0
                await send_motor_speeds(node, 0.0, 0.0, wheel_radius)
                # Stay in loop, but with v=w=0 while in air

            # If we were kidnapped and see ground again, recalculate path + obstacles locally
            if kidnapped and ground_sum > ground_place_threshold:
                print("Robot placed, recalculating path + obstacles.")
                snap = tracker.get_navigation_snapshot(red_cell_size_px=1)
                if snap is not None:
                    occ_grid_coarse_new, thymio_cell_new, goal_cell_new = snap
                    if occ_grid_coarse_new is not None:
                        map_grid = _build_navigation_map(occ_grid_coarse_new)
                        # Update goal_cell if we have new goal cell
                        if goal_cell_new is not None:
                            goal_cell = goal_cell_new
                        # Start = Thymio cell if available, otherwise from vision pose
                        pose_vis_recalc = tracker.get_pose_m()
                        if thymio_cell_new is not None:
                            new_start_cell = thymio_cell_new
                        elif pose_vis_recalc is not None:
                            x_vis_recalc, y_vis_recalc, th_vis_recalc = pose_vis_recalc
                            grid_h = occ_grid_coarse_new.shape[0]
                            new_start_col = int((x_vis_recalc / resolution_m) - 0.5)
                            new_start_row = int(grid_h - 1 - (y_vis_recalc / resolution_m) + 0.5)
                            new_start_cell = (new_start_row, new_start_col)
                            # Reset Kalman state with new pose (pose-only)
                            x[0, 0] = x_vis_recalc
                            x[1, 0] = y_vis_recalc
                            x[2, 0] = th_vis_recalc
                            P = np.eye(3) * 1e-3
                        else:
                            new_start_cell = start_cell

                        new_path, _, op_count_kidnap = grid_search(map_grid, new_start_cell, goal_cell)
                        if new_path is not None and len(new_path) > 0:
                            path_cells = new_path
                            height_px = max(2, int(board_height_m / resolution_m))
                            path_xy = grid_path_to_world_xy(
                                path_cells,
                                cell_size_m=resolution_m,
                                grid_height=height_px,
                            )
                            goal_x, goal_y = path_xy[-1]
                            initial_goal_m = (goal_x, goal_y)
                            lf_index = 0
                            # Update path line in visualization
                            if viz_enabled:
                                new_xs, new_ys = zip(*path_xy)
                                path_line.set_data(new_xs, new_ys)
                                ax.set_xlim(min(new_xs) - margin, max(new_xs) + margin)
                                ax.set_ylim(min(new_ys) - margin, max(new_ys) + margin)
                kidnapped = False

            if kidnapped:
                # Force v, w to zero; wheel speeds already set to 0
                v = 0.0
                w = 0.0
            elif mode == "AVOID":
                # Improved avoidance mode with loop:
                # Phase 1: TURN_AWAY - Turn in opposite direction to obstacle
                # Phase 2: FORWARD - Advance for 3 seconds
                # Phase 3: CHECK - Check if obstacle is still there
                #   - If yes → return to TURN_AWAY (loop)
                #   - If no → recalculate path and exit AVOID mode
                
                # Timer already updated at start of loop
                
                if avoid_state == "TURN_AWAY":
                    # Turn in opposite direction to obstacle
                    # Stronger rotation gain to give real "steering" effect
                    turn_speed = 0.8  # rad/s
                    v = 0.0  # No advance during rotation
                    w = avoid_direction * turn_speed  # Turn right if obstacle on left, and vice versa
                    
                    # Move to next phase after certain time
                    if avoid_state_timer >= 0.4:  # Turn for 0.4s
                        avoid_state = "FORWARD"
                        avoid_state_timer = 0.0
                
                elif avoid_state == "FORWARD":
                    # Advance straight for 3 seconds
                    # Slightly faster to move away from obstacle
                    forward_speed = 0.04  # m/s
                    v = forward_speed
                    w = 0.0  # No rotation
                    
                    # Move to CHECK phase after 3 seconds
                    if avoid_state_timer >= avoid_forward_time:
                        avoid_state = "CHECK"
                        avoid_state_timer = 0.0
                
                elif avoid_state == "CHECK":
                    # Stop robot and check if obstacle is still there
                    v = 0.0
                    w = 0.0
                    
                    # Check sensors (stay in CHECK for one iteration to allow time to check)
                    if avoid_state_timer < 0.1:  # Wait a bit to stabilize reading
                        pass
                    elif sum_left > obstThrH or sum_right > obstThrH:
                        # Obstacle still there, return to TURN_AWAY for new avoidance
                        avoid_state = "TURN_AWAY"
                        avoid_state_timer = 0.0
                        # Redetermine obstacle direction
                        if sum_left > sum_right:
                            avoid_direction = -1  # Obstacle on left, turn right
                        else:
                            avoid_direction = +1  # Obstacle on right, turn left
                    # If obstacle disappeared, exit AVOID mode (path recalculation done above)
                
                # Convert (v, w) to wheel speeds
                v_l_cmd = v - (axle_length / 2.0) * w
                v_r_cmd = v + (axle_length / 2.0) * w
                
            elif mode == "ALIGN":
                # Rotate in place to align with direction of first path segment
                # To avoid target moving during ALIGN due to small estimated position drifts,
                # use ONLY start position (center of start cell) as geometric origin
                x_start, y_start = path_xy[0]
                if len(path_xy) >= 2:
                    x_wp, y_wp = path_xy[1]
                else:
                    x_wp, y_wp = path_xy[0]
                desired_angle = math.atan2(y_wp - y_start, x_wp - x_start)
                heading_error_align = desired_angle - theta_rad
                # Normalize angle to [-pi, pi]
                while heading_error_align > math.pi:
                    heading_error_align -= 2 * math.pi
                while heading_error_align < -math.pi:
                    heading_error_align += 2 * math.pi

                # Gain that decreases near target to avoid overshoot
                # Increased gains to rotate a bit faster in ALIGN
                Kp_align_max = 0.8  # Faster rotation far from target
                Kp_align_min = 0.20  # Slightly stronger near target
                err_norm = min(abs(heading_error_align) / math.radians(45.0), 1.0)
                Kp_align = Kp_align_min + (Kp_align_max - Kp_align_min) * err_norm

                v = 0.0
                w = Kp_align * heading_error_align

                # When angle is small, switch to path following
                if abs(heading_error_align) < math.radians(12.0):
                    mode = "TRACK"
                    # Complete Kalman state reset with vision to start
                    # with real physical robot position
                    pose_vis_reset = tracker.get_pose_m()
                    if pose_vis_reset is not None:
                        x_vis_reset, y_vis_reset, th_vis_reset = pose_vis_reset
                        x[0, 0] = x_vis_reset  # Real x position
                        x[1, 0] = y_vis_reset  # Real y position
                        x[2, 0] = th_vis_reset  # Real orientation
                        # Reset P (pose-only) to indicate we trust this measurement
                        P = np.eye(3) * 1e-3
                    # Reset pure pursuit index to restart from path beginning
                    lf_index = 0
            else:
                # TRACK mode: pure pursuit + slow advance
                # Linear velocity proportional to distance, bounded by base_speed
                Kp_v = 0.4
                v = min(base_speed, Kp_v * dist_to_goal)
                if dist_to_goal < goal_tolerance:
                    v = 0.0

                # Adapt speed to angle error:
                # - if error > 80°, only rotate (v = 0)
                # - if 30° < error <= 80°, reduce speed
                abs_err_deg = abs(turn_error_deg)
                if abs_err_deg > 80.0:
                    v = 0.0
                elif abs_err_deg > 30.0:
                    v *= 0.4

                # Angular correction based on pure pursuit
                # Moderate gain to avoid oscillations
                Kp_angle = 1
                w = Kp_angle * heading_error_pp

            # Strict saturation on w to avoid too violent rotations (except in AVOID mode)
            if mode != "AVOID":
                w = max(-max_w, min(max_w, w))
            # Conversion (v, w) -> linear wheel speeds
            v_r_cmd = v + (axle_length / 2.0) * w
            v_l_cmd = v - (axle_length / 2.0) * w
            # In AVOID mode, v_l_cmd and v_r_cmd already calculated above

            # Send to Thymio motors with commanded speeds
            await send_motor_speeds(node, v_l_cmd, v_r_cmd, wheel_radius)

            # --- Read actual wheel speeds for Kalman prediction ---
            # Same logic as proximity sensors: properly convert
            # tdmclient values (ArrayCache, list, int, etc.) to integers
            await node.wait_for_variables({"motor.left.speed", "motor.right.speed"})
            left_speed_val = node["motor.left.speed"]
            right_speed_val = node["motor.right.speed"]

            # Robust handling of different possible types (ArrayCache, list, int, etc.)
            def _extract_first_int(value):
                if isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        return 0
                    return int(value[0])
                try:
                    return int(value)
                except Exception:
                    return 0

            left_speed_raw = _extract_first_int(left_speed_val)
            right_speed_raw = _extract_first_int(right_speed_val)

            thymio_factor = 0.00040816326530612246  # m/s per motor unit (recalibrate if necessary)
            v_l_meas = left_speed_raw * thymio_factor
            v_r_meas = right_speed_raw * thymio_factor

            # If sensors return (quasi) 0 when we clearly sent
            # non-zero command, fall back to commanded speeds
            # Test on command norm to avoid sending during stop
            eps_speed = 0.001  # ~1 mm/s
            if abs(v_l_meas) < eps_speed and abs(v_r_meas) < eps_speed:
                cmd_norm = abs(v_l_cmd) + abs(v_r_cmd)
                if cmd_norm > eps_speed:
                    v_l_meas = v_l_cmd
                    v_r_meas = v_r_cmd

            # In ALIGN mode, assume robot rotates in place:
            # cancel linear component in model (no x/y displacement)
            # In AVOID mode, use commanded speeds as measurement
            if mode == "ALIGN":
                v_l_meas = -v_r_meas  # v_cmd = (v_r + v_l)/2 ≈ 0
            elif mode == "AVOID":
                # In AVOID mode, use commanded speeds as measurement
                v_l_meas = v_l_cmd
                v_r_meas = v_r_cmd

            # Command vector for Kalman based on MEASURED speeds
            u_vec = np.array([[v_r_meas], [v_l_meas], [0.0]])

            # Current vision measurement for Kalman update
            pose_vis = tracker.get_pose_m()
            if pose_vis is None:
                # If vision is momentarily unavailable, just skip correction
                y_vec = None
                th_meas = None
                # Reduce odometric integration when we no longer have camera
                v_l_meas *= 0.8
                v_r_meas *= 0.8
            else:
                x_meas, y_meas, th_meas = pose_vis
                y_vec = np.array([[x_meas], [y_meas], [th_meas]])

            # Real time step
            elapsed_for_dt = time.time() - loop_start
            dt_real = max(0.001, min(elapsed_for_dt, dt * 2.0))

            # Kalman filter: always do odometric prediction
            # If y_vec is None (camera hidden / vision unavailable), Kalman
            # simply skips correction and uses only wheels for pose
            xk_new, yk_new, th_new, v_new, w_new, P = kalman_filter(
                x=x,
                u=u_vec,
                y=y_vec,
                P=P,
                Q=Q,
                R=R,
                dt=dt_real,
                r=wheel_radius,
                L=axle_length,
            )

            # Update estimated state (pose-only)
            x[0, 0] = xk_new
            x[1, 0] = yk_new
            x[2, 0] = th_new

            # Real-time visualization update
            if viz_enabled:
                # Update real-time visualization (EKF pose + orientation + PP target)
                arrow_len = 0.12  # Orientation vector length (m)
                # Thymio point (EKF state)
                robot_point.set_data([x_m], [y_m])
                # Orientation segment
                hx = x_m + arrow_len * math.cos(theta_rad)
                hy = y_m + arrow_len * math.sin(theta_rad)
                heading_line.set_data([x_m, hx], [y_m, hy])
                # Pure pursuit target point (if available)
                if goal_pt is not None:
                    target_point.set_data([goal_pt[0]], [goal_pt[1]])
                else:
                    target_point.set_data([], [])

                fig.canvas.draw_idle()
                plt.pause(0.001)

            # Stop condition: close to final goal (from estimated state)
            # (except in AVOID mode where we continue avoidance)
            if mode != "AVOID" and dist_to_goal < goal_tolerance:
                print("Goal reached, stopping motors.")
                await send_motor_speeds(node, 0.0, 0.0, wheel_radius)
                break

            # Wait until next cycle (based on nominal dt)
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # Safety: stop motors, release robot and close camera
        await send_motor_speeds(node, 0.0, 0.0, wheel_radius)
        await node.unlock()
        try:
            client.close()
        except Exception:
            pass
        tracker.release()
        print("Thymio disconnected, camera released.")


async def send_motor_speeds(node, v_l: float, v_r: float, wheel_radius: float) -> None:
    """
    Thymio (Aseba) connection/command:
      - converts v_l, v_r (m/s) to 'motor.*.target' ([-500, 500]),
      - writes to Aseba variables via tdmclient.
    """
    # Empirical factor: Thymio speed (units) -> m/s
    # TODO: recalibrate properly if necessary
    thymio_factor = 0.00040816326530612246
    left_target = int(v_l / thymio_factor)
    right_target = int(v_r / thymio_factor)

    # Saturation to Thymio bounds
    left_target = max(-500, min(500, left_target))
    right_target = max(-500, min(500, right_target))

    await node.set_variables(
        {
            "motor.left.target": [left_target],
            "motor.right.target": [right_target],
        }
    )
