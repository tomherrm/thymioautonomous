import math
from typing import List, Tuple

import numpy as np


def sgn(x: float) -> float:
    """Returns the sign of x (-1, 0, +1)."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def pt_to_pt_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def pure_pursuit_step(
    path: List[Tuple[float, float]],
    current_pos: Tuple[float, float, float],
    lookAheadDis: float,
    LFindex: int,
) -> Tuple[Tuple[float, float], int, float]:
    """
    Pure pursuit implementation (TP version) in metric coordinates.

    - path: list of points (x, y) in meters
    - current_pos: (x, y, heading_deg) where heading is in degrees
    - lookAheadDis: look-ahead distance (m)
    - LFindex: index of last reached point in path

    Returns (goal_point(x,y), new_index, turnError_deg)
    """
    currentX = current_pos[0]
    currentY = current_pos[1]
    currentHeading = current_pos[2]  # in degrees

    lastFoundIndex = LFindex
    startingIndex = lastFoundIndex

    # Search for ALL possible intersections and take the farthest valid one
    best_goal_pt = None
    best_index = lastFoundIndex
    best_distance_along_path = -1.0  # Distance along path from start

    # Search for intersection between look-ahead circle and path segment
    for i in range(startingIndex, len(path) - 1):
        x1 = path[i][0] - currentX
        y1 = path[i][1] - currentY
        x2 = path[i + 1][0] - currentX
        y2 = path[i + 1][1] - currentY
        dx = x2 - x1
        dy = y2 - y1
        dr = math.hypot(dx, dy)
        D = x1 * y2 - x2 * y1
        discriminant = (lookAheadDis**2) * (dr**2) - D**2

        if discriminant >= 0 and dr > 1e-9:
            sqrt_disc = math.sqrt(discriminant)
            sol_x1 = (D * dy + sgn(dy) * dx * sqrt_disc) / (dr**2)
            sol_x2 = (D * dy - sgn(dy) * dx * sqrt_disc) / (dr**2)
            sol_y1 = (-D * dx + abs(dy) * sqrt_disc) / (dr**2)
            sol_y2 = (-D * dx - abs(dy) * sqrt_disc) / (dr**2)

            sol_pt1 = (sol_x1 + currentX, sol_y1 + currentY)
            sol_pt2 = (sol_x2 + currentX, sol_y2 + currentY)

            # Check if intersection points are actually on the line segment
            def point_on_segment(pt, seg_start, seg_end, tolerance=1e-6):
                """Check if point is on line segment using parametric form."""
                dx_seg = seg_end[0] - seg_start[0]
                dy_seg = seg_end[1] - seg_start[1]
                seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
                
                if seg_len_sq < 1e-12:  # Segment is too short
                    dist_to_start = pt_to_pt_distance(pt, seg_start)
                    dist_to_end = pt_to_pt_distance(pt, seg_end)
                    return dist_to_start < tolerance or dist_to_end < tolerance
                
                dx_pt = pt[0] - seg_start[0]
                dy_pt = pt[1] - seg_start[1]
                t = (dx_pt * dx_seg + dy_pt * dy_seg) / seg_len_sq
                
                if 0 <= t <= 1:
                    proj_x = seg_start[0] + t * dx_seg
                    proj_y = seg_start[1] + t * dy_seg
                    dist_to_line = math.hypot(pt[0] - proj_x, pt[1] - proj_y)
                    return dist_to_line < tolerance
                return False
            
            in1 = point_on_segment(sol_pt1, path[i], path[i + 1])
            in2 = point_on_segment(sol_pt2, path[i], path[i + 1])

            if in1 or in2:
                # Choose intersection point that's ahead of robot (farther along path)
                if in1 and in2:
                    # Calculate which point is farther along the path from path[i]
                    dist1_from_seg_start = pt_to_pt_distance(sol_pt1, path[i])
                    dist2_from_seg_start = pt_to_pt_distance(sol_pt2, path[i])
                    goalPt = sol_pt1 if dist1_from_seg_start > dist2_from_seg_start else sol_pt2
                else:
                    goalPt = sol_pt1 if in1 else sol_pt2

                # Check that intersection point is ahead of robot (don't go backward)
                # Calculate path direction vector
                path_dx = path[i + 1][0] - path[i][0]
                path_dy = path[i + 1][1] - path[i][1]
                path_len = math.hypot(path_dx, path_dy)
                
                if path_len > 1e-9:
                    # Vector from path[i] to goal point
                    goal_dx = goalPt[0] - path[i][0]
                    goal_dy = goalPt[1] - path[i][1]
                    
                    # Vector from robot to goal point
                    robot_to_goal_dx = goalPt[0] - currentX
                    robot_to_goal_dy = goalPt[1] - currentY
                    
                    # Check if goal is ahead: dot product of path direction and robot-to-goal should be positive
                    # OR if goal is on a segment ahead of the robot's current segment
                    dot_product = (path_dx * robot_to_goal_dx + path_dy * robot_to_goal_dy) / path_len
                    goal_ahead = dot_product > -0.01  # Small tolerance
                    
                    # Also check if goal is on a segment ahead of robot's current position
                    # by checking if distance along path from path[i] to goal > 0
                    goal_dist_along_seg = (goal_dx * path_dx + goal_dy * path_dy) / path_len
                    goal_on_ahead_segment = (i > lastFoundIndex) or (goal_dist_along_seg > -0.01)
                    
                    if goal_ahead or goal_on_ahead_segment:
                        # Calculate distance along path to this point
                        dist_along_path = 0.0
                        for j in range(i):
                            dist_along_path += pt_to_pt_distance(path[j], path[j + 1])
                        dist_along_path += pt_to_pt_distance(path[i], goalPt)
                        
                        # Keep farthest intersection along path
                        if dist_along_path > best_distance_along_path:
                            best_goal_pt = goalPt
                            # Update index: if robot is very close to path[i+1], advance index
                            if i < len(path) - 1:
                                dist_robot_to_next = pt_to_pt_distance((currentX, currentY), path[i + 1])
                                if dist_robot_to_next < 0.02:  # Robot very close to path[i+1]
                                    best_index = i + 1
                                else:
                                    best_index = i
                            else:
                                best_index = i
                            best_distance_along_path = dist_along_path
    
    if best_goal_pt is not None:
        goalPt = best_goal_pt
        lastFoundIndex = best_index
    else:
        # No new intersection found: check if we can advance index
        # If robot has passed next point, we can advance
        if lastFoundIndex < len(path) - 1:
            dist_to_next = pt_to_pt_distance(
                (currentX, currentY),
                path[lastFoundIndex + 1]
            )
            # If very close to next point (< lookahead), we can use it
            if dist_to_next < lookAheadDis * 0.5:
                lastFoundIndex = lastFoundIndex + 1
                goalPt = path[lastFoundIndex]
            else:
                goalPt = (path[lastFoundIndex][0], path[lastFoundIndex][1])
        else:
            # Already at last point
            goalPt = (path[lastFoundIndex][0], path[lastFoundIndex][1])
    
    # Force index advancement if robot has passed several points (before calculating angle)
    # (to avoid lf_index getting stuck, but only if we didn't find a better intersection)
    if best_goal_pt is None:  # Only advance if no intersection found
        while lastFoundIndex < len(path) - 1:
            dist_to_next = pt_to_pt_distance(
                (currentX, currentY),
                path[lastFoundIndex + 1]
            )
            # If robot is very close to next point, we can advance it
            if dist_to_next < lookAheadDis * 0.3:  # 30% of lookahead
                lastFoundIndex += 1
            else:
                break

    # Calculate absolute angle to goal point (in degrees)
    absTargetAngle = math.degrees(math.atan2(goalPt[1] - currentY, goalPt[0] - currentX))
    if absTargetAngle < 0:
        absTargetAngle += 360.0

    # Minimal angle error in degrees
    turnError = absTargetAngle - currentHeading
    if turnError > 180.0 or turnError < -180.0:
        turnError = -sgn(turnError) * (360.0 - abs(turnError))

    return goalPt, lastFoundIndex, turnError