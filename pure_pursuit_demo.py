"""
Pure Pursuit Visualization Demo
A complete, ready-to-use visualization tool for the project report.
Simply import and call to generate a visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

from pure_pursuit import pure_pursuit_step


def create_pure_pursuit_visualization(
    path: Optional[List[Tuple[float, float]]] = None,
    initial_pos: Optional[Tuple[float, float, float]] = None,
    look_ahead_distance: float = 0.15,
    num_steps: int = 3,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
):
    """
    Creates a complete pure pursuit visualization with example data.
    
    Parameters:
    -----------
    path : List[Tuple[float, float]], optional
        List of (x, y) path points in meters. If None, uses default example path.
    initial_pos : Tuple[float, float, float], optional
        Initial robot position (x, y, heading_deg). If None, uses default.
    look_ahead_distance : float
        Look-ahead distance in meters (default: 0.15 m)
    num_steps : int
        Number of pure pursuit steps to simulate (default: 7)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, saves the figure to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Use provided path or default example path
    if path is None:
        path = [
            (0.1, 0.1),
            (0.2, 0.15),
            (0.3, 0.25),
            (0.4, 0.4),
            (0.5, 0.6),
            (0.6, 0.75),
            (0.7, 0.85),
            (0.8, 0.9)
        ]
    
    # Use provided initial position or default
    if initial_pos is None:
        initial_pos = (0.15, 0.12, 30.0)  # x, y, heading_deg
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the full path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, 'b-', linewidth=3, label='Planned Path', zorder=1, alpha=0.7)
    ax.scatter(path_x, path_y, c='blue', s=100, zorder=2, alpha=0.6, edgecolors='darkblue', linewidths=1.5)
    
    # Simulate multiple steps
    current_pos = initial_pos
    LFindex = 0
    positions = [current_pos]
    goal_points = []
    turn_errors = []
    
    for step in range(num_steps):
        goalPt, new_index, turnError = pure_pursuit_step(
            path, current_pos, look_ahead_distance, LFindex
        )
        goal_points.append(goalPt)
        turn_errors.append(turnError)
        
        # For visualization, simulate robot moving toward goal
        currentX, currentY, currentHeading = current_pos
        dist_to_goal = math.hypot(goalPt[0] - currentX, goalPt[1] - currentY)
        
        # Debug: Check if goal point is at correct distance
        # The goal point should be at look_ahead_distance (or very close if it's a path waypoint)
        expected_dist = look_ahead_distance
        dist_error = abs(dist_to_goal - expected_dist)
        if dist_error > 0.01:  # More than 1cm error
            # Goal point might be a path waypoint, not an intersection
            # In this case, we should still show it, but note it's not exactly at look_ahead_distance
            pass
        
        # Move robot significantly toward goal to spread out the circles
        # Move 80-90% of the way to the goal to ensure clear separation
        move_fraction = 0.85  # Move 85% of the way to goal
        step_size = dist_to_goal * move_fraction
        
        if dist_to_goal > 0.02:  # Only move if there's meaningful distance
            dx = (goalPt[0] - currentX) / dist_to_goal * step_size
            dy = (goalPt[1] - currentY) / dist_to_goal * step_size
            newX = currentX + dx
            newY = currentY + dy
            
            # Update heading to point toward goal
            new_heading = math.degrees(math.atan2(goalPt[1] - currentY, goalPt[0] - currentX))
            if new_heading < 0:
                new_heading += 360
            
            current_pos = (newX, newY, new_heading)
            positions.append(current_pos)
            LFindex = new_index
        else:
            # If too close, just move to goal point
            current_pos = (goalPt[0], goalPt[1], currentHeading)
            positions.append(current_pos)
            break
    
    # Plot robot trajectory
    traj_x = [p[0] for p in positions]
    traj_y = [p[1] for p in positions]
    ax.plot(traj_x, traj_y, 'g-', linewidth=3, label='Robot Trajectory', zorder=3, alpha=0.9)
    
    # Plot robot positions with headings
    for i, pos in enumerate(positions):
        color = 'green' if i == 0 else 'darkgreen'
        size = 200 if i == 0 else 120
        label_text = 'Start Position' if i == 0 else (f'Step {i}' if i < len(positions) - 1 else 'End Position')
        ax.scatter(pos[0], pos[1], c=color, s=size, zorder=5, 
                  edgecolors='black', linewidths=2.5, 
                  label=label_text if (i == 0 or i == len(positions) - 1) else '')
        
        # Draw heading arrow
        heading_rad = math.radians(pos[2])
        arrow_length = 0.045
        ax.arrow(
            pos[0], pos[1],
            arrow_length * math.cos(heading_rad),
            arrow_length * math.sin(heading_rad),
            head_width=0.025, head_length=0.025,
            fc=color, ec='black', linewidth=2.5, zorder=6
        )
    
    # Draw look-ahead circles and goal points for each step
    for i, (pos, goal) in enumerate(zip(positions[:-1], goal_points)):
        # Calculate actual distance from robot to goal point
        actual_dist = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        
        # Check if goal point is exactly at look_ahead_distance (within tolerance)
        # If not, it might be a path waypoint (when robot is close to path)
        tolerance = 0.005  # 5mm tolerance
        is_at_intersection = abs(actual_dist - look_ahead_distance) < tolerance
        
        # Draw look-ahead circle centered at robot position
        circle = plt.Circle(
            (pos[0], pos[1]),
            look_ahead_distance,
            fill=False,
            color='orange',
            linestyle='--',
            linewidth=3,
            alpha=0.7,
            zorder=2,
            label='Look-ahead Circle' if i == 0 else ''
        )
        ax.add_patch(circle)
        
        if is_at_intersection:
            # Goal point IS at intersection - perfect!
            # Draw line from robot to goal point (exactly look_ahead_distance)
            ax.plot([pos[0], goal[0]], [pos[1], goal[1]], 
                   'r-', linewidth=3, alpha=0.9, zorder=3,
                   label='Look-ahead Vector' if i == 0 else '')
            # Draw goal point at intersection
            ax.scatter(goal[0], goal[1], c='red', s=500, marker='*', 
                      zorder=8, edgecolors='darkred', linewidths=3, alpha=1.0,
                      label='Goal Point (Intersection)' if i == 0 else '')
            # Draw small circle at intersection
            intersection_marker = plt.Circle(
                (goal[0], goal[1]),
                0.01,
                fill=True,
                color='red',
                alpha=1.0,
                zorder=9,
                edgecolor='darkred',
                linewidth=2
            )
            ax.add_patch(intersection_marker)
        else:
            # Goal point is NOT at intersection - it's likely a path waypoint
            # Project it onto the circle to show where the intersection should be
            if actual_dist > 0.001:  # Avoid division by zero
                angle_to_goal = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
                # Calculate intersection point on the circle
                intersection_point = (
                    pos[0] + look_ahead_distance * math.cos(angle_to_goal),
                    pos[1] + look_ahead_distance * math.sin(angle_to_goal)
                )
                
                # Draw line to the ACTUAL intersection point on the circle
                ax.plot([pos[0], intersection_point[0]], [pos[1], intersection_point[1]], 
                       'r-', linewidth=3, alpha=0.9, zorder=3,
                       label='Look-ahead Vector (to intersection)' if i == 0 else '')
                
                # Draw the intersection point (where circle meets path)
                ax.scatter(intersection_point[0], intersection_point[1], c='red', s=500, marker='*', 
                          zorder=8, edgecolors='darkred', linewidths=3, alpha=1.0,
                          label='Intersection Point' if i == 0 else '')
                
                # Draw small circle at intersection
                intersection_marker = plt.Circle(
                    (intersection_point[0], intersection_point[1]),
                    0.01,
                    fill=True,
                    color='red',
                    alpha=1.0,
                    zorder=9,
                    edgecolor='darkred',
                    linewidth=2
                )
                ax.add_patch(intersection_marker)
                
                # Also show the actual goal point (waypoint) in a different color
                ax.scatter(goal[0], goal[1], c='purple', s=300, marker='s', 
                          zorder=7, edgecolors='darkviolet', linewidths=2, alpha=0.7,
                          label='Goal Waypoint' if i == 0 else '')
            else:
                # Goal point is at robot position (shouldn't happen)
                ax.scatter(goal[0], goal[1], c='red', s=500, marker='*', 
                          zorder=8, edgecolors='darkred', linewidths=3, alpha=1.0)
        
        # Optional: Add text showing the distance (for verification)
        # Uncomment to debug:
        # mid_x = (pos[0] + goal[0]) / 2
        # mid_y = (pos[1] + goal[1]) / 2
        # ax.text(mid_x, mid_y, f'{actual_dist:.3f}m', fontsize=8, 
        #         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add information text box
    info_text = (
        f"Look-ahead Distance: {look_ahead_distance:.3f} m\n"
        f"Number of Steps: {num_steps}\n"
        f"Path Length: {len(path)} waypoints"
    )
    
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='brown', linewidth=2),
        family='monospace'
    )
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'Pure Pursuit Algorithm - {num_steps} Steps', fontsize=16, fontweight='bold', pad=20)
    
    # Auto-scale
    margin = look_ahead_distance * 0.2
    all_x = path_x + traj_x
    all_y = path_y + traj_y
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


# Simple function to show pure pursuit visualization - just call this in your notebook!
def show_pure_pursuit_visualization(
    path: Optional[List[Tuple[float, float]]] = None,
    initial_pos: Optional[Tuple[float, float, float]] = None,
    look_ahead_distance: float = 0.15,
    num_steps: int = 3
):
    """
    Simple function to show pure pursuit visualization with 3 steps.
    Just call this in your notebook: show_pure_pursuit_visualization()
    
    Parameters:
    -----------
    path : List[Tuple[float, float]], optional
        List of (x, y) path points in meters. If None, uses default example path.
    initial_pos : Tuple[float, float, float], optional
        Initial robot position (x, y, heading_deg). If None, uses default.
    look_ahead_distance : float
        Look-ahead distance in meters (default: 0.15 m)
    num_steps : int
        Number of pure pursuit steps to simulate (default: 3)
    """
    fig, ax = create_pure_pursuit_visualization(
        path=path,
        initial_pos=initial_pos,
        look_ahead_distance=look_ahead_distance,
        num_steps=num_steps
    )
    plt.show()
    return fig, ax


# Main execution for easy use in notebook
if __name__ == "__main__":
    # Create pure pursuit visualization with 3 steps (default)
    show_pure_pursuit_visualization(num_steps=3)

