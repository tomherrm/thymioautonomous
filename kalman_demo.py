"""
Kalman Filter Visualization Demo
A complete, ready-to-use visualization tool for the project report.
Simply import and call to generate a visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from kalmanpy import kalman_filter


def create_kalman_visualization(
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    cam_noise_xy: float = 0.0005,
    cam_noise_th: float = 0.3,
    wheel_noise_std: float = 0.012,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Creates a complete Kalman filter visualization with example data.
    
    Parameters:
    -----------
    Q : np.ndarray, optional
        Process noise covariance matrix (3x3). If None, uses default values.
    R : np.ndarray, optional
        Measurement noise covariance matrix (3x3). If None, uses default values.
    cam_noise_xy : float
        Camera position noise standard deviation in meters (default: 0.0005 m = 0.5 mm)
    cam_noise_th : float
        Camera angle noise standard deviation in degrees (default: 0.3°)
    wheel_noise_std : float
        Wheel speed measurement noise standard deviation in m/s (default: 0.012 m/s)
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, saves the figure to this path
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    # --- Parameters ---
    dt = 0.05
    steps = 400
    wheel_radius = 0.023
    axle_length = 0.093
    
    # Same Thymio factor as in motion_control.py (motor units -> m/s)
    thymio_factor = 0.00040816326530612246
    
    # --- Process/measurement covariances (as in motion_control.py) ---
    P = np.eye(3) * 1e-3
    dt_q = dt
    
    # Use provided Q or calculate default
    if Q is None:
        sigma_v2 = 151.4387409522633e-6  # (m/s)^2
        sigma_omega = 2 * sigma_v2 / axle_length**2
        q_theta = sigma_omega * dt_q**2
        q_pos = sigma_v2 * dt_q**2
        Q = np.zeros((3, 3))
        Q[0, 0] = q_pos
        Q[1, 1] = q_pos
        Q[2, 2] = q_theta
    
    # Use provided R or calculate default
    if R is None:
        R = np.diag(
            [
                0.1e-6,
                0.1e-6,
                (0.3 * np.pi / 180.0) ** 2,  # ~2° error on angle
            ]
        )
    
    # Extract sigma values from Q for odometry simulation
    sigma_v2 = Q[0, 0] / dt_q**2 if Q[0, 0] > 0 else 151.4387409522633e-6
    sigma_omega = Q[2, 2] / dt_q**2 if Q[2, 2] > 0 else 2 * sigma_v2 / axle_length**2
    sigma_v = np.sqrt(sigma_v2)
    sigma_w = np.sqrt(sigma_omega)
    
    # Convert camera angle noise from degrees to radians
    cam_noise_th_rad = np.deg2rad(cam_noise_th)
    
    # --- Ground truth + synthetic sensors ---
    true_states = []     # [x, y, theta]
    odo_states = []      # dead-reckoning (no camera)
    kalman_states = []   # EKF estimate
    meas_states = []     # noisy "camera"
    
    # Initial true pose
    x_true = np.array([[0.0], [0.0], [0.0]])
    
    # Initial EKF state
    x_kalman = x_true.copy()
    P_kalman = P.copy()
    
    # For simple dead-reckoning (use same motion model without correction)
    x_odo = x_true.copy()
    
    # Command pattern: go straight, then turn, etc.
    # We define desired v,w in physical units, then convert to Thymio motor units.
    def command_pattern(k):
        if k < 150:
            v = 0.04      # m/s
            w = 0.0       # rad/s
        elif k < 250:
            v = 0.03
            w = 0.6
        else:
            v = 0.04
            w = 0.0
    
        # Physical wheel speeds (m/s)
        v_r_true = v + (axle_length / 2.0) * w
        v_l_true = v - (axle_length / 2.0) * w
    
        # Convert to Thymio motor units (same way as in send_motor_speeds, but inverted)
        right_raw = v_r_true / thymio_factor
        left_raw = v_l_true / thymio_factor
    
        # Clip to approximate Thymio limits [-500, 500]
        right_raw = max(-500, min(500, right_raw))
        left_raw = max(-500, min(500, left_raw))
    
        return int(left_raw), int(right_raw)
    
    for k in range(steps):
        # --- "Command" from controller in Thymio units ---
        left_raw_cmd, right_raw_cmd = command_pattern(k)
    
        # True wheel speeds in m/s (no actuation noise here)
        v_l_true = left_raw_cmd * thymio_factor
        v_r_true = right_raw_cmd * thymio_factor
    
        v_cmd = (v_r_true + v_l_true) / 2.0
        w_cmd = (v_r_true - v_l_true) / axle_length
    
        # --- Ground truth propagation (RK2, same as kalman) ---
        theta_mid = x_true[2, 0] + 0.5 * w_cmd * dt
        x_true = np.array(
            [
                [x_true[0, 0] + v_cmd * np.cos(theta_mid) * dt],
                [x_true[1, 0] + v_cmd * np.sin(theta_mid) * dt],
                [x_true[2, 0] + w_cmd * dt],
            ]
        )
        x_true[2, 0] = (x_true[2, 0] + np.pi) % (2 * np.pi) - np.pi
    
        # --- Simulated noisy wheel speeds ("measured" like in motion_control) ---
        v_l_meas = v_l_true + np.random.normal(0, wheel_noise_std)
        v_r_meas = v_r_true + np.random.normal(0, wheel_noise_std)
        u_vec = np.array([[v_r_meas], [v_l_meas], [0.0]])
    
        # --- Simulated noisy camera measurement ---
        y_vec = np.array(
            [
                [x_true[0, 0] + np.random.normal(0, cam_noise_xy)],
                [x_true[1, 0] + np.random.normal(0, cam_noise_xy)],
                [x_true[2, 0] + np.random.normal(0, cam_noise_th_rad)],
            ]
        )
    
        # --- Dead-reckoning (odometry only, using same noise that defines Q) ---
        # Add noise to commanded linear and angular velocities
        v_cmd_odo = v_cmd + np.random.normal(0, sigma_v)
        w_cmd_odo = w_cmd + np.random.normal(0, sigma_w)
    
        theta_mid_odo = x_odo[2, 0] + 0.5 * w_cmd_odo * dt
        x_odo = np.array(
            [
                [x_odo[0, 0] + v_cmd_odo * np.cos(theta_mid_odo) * dt],
                [x_odo[1, 0] + v_cmd_odo * np.sin(theta_mid_odo) * dt],
                [x_odo[2, 0] + w_cmd_odo * dt],
            ]
        )
        x_odo[2, 0] = (x_odo[2, 0] + np.pi) % (2 * np.pi) - np.pi
    
        # --- Kalman filter update (same API as in motion_control) ---
        # Shut off camera in second quarter to show pure odometry behavior
        quarter = steps // 4
        half = steps // 2
        y_vec_kalman = None if (quarter <= k < half) else y_vec
        
        xk_new, yk_new, th_new, _, _, P_kalman = kalman_filter(
            x=x_kalman,
            u=u_vec,
            y=y_vec_kalman,  # None for second quarter, then camera measurements
            P=P_kalman,
            Q=Q,
            R=R,
            dt=dt,
            r=wheel_radius,
            L=axle_length,
        )
        x_kalman = np.array([[xk_new], [yk_new], [th_new]])
    
        # Store data
        true_states.append(x_true.flatten())
        odo_states.append(x_odo.flatten())
        kalman_states.append(x_kalman.flatten())
        meas_states.append(y_vec.flatten())
    
    true_states = np.array(true_states)
    odo_states = np.array(odo_states)
    kalman_states = np.array(kalman_states)
    meas_states = np.array(meas_states)
    
    # --- Angle unwrapping for nicer heading plot ---
    true_theta = np.unwrap(true_states[:, 2])
    odo_theta = np.unwrap(odo_states[:, 2])
    kalman_theta = np.unwrap(kalman_states[:, 2])
    meas_theta = np.unwrap(meas_states[:, 2])
    
    # --- Plots ---
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Trajectories in XY
    ax = axs[0]
    ax.plot(true_states[:, 0], true_states[:, 1], "k-", linewidth=2, label="True", zorder=3)
    ax.plot(odo_states[:, 0], odo_states[:, 1], "r--", linewidth=2, label="Odometry only", zorder=2)
    ax.plot(kalman_states[:, 0], kalman_states[:, 1], "b-", linewidth=2, label="Kalman (odo + camera)", zorder=3)
    
    # Show camera measurements only when camera is "on" (first quarter + last half)
    quarter = steps // 4
    half = steps // 2
    # First quarter
    ax.scatter(meas_states[:quarter, 0], meas_states[:quarter, 1], 
               s=8, c="g", alpha=0.5, label="Camera meas.", zorder=1)
    # Last half (skip second quarter)
    ax.scatter(meas_states[half:, 0], meas_states[half:, 1], 
               s=8, c="g", alpha=0.5, zorder=1)
    
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12, fontweight='bold')
    ax.set_ylabel("y [m]", fontsize=12, fontweight='bold')
    ax.set_title("Trajectory", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Heading vs time
    t = np.arange(steps) * dt
    ax2 = axs[1]
    ax2.plot(t, true_theta, "k-", linewidth=2, label="True θ", zorder=3)
    ax2.plot(t, odo_theta, "r--", linewidth=2, label="Odometry θ", zorder=2)
    ax2.plot(t, kalman_theta, "b-", linewidth=2, label="Kalman θ", zorder=3)
    
    # Show camera measurements only when camera is "on" (first quarter + last half)
    # First quarter
    ax2.scatter(t[:quarter], meas_theta[:quarter], 
               s=8, c="g", alpha=0.5, label="Camera meas.", zorder=1)
    # Last half (skip second quarter)
    ax2.scatter(t[half:], meas_theta[half:], 
               s=8, c="g", alpha=0.5, zorder=1)
    
    ax2.set_xlabel("time [s]", fontsize=12, fontweight='bold')
    ax2.set_ylabel("θ [rad] (unwrapped)", fontsize=12, fontweight='bold')
    ax2.set_title("Heading", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axs


# Simple function to show Kalman filter visualization - just call this in your notebook!
def show_kalman_visualization(
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    cam_noise_xy: float = 0.0005,
    cam_noise_th: float = 0.3,
    wheel_noise_std: float = 0.012
):
    """
    Simple function to show Kalman filter visualization.
    Just call this in your notebook: show_kalman_visualization()
    
    Parameters:
    -----------
    Q : np.ndarray, optional
        Process noise covariance matrix (3x3). If None, uses default values.
    R : np.ndarray, optional
        Measurement noise covariance matrix (3x3). If None, uses default values.
    cam_noise_xy : float
        Camera position noise standard deviation in meters (default: 0.0005 m = 0.5 mm)
    cam_noise_th : float
        Camera angle noise standard deviation in degrees (default: 0.3°)
    wheel_noise_std : float
        Wheel speed measurement noise standard deviation in m/s (default: 0.012 m/s)
    """
    fig, axs = create_kalman_visualization(
        Q=Q,
        R=R,
        cam_noise_xy=cam_noise_xy,
        cam_noise_th=cam_noise_th,
        wheel_noise_std=wheel_noise_std
    )
    plt.show()
    return fig, axs


# Main execution for easy use in notebook
if __name__ == "__main__":
    # Create Kalman filter visualization with default parameters
    show_kalman_visualization()

