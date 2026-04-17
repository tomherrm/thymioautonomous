import numpy as np


def _normalize_angle(angle):
    """Normalizes an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def kalman_filter(
    x, u, y, P, Q, R, dt, r, L,
    
):
    """
    
    - Process noise Q oriented in robot frame then global rotation
    - Angle normalization to avoid jumps around ±π

    x : current state [x, y, theta]^T (shape (3,1))
    u : command [v_r, v_l, a_f]^T (shape (3,1))
    y : measurement [x_meas, y_meas, theta_meas]^T (shape (3,1))
    P : state covariance matrix (3x3)
    Q : model covariance (3x3) - can be None, will be recalculated
    R : measurement covariance (3x3)
    dt : time step (s)
    r : wheel radius (m) - not used directly here
    L : axle length in m
    var_trans : translational noise standard deviation (m/s)
    var_rot : rotational noise standard deviation (rad/s)
    var_side : lateral noise standard deviation (m/s) - generally very small
    """
    # State decomposition
    xk = x[0, 0]
    yk = x[1, 0]
    th = x[2, 0]

    # Commands (wheel speeds in m/s)
    v_thymio_r = u[0, 0]
    v_thymio_l = u[1, 0]

    # Convert wheel speeds -> linear/angular velocities
    v_cmd = (v_thymio_r + v_thymio_l) / 2.0
    w_cmd = (v_thymio_r - v_thymio_l) / L

    # --- RK2 method: use theta_mid for better accuracy ---
    theta_mid = th + (w_cmd * dt * 0.5)

    # Transition matrix F (linearization around theta_mid) for state [x, y, theta]
    F = np.array(
        [
            [1.0, 0.0, -v_cmd * np.sin(theta_mid) * dt],
            [0.0, 1.0,  v_cmd * np.cos(theta_mid) * dt],
            [0.0, 0.0, 1.0],
        ]
    )

    # Nonlinear transition function applied to x (with theta_mid), pose-only state
    x_apriori = np.array(
        [
            [xk + v_cmd * np.cos(theta_mid) * dt],
            [yk + v_cmd * np.sin(theta_mid) * dt],
            [th + w_cmd * dt],
        ]
    )
    # Normalize angle after prediction
    x_apriori[2, 0] = _normalize_angle(x_apriori[2, 0])

           

    # Observation matrix H: we measure [x, y, theta]
    H = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # ---------- Manual EKF using matrix multiplications (no FilterPy) ----------
    # A = F is the Jacobian of the motion model wrt state
    A = F

    # A priori covariance propagation
    P_apriori = A @ P @ A.T
    if Q is not None:
        P_apriori = P_apriori + Q

    # A priori state (already computed by nonlinear model)
    x_apriori_vec = x_apriori.reshape(3, 1)

    if y is not None:
        # Measurement vector
        z = y.reshape(3, 1).copy()

        # Innovation with special handling for angle
        theta_meas = z[2, 0]
        theta_pred = x_apriori_vec[2, 0]
        theta_innov = _normalize_angle(theta_meas - theta_pred)
        # Limit angular innovation to ~45° to avoid jumps
        max_innov = np.deg2rad(45.0)
        if abs(theta_innov) > max_innov:
            theta_innov = np.sign(theta_innov) * max_innov

        innovation = z - H @ x_apriori_vec
        innovation[2, 0] = theta_innov

        # Innovation covariance
        S = H @ P_apriori @ H.T + R
        K = P_apriori @ H.T @ np.linalg.inv(S)

        # A posteriori update
        x_new = x_apriori_vec + K @ innovation
        P_new = P_apriori - K @ H @ P_apriori
    else:
        # No measurement available: pure prediction
        x_new = x_apriori_vec
        P_new = P_apriori

    # Normalize angle in state
    x_new[2, 0] = _normalize_angle(x_new[2, 0])

    xk_new = x_new[0, 0]
    yk_new = x_new[1, 0]
    th_new = x_new[2, 0]  # already normalized

    # For velocities, we want to reflect **instantaneous motor measurement**,
    # i.e. what comes from wheel speeds (v_thymio_r, v_thymio_l).
    # So we directly reuse v_cmd and w_cmd calculated at the beginning.
    v_new = v_cmd
    w_new = w_cmd

    return xk_new, yk_new, th_new, v_new, w_new, P_new

