import time
import numpy as np

from vision_tracker import ThymioPoseTracker


def estimate_camera_covariance(
    cam_index: int = 0,
    board_width_m: float = 0.84,
    board_height_m: float = 0.891,
    resolution_m: float = 0.01,
    n_samples: int = 100,
    sleep_dt: float = 0.02,
) -> None:
    """
    Collects a bunch of camera poses for a *static* robot and computes
    the empirical covariance matrix of [x, y, theta]^T measurements.
    """
    tracker = ThymioPoseTracker(
        cam_index=cam_index,
        board_width_m=board_width_m,
        board_height_m=board_height_m,
        resolution_m=resolution_m,
    )

    poses = []

    print(f"Collecting {n_samples} camera samples...")
    try:
        while len(poses) < n_samples:
            pose = tracker.get_pose_m()
            if pose is not None:
                x, y, th = pose
                poses.append([x, y, th])

            time.sleep(sleep_dt)

    finally:
        tracker.release()

    poses = np.array(poses)
    mean = poses.mean(axis=0)
    # rows = variables, columns = samples → transpose
    cov = np.cov(poses.T)

    print("\nNumber of valid samples:", poses.shape[0])
    print("Mean pose [x, y, theta]:")
    print(mean)
    print("\nEmpirical covariance matrix R_cam (for [x, y, theta]):")
    print(cov)
    print("\nSuggested R for Kalman (diagonal only):")
    print("R = np.diag([")
    print(f"    {cov[0, 0]:.8f},  # var x")
    print(f"    {cov[1, 1]:.8f},  # var y")
    print(f"    {cov[2, 2]:.8f},  # var theta")
    print("])")


if __name__ == "__main__":
    estimate_camera_covariance()