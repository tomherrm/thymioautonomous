import math
from typing import Optional, Tuple

import cv2
import numpy as np

from vision_recognition import (
    detect_board_and_corners,
    detect_thymio_green,
    detect_thymio_orientation,
    detect_goal_purple,
    build_red_occupancy,
    locate_thymio_cells,
)


class ThymioPoseTracker:
    """
    Manages a continuously open camera stream and allows reading Thymio pose
    (x, y, theta) in meters / radians at each call.
    """

    def __init__(
        self,
        cam_index: int = 0,
        board_width_m: float = 0.84,
        board_height_m: float = 0.89,
        resolution_m: float = 0.01,
        thymio_min_area_px: float = 5.0,
        thymio_min_area_ratio: float = 0.0002,
    ) -> None:
        self.cam_index = cam_index
        self.board_width_m = board_width_m
        self.board_height_m = board_height_m
        self.resolution_m = resolution_m
        self.thymio_min_area_px = thymio_min_area_px
        self.thymio_min_area_ratio = thymio_min_area_ratio
        # Last known goal position (in meters, same frame as Thymio)
        self.last_goal_m: Optional[Tuple[float, float]] = None

        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.cam_index}.")

        # Stabilize autofocus a bit
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        for _ in range(10):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("Cannot read image from camera.")

        # Detect board once and calculate homography
        src_pts = detect_board_and_corners(frame)
        self.width_px = max(2, int(self.board_width_m / self.resolution_m))
        self.height_px = max(2, int(self.board_height_m / self.resolution_m))

        dst_pts = np.array(
            [
                [0, 0],
                [self.width_px - 1, 0],
                [self.width_px - 1, self.height_px - 1],
                [0, self.height_px - 1],
            ],
            dtype=np.float32,
        )
        self.homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def get_pose_m(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (x, y, theta) in meters / radians, or None if not detected.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        warped = cv2.warpPerspective(
            frame,
            self.homography,
            (self.width_px, self.height_px),
        )

        thymio_pos, _ = detect_thymio_green(
            warped,
            min_area_px=self.thymio_min_area_px,
            min_area_ratio=self.thymio_min_area_ratio,
            debug=False,
        )

        if thymio_pos is None:
            return None

        # Orientation from blue marker
        max_distance_orientation = int(50 * (0.02 / self.resolution_m))
        theta = None
        if thymio_pos is not None:
            theta, _ = detect_thymio_orientation(
                warped,
                thymio_pos,
                max_distance_px=max_distance_orientation,
                min_area_px=0.0,
                debug=False,
            )

        if theta is None:
            return None

        # Use opposite of measured angle (simple calibration requested)
        theta_world = -theta
        while theta_world > math.pi:
            theta_world -= 2 * math.pi
        while theta_world < -math.pi:
            theta_world += 2 * math.pi

        cx, cy = thymio_pos
        # Metric coordinates with origin at bottom-left:
        #  - x to the right
        #  - y upward (invert vertical axis of image)
        x_m = cx * self.resolution_m
        y_m = (self.height_px - 1 - cy) * self.resolution_m

        # --- Detect goal (yellow/purple square) on same image ---
        # Simply store its metric position for possible path recalculation
        goal_pos, _ = detect_goal_purple(
            warped,
            min_area_px=self.thymio_min_area_px,
            min_area_ratio=self.thymio_min_area_ratio,
            debug=False,
        )
        if goal_pos is not None:
            gx, gy = goal_pos
            goal_x_m = gx * self.resolution_m
            goal_y_m = (self.height_px - 1 - gy) * self.resolution_m
            self.last_goal_m = (goal_x_m, goal_y_m)

        return x_m, y_m, theta_world

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_navigation_snapshot(
        self,
        red_cell_size_px: int = 1,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int] | None, Tuple[int, int] | None]]:
        """
        Captures a frame, reconstructs occupancy grid and returns:
          - occ_grid_coarse (fine obstacle occupancy grid)
          - thymio_cell (robot cell in this grid)
          - goal_cell (goal cell in this grid)
        Uses already open camera and homography (not run_recognition).
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Apply homography to get rectified "board" image
        warped = cv2.warpPerspective(
            frame,
            self.homography,
            (self.width_px, self.height_px),
        )
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # Thymio detection (green mask)
        thymio_pos, thymio_mask = detect_thymio_green(
            warped,
            min_area_px=self.thymio_min_area_px,
            min_area_ratio=self.thymio_min_area_ratio,
            debug=False,
        )

        # Goal detection
        goal_pos, goal_mask = detect_goal_purple(
            warped,
            min_area_px=self.thymio_min_area_px,
            min_area_ratio=self.thymio_min_area_ratio,
            debug=False,
        )

        # Obstacle occupancy grid (yellow)
        (
            occ_grid_fine,
            occ_grid_coarse,
            obstacle_cells_coarse,
            obstacle_cells_full,
        ) = build_red_occupancy(warped_hsv, cell_size_px=red_cell_size_px)

        # Locate Thymio and goal in fine grid
        thymio_mask_for_cells = thymio_mask if thymio_pos is not None else None
        thymio_pixels, thymio_cell, thymio_cells = locate_thymio_cells(
            thymio_mask_for_cells, thymio_pos, occ_grid_fine.shape
        )
        goal_mask_for_cells = goal_mask if goal_pos is not None else None
        goal_pixels, goal_cell, goal_cells = locate_thymio_cells(
            goal_mask_for_cells, goal_pos, occ_grid_fine.shape
        )

        return occ_grid_coarse, thymio_cell, goal_cell

