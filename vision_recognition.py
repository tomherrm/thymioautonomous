from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np






@dataclass
class RecognitionResult:
    frame_bgr: np.ndarray
    warped_bgr: np.ndarray
    warped_hsv: np.ndarray
    homography: np.ndarray
    thymio_pixel: tuple[int, int] | None
    thymio_cell: tuple[int, int] | None
    thymio_cells: np.ndarray | None
    thymio_orientation: float | None  # Angle in radians relative to x-axis (0 = to the right)
    goal_pixel: tuple[int, int] | None
    goal_cell: tuple[int, int] | None
    goal_cells: np.ndarray | None
    obstacle_cells: np.ndarray
    occ_grid_fine: np.ndarray
    occ_grid_coarse: np.ndarray

def export_map_data(result: RecognitionResult, filename: str = "map_data.json"):
    """Exports map data for pathfinding"""

    # Convert numpy arrays to Python lists
    map_data = {
        "occupancy_grid": result.occ_grid_coarse.tolist(),  # Occupancy grid
        "thymio_position": result.thymio_cell,  # Thymio position in cells
        "thymio_pixel": result.thymio_pixel,    # Position in pixels (optional)
        "grid_shape": result.occ_grid_coarse.shape,
        "obstacles": result.obstacle_cells.tolist() if hasattr(result.obstacle_cells, 'tolist') else result.obstacle_cells
    }

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders board points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32)
    idx = np.argsort(pts[:, 1])
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bottom = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def detect_board_and_corners(
    frame_bgr: np.ndarray,
    dark_percentile: float = 2.0,
    min_black_area: float = 30.0,
    board_min_area_ratio: float = 0.1,
) -> np.ndarray:
    """
    1) Detects main white surface (board).
    2) In each corner of board, finds black square.
    Returns source points (TL, TR, BR, BL).
    """
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, white_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No white surface detected.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_cnt = None
    img_area = w * h
    for cnt in contours:
        if cv2.contourArea(cnt) > board_min_area_ratio * img_area:
            board_cnt = cnt
            break
    if board_cnt is None:
        raise RuntimeError("No white board large enough detected.")

    x_board, y_board, w_board, h_board = cv2.boundingRect(board_cnt)

    centers: list[list[float]] = []
    roi_w = max(5, int(w_board * 0.25))
    roi_h = max(5, int(h_board * 0.25))
    corners_info = [
        (x_board, y_board),
        (x_board + w_board - roi_w, y_board),
        (x_board + w_board - roi_w, y_board + h_board - roi_h),
        (x_board, y_board + h_board - roi_h),
    ]

    for x0, y0 in corners_info:
        roi = blur[y0:y0 + roi_h, x0:x0 + roi_w]
        if roi.size == 0:
            raise RuntimeError("Empty ROI during corner search.")

        thr_val = np.percentile(roi, dark_percentile)
        roi_mask = (roi <= thr_val).astype(np.uint8) * 255

        kernel_small = np.ones((3, 3), np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        contours_roi, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours_roi:
            raise RuntimeError("No black square found in corner.")

        cnt = max(contours_roi, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_black_area:
            raise RuntimeError("Black square too small.")

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            raise RuntimeError("Null moment for black square.")

        cx = x0 + M["m10"] / M["m00"]
        cy = y0 + M["m01"] / M["m00"]
        centers.append([cx, cy])

    return order_points(np.array(centers, dtype=np.float32))


def detect_thymio_green(
    warped_bgr: np.ndarray,
    min_area_px: float = 5.0,
    min_area_ratio: float = 0.0,
    debug: bool = False,
) -> tuple[tuple[int, int] | None, np.ndarray | None]:
    """Detects Thymio green patch from rectified image."""
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 40, 67], np.uint8)
    upper_green = np.array([85, 160, 187], np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    min_area_dynamic = max(min_area_px, min_area_ratio * warped_bgr.shape[0] * warped_bgr.shape[1])
    if area < min_area_dynamic:
        return None, mask

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def detect_thymio_orientation(
    warped_bgr: np.ndarray,
    thymio_center: tuple[int, int],
    max_distance_px: int = 50,
    min_area_px: float = 0.0,
    debug: bool = False,
) -> tuple[float | None, tuple[int, int] | None]:
    """
    Detects Thymio orientation by finding a blue point (directional marker)
    near Thymio center. Optimized to detect even a single blue pixel.
    
    Args:
        warped_bgr: Rectified image in BGR
        thymio_center: Thymio center position (x, y)
        max_distance_px: Maximum distance of blue marker from center
        min_area_px: Minimum area of blue marker
        debug: Debug mode
    
    Returns:
        Tuple (angle_in_radians, marker_position) or (None, None) if not detected
    """
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    # Range for blue (H: 100-130 for medium blue)
    lower_blue = np.array([85, 50, 50], np.uint8)
    upper_blue = np.array([110, 255, 255], np.uint8)
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # For single pixel, avoid morphological operations that can remove it
    # Use only light CLOSE to group nearby pixels
    kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Don't use OPEN as it removes small elements
    
    # Method 1: Search via contours (for multi-pixel zones)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tx, ty = thymio_center
    best_center = None
    best_distance = float('inf')
    
    # Search in contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distance = math.sqrt((cx - tx)**2 + (cy - ty)**2)
        
        if distance < max_distance_px and distance < best_distance:
            best_distance = distance
            best_center = (cx, cy)
    
    # Method 2: If no contour found, search directly for blue pixels
    if best_center is None:
        # Extract coordinates of all blue pixels
        blue_pixels = np.argwhere(mask > 0)
        
        if len(blue_pixels) > 0:
            for pixel in blue_pixels:
                py, px = pixel  # Note: argwhere returns (row, col) so (y, x)
                distance = math.sqrt((px - tx)**2 + (py - ty)**2)
                
                if distance < max_distance_px and distance < best_distance:
                    best_distance = distance
                    best_center = (px, py)
    
    if best_center is None:
        return None, None
    
    # Calculate angle relative to x-axis (0 = to the right, counterclockwise)
    dx = best_center[0] - tx
    dy = best_center[1] - ty
    angle = math.atan2(dy, dx)  # Angle in radians
    
    return angle, best_center


def detect_goal_purple(
    warped_bgr: np.ndarray,
    min_area_px: float = 5.0,
    min_area_ratio: float = 0.0,
    debug: bool = False,
) -> tuple[tuple[int, int] | None, np.ndarray | None]:
    """
    Detects purple square (goal) from rectified image.
    Reference color: HSV [123, 80, 130]
    """
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    # Range around HSV [123, 80, 130]
    # H: 123 ± 15 (108-138 for purple)
    # S: 80 ± 40 (40-120)
    # V: 130 ± 50 (80-180)
    lower_purple = np.array([108, 40, 80], np.uint8)
    upper_purple = np.array([138, 180, 200], np.uint8)

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    min_area_dynamic = max(min_area_px, min_area_ratio * warped_bgr.shape[0] * warped_bgr.shape[1])
    if area < min_area_dynamic:
        return None, mask

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def build_red_occupancy(
    warped_hsv: np.ndarray,
    cell_size_px: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds yellow occupancy grid and returns occupied cells.
    
    Based on average HSV of 5 pixels: H=19, S=85, V=218
    """
    # Yellow range based on average: H=19, S=85, V=218
    # Tolerance: H ±10, S ±30, V ±40
    lower_yellow = np.array([10, 55, 178])   # H: 9-29, S: 55-115, V: 178-255
    upper_yellow = np.array([40, 115, 255])

    mask_yellow = cv2.inRange(warped_hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=1)
    occ_grid = (mask_yellow > 0).astype(np.uint8)
    
    # Dilate obstacles to create safety margin (Thymio size)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    occ_grid = cv2.dilate(occ_grid, dilation_kernel, iterations=1)
    
    occupied_cells_full = np.argwhere(occ_grid == 1)

    h, w = occ_grid.shape
    cell_size_px = max(1, cell_size_px)
    grid_h = max(1, h // cell_size_px)
    grid_w = max(1, w // cell_size_px)
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for i in range(grid_h):
        for j in range(grid_w):
            patch = occ_grid[
                i * cell_size_px:(i + 1) * cell_size_px,
                j * cell_size_px:(j + 1) * cell_size_px,
            ]
            if np.any(patch):
                grid[i, j] = 1

    obstacles = np.argwhere(grid == 1)
    return occ_grid, grid, obstacles, occupied_cells_full


def locate_thymio_cells(
    thymio_mask: np.ndarray | None,
    thymio_pos: tuple[int, int] | None,
    grid_shape: tuple[int, int],
) -> tuple[tuple[float, float] | None, tuple[int, int] | None, np.ndarray | None]:
    """Projects Thymio green patch onto coarse grid."""
    if thymio_mask is None:
        return None, None, None

    gh, gw = grid_shape
    h_mask, w_mask = thymio_mask.shape
    cell_size_y = max(1, h_mask // gh)
    cell_size_x = max(1, w_mask // gw)

    thymio_cell = None
    thymio_pixels = None
    if thymio_pos is not None:
        cx, cy = thymio_pos
        thymio_pixels = (float(cx), float(cy))
        thymio_cell = (min(gh - 1, cy // cell_size_y), min(gw - 1, cx // cell_size_x))

    ys, xs = np.nonzero(thymio_mask)
    rows = np.clip(ys // cell_size_y, 0, gh - 1)
    cols = np.clip(xs // cell_size_x, 0, gw - 1)
    thymio_cells = np.unique(np.stack((rows, cols), axis=1), axis=0)

    return thymio_pixels, thymio_cell, thymio_cells


def _show_frame(title: str, image_bgr: np.ndarray) -> None:
    """Displays a BGR image with matplotlib."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)


def _show_occ_grid(grid: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)


def _show_mask(mask: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)


def run_recognition(
    cam_index: int = 1,
    board_width_m: float = 0.84,
    board_height_m: float = 0.89,
    resolution_m: float = 0.01,  
    red_cell_size_px: int = 1,
    thymio_min_area_px: float = 5.0,
    thymio_min_area_ratio: float = 0.0002,
    display: bool = False,
) -> RecognitionResult:
    """Complete pipeline: capture -> homography -> Thymio + obstacles."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}.")

    # Let autofocus stabilize
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    
    # Wait for autofocus to stabilize (10 frames)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read image from camera.")

    src_pts = detect_board_and_corners(frame)
    width_px = max(2, int(board_width_m / resolution_m))
    height_px = max(2, int(board_height_m / resolution_m))

    dst_pts = np.array(
        [
            [0, 0],
            [width_px - 1, 0],
            [width_px - 1, height_px - 1],
            [0, height_px - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, homography, (width_px, height_px))
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    thymio_pos, thymio_mask = detect_thymio_green(
        warped,
        min_area_px=thymio_min_area_px,
        min_area_ratio=thymio_min_area_ratio,
        debug=display,
    )
    
    # Detect Thymio orientation (directional blue point)
    # Adjust max distance based on resolution (more pixels = larger distance)
    max_distance_orientation = int(50 * (0.02 / resolution_m))  # Proportional adjustment
    thymio_orientation = None
    if thymio_pos is not None:
        thymio_orientation, _ = detect_thymio_orientation(
            warped,
            thymio_pos,
            max_distance_px=max_distance_orientation,
            min_area_px=0.0,  # Allows detecting even a single pixel
            debug=display,
        )
    
    goal_pos, goal_mask = detect_goal_purple(
        warped,
        min_area_px=thymio_min_area_px,
        min_area_ratio=thymio_min_area_ratio,
        debug=display,
    )
    (
        occ_grid_fine,
        occ_grid_coarse,
        obstacle_cells_coarse,
        obstacle_cells_full,
    ) = build_red_occupancy(warped_hsv, cell_size_px=red_cell_size_px)
    thymio_mask_for_cells = thymio_mask if thymio_pos is not None else None
    thymio_pixels, thymio_cell, thymio_cells = locate_thymio_cells(
        thymio_mask_for_cells, thymio_pos, occ_grid_fine.shape
    )
    goal_mask_for_cells = goal_mask if goal_pos is not None else None
    goal_pixels, goal_cell, goal_cells = locate_thymio_cells(
        goal_mask_for_cells, goal_pos, occ_grid_fine.shape
    )

    result = RecognitionResult(
        frame_bgr=frame,
        warped_bgr=warped,
        warped_hsv=warped_hsv,
        homography=homography,
        thymio_pixel=thymio_pixels,
        thymio_cell=thymio_cell,
        thymio_cells=thymio_cells,
        thymio_orientation=thymio_orientation,
        goal_pixel=goal_pixels,
        goal_cell=goal_cell,
        goal_cells=goal_cells,
        obstacle_cells=obstacle_cells_full,
        occ_grid_fine=occ_grid_fine,
        occ_grid_coarse=occ_grid_coarse,
    )

    if display:
        _show_frame("Camera Image", result.frame_bgr)
        _show_frame("Rectified Board", result.warped_bgr)
        if thymio_mask is not None:
            _show_mask(thymio_mask, "Green Mask (Thymio)")
        if goal_mask is not None:
            _show_mask(goal_mask, "Yellow Mask (Goal)")
        _show_occ_grid(result.occ_grid_coarse, "Occupancy Grid (yellow)")
        plt.show()

    return result

