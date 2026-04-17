"""
Outil interactif pour calibrer la couleur du patch vert du Thymio.
Lance : python color_tuner.py --cam 0
Clique sur le patch dans la fenêtre pour afficher les valeurs BGR/HSV.
Utilise les trackbars pour ajuster un masque HSV et visualiser le résultat.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def nothing(_value: int) -> None:
    """Callback no-op pour les trackbars."""


def create_trackbars() -> None:
    cv2.namedWindow("HSV mask", cv2.WINDOW_NORMAL)
    for name, max_val in (
        ("H min", 179),
        ("H max", 179),
        ("S min", 255),
        ("S max", 255),
        ("V min", 255),
        ("V max", 255),
    ):
        cv2.createTrackbar(name, "HSV mask", 0 if "min" in name else max_val, max_val, nothing)
    cv2.setTrackbarPos("H max", "HSV mask", 179)
    cv2.setTrackbarPos("S max", "HSV mask", 255)
    cv2.setTrackbarPos("V max", "HSV mask", 255)


def read_trackbar_values() -> tuple[np.ndarray, np.ndarray]:
    h_min = cv2.getTrackbarPos("H min", "HSV mask")
    h_max = cv2.getTrackbarPos("H max", "HSV mask")
    s_min = cv2.getTrackbarPos("S min", "HSV mask")
    s_max = cv2.getTrackbarPos("S max", "HSV mask")
    v_min = cv2.getTrackbarPos("V min", "HSV mask")
    v_max = cv2.getTrackbarPos("V max", "HSV mask")
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower, upper


def set_trackbar_range(h: int, s: int, v: int, h_tol: int, s_tol: int, v_tol: int) -> None:
    """Positionne automatiquement les trackbars autour d'un point HSV."""
    cv2.setTrackbarPos("H min", "HSV mask", max(0, h - h_tol))
    cv2.setTrackbarPos("H max", "HSV mask", min(179, h + h_tol))
    cv2.setTrackbarPos("S min", "HSV mask", max(0, s - s_tol))
    cv2.setTrackbarPos("S max", "HSV mask", min(255, s + s_tol))
    cv2.setTrackbarPos("V min", "HSV mask", max(0, v - v_tol))
    cv2.setTrackbarPos("V max", "HSV mask", min(255, v + v_tol))


def main(cam_index: int, h_tol: int, s_tol: int, v_tol: int) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {cam_index}")

    create_trackbars()
    clicked_bgr = None
    clicked_hsv = None

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal clicked_bgr, clicked_hsv
        if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
            clicked_bgr = frame[y, x].tolist()
            clicked_hsv = hsv_frame[y, x].tolist()
            print(f"Pixel ({x}, {y}) -> BGR {clicked_bgr} | HSV {clicked_hsv}")
            set_trackbar_range(
                clicked_hsv[0], clicked_hsv[1], clicked_hsv[2], h_tol, s_tol, v_tol
            )

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("camera", on_mouse)

    frame = None
    hsv_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame non valide, arrêt.")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = read_trackbar_values()
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Affichage
        info = frame.copy()
        if clicked_bgr is not None and clicked_hsv is not None:
            txt = f"BGR {clicked_bgr} | HSV {clicked_hsv}"
            cv2.putText(info, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera", info)
        cv2.imshow("HSV mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuner HSV pour détecter le Thymio.")
    parser.add_argument("--cam", type=int, default=0, help="Index de la caméra (défaut 0)")
    parser.add_argument("--h_tol", type=int, default=10, help="Tolérance autour de H cliqué")
    parser.add_argument("--s_tol", type=int, default=60, help="Tolérance autour de S cliqué")
    parser.add_argument("--v_tol", type=int, default=60, help="Tolérance autour de V cliqué")
    args = parser.parse_args()
    main(cam_index=args.cam, h_tol=args.h_tol, s_tol=args.s_tol, v_tol=args.v_tol)

