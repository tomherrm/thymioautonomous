# Autonomous Navigation System for Thymio Robot

This project presents a complete autonomous navigation system for the Thymio robot, integrating computer vision, global path planning, and robust state estimation. The system enables the robot to navigate through an environment with static and dynamic obstacles using a top-down camera feed as its primary sensor.

## 🚀 Key Features
* **Computer Vision Pipeline**: Real-time detection and tracking of the robot, goal, and obstacles using color-based segmentation and homography for perspective correction.
* **Path Planning**: Implementation of the **A* algorithm** with a custom **Chebyshev distance** heuristic and diagonal movement penalties to ensure smooth, optimal trajectories.
* **Robust State Estimation**: An **Extended Kalman Filter (EKF)** that fuses wheel odometry with vision data to maintain accurate pose estimation even during sensor noise or brief camera occlusions.
* **Motion Control**: A **Pure Pursuit** algorithm for precise path following, combined with a dedicated "Alignment" mode for initial orientation.
* **Obstacle Avoidance & Recovery**: A reactive state machine that handles unexpected obstacles and "kidnapping" scenarios (detecting when the robot is lifted) using ground and proximity sensors.


## Technologies Used
* **Language**: Python
* **Libraries**: OpenCV (Vision), NumPy (Numerical computation), `tdmclient` (Robot communication)
* **Hardware**: Thymio II Robot, overhead camera.

