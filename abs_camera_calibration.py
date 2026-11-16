
"""
ABS CAMERA CALIBRATION
======================

Senior Project: Automated Ball/Strike System

Modes:
    python abs_camera_calibration.py board
    python abs_camera_calibration.py intrinsics
    python abs_camera_calibration.py pose

Install:
    pip install opencv-contrib-python numpy matplotlib

Controls:
    SPACE = capture calibration sample / save camera poses
    Q     = quit current mode
    R     = reset collected intrinsic samples

Coordinate conventions:
    World X = right across the board
    World Y = along the board
    World Z = upward from the printed board surface

    Camera poses are reported in world coordinates.

    Euler angles use the ZYX convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

IMPORTANT:
    The board must be placed in the same orientation
    relative to home plate every time.

    This program assumes the CENTER of the board is
    the world origin. If you want the origin at a
    particular point on home plate, add the appropriate
    board-to-field transformation.

    The program has not been validated with your
    physical cameras. Verify measurements before
    using them for ball/strike decisions.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SECTION 1: USER-ADJUSTABLE SETTINGS
# ============================================================

# Your two USB cameras.
# Camera 0 is normally the built-in webcam.
CAMERAS = {
    "cam1": 0,
    "cam2": 1,
}

# Requested camera resolution.
# Actual resolution depends on camera and USB bandwidth.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Windows DirectShow capture backend.
CAMERA_BACKEND = cv2.CAP_DSHOW

# Expected distance is advisory only.
# It does NOT force either camera to be at this distance.
EXPECTED_DISTANCE_FT = 15.0

# Physical board dimensions.
BOARD_SIZE_IN = 17.0
INCH_TO_M = 0.0254
BOARD_SIZE_M = BOARD_SIZE_IN * INCH_TO_M

# ChArUco board layout.
# 8 squares across and 8 squares down.
SQUARES_X = 8
SQUARES_Y = 8

# Since the board is exactly 17 inches wide,
# each square is 17 / 8 inches.
SQUARE_LENGTH_M = BOARD_SIZE_M / SQUARES_X

# Marker size is 75% of each square.
MARKER_LENGTH_M = SQUARE_LENGTH_M * 0.75

# Marker dictionary.
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_100

# Number of intrinsic calibration samples.
MIN_INTRINSIC_SAMPLES = 20

# Minimum ChArUco corners for a usable observation.
MIN_CORNERS = 6

# Maximum reprojection RMS allowed for a pose.
# This is a starting threshold, not a guarantee of
# sufficient 3D accuracy.
MAX_POSE_RMS_PX = 2.0

# Output directory.
OUTPUT = Path("abs_calibration")

# Intrinsic calibration filenames.
INTRINSIC_FILES = {
    "cam1": OUTPUT / "cam1_intrinsics.json",
    "cam2": OUTPUT / "cam2_intrinsics.json",
}

# Pose calibration output.
POSE_FILE = OUTPUT / "camera_poses.json"

# Live 3D visualization update interval.
PLOT_UPDATE_SECONDS = 0.15

# Set this to False if live plots slow camera capture.
ENABLE_3D_VISUALIZATION = True


# ============================================================
# SECTION 2: CREATE THE CHARUCO BOARD
# ============================================================

def create_board():
    """
    Create the ChArUco calibration board.

    OpenCV knows the exact 3D coordinates of every
    chessboard intersection based on the physical
    square dimensions.

    Those known coordinates are used for calibration.
    """

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT_ID
    )

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary
    )

    return board, dictionary


BOARD, DICTIONARY = create_board()


def generate_board_image():
    """
    Generate a high-resolution board image.

    The PNG itself has no guaranteed physical print
    size. Print it at EXACTLY 17 x 17 inches.

    Do not use 'fit to page' if it changes the size.
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    dpi = 300

    pixels = round(BOARD_SIZE_IN * dpi)

    image = BOARD.generateImage(
        (pixels, pixels),
        marginSize=0,
        borderBits=1
    )

    filename = OUTPUT / "charuco_17x17.png"

    cv2.imwrite(str(filename), image)

    print("\nChArUco board generated.")
    print(f"File: {filename}")
    print(f"Physical size: {BOARD_SIZE_IN} x {BOARD_SIZE_IN} inches")
    print(f"Squares: {SQUARES_X} x {SQUARES_Y}")
    print(f"Square width: {SQUARE_LENGTH_M * 1000:.3f} mm")
    print(f"Marker width: {MARKER_LENGTH_M * 1000:.3f} mm")
    print("\nPrint at 100% physical size.")
    print("Measure the printed squares before calibration.")
    print("Mount the print on a rigid, flat surface.")


# ============================================================
# SECTION 3: CAMERA CONNECTION
# ============================================================

def open_camera(camera_id):
    """
    Open one camera and request the selected resolution.

    MJPG is requested because it can reduce USB bandwidth
    compared with uncompressed video.

    The camera may ignore some requested settings.
    """

    cap = cv2.VideoCapture(
        camera_id,
        CAMERA_BACKEND
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open camera index {camera_id}"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(
        f"Camera {camera_id} opened: "
        f"{width:.0f} x {height:.0f}"
    )

    return cap


def open_both_cameras():
    """
    Open both cameras.

    If the second camera fails, close the first camera
    before reporting the error.
    """

    caps = {}

    try:
        for name, index in CAMERAS.items():
            caps[name] = open_camera(index)

    except Exception:
        for cap in caps.values():
            cap.release()
        raise

    return caps


def release_cameras(caps):
    """Release USB cameras and close OpenCV windows."""

    for cap in caps.values():
        cap.release()

    cv2.destroyAllWindows()


# ============================================================
# SECTION 4: DETECT CHARUCO CORNERS
# ============================================================

def detect_board(frame):
    """
    Detect ArUco markers and ChArUco corners.

    This version checks that valid corners exist before
    trying to draw them, preventing OpenCV crashes.
    """

    # Make a copy so we don't modify the original frame.
    preview = frame.copy()

    # Convert the image to grayscale for detection.
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Create the ChArUco detector.
    detector = cv2.aruco.CharucoDetector(BOARD)

    # Detect the board and its markers.
    corners, ids, marker_corners, marker_ids = (
        detector.detectBoard(gray)
    )
    #print(marker_ids)
    #print(marker_corners)
    
    # Draw ArUco markers only when valid markers exist.
    if (
        marker_ids is not None
        and marker_corners is not None
        and len(marker_ids) > 0
    ):
        cv2.aruco.drawDetectedMarkers(
            preview,
            marker_corners,
            marker_ids
        )

    # # Check that ChArUco corners and IDs are valid.
    # if (
    #     corners is not None
    #     and ids is not None
    #     and len(corners) > 0
    #     and len(ids) > 0
    #     and len(corners) == len(ids)
    # ):

    #     # Draw the detected corners.
    #     cv2.aruco.drawDetectedCornersCharuco(
    #         preview,
    #         corners,
    #         ids
    #     )

    else:
        # Return None when there are no valid corners.
        # This prevents later calibration functions
        # from trying to use empty detection arrays.
        corners = None
        ids = None

        cv2.putText(
            preview,
            "No ChArUco corners detected",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return corners, ids, preview


def valid_detection(corners, ids):
    """Check whether enough corners were detected."""

    return (
        corners is not None
        and ids is not None
        and len(ids) >= MIN_CORNERS
    )


# ============================================================
# SECTION 5: CAMERA MATRIX AND DISTORTION
# ============================================================

def save_intrinsics(name, camera_matrix, distortion,
                    image_size, rms):
    """
    Save the lens calibration.

    Camera matrix contains:
        fx, fy = focal lengths in pixels
        cx, cy = optical center in pixels

    Distortion coefficients describe lens distortion.
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "camera_name": name,
        "camera_index": CAMERAS[name],
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "reprojection_rms_px": float(rms),
        "board_size_m": BOARD_SIZE_M,
        "square_length_m": SQUARE_LENGTH_M,
        "marker_length_m": MARKER_LENGTH_M,
    }

    path = INTRINSIC_FILES[name]

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    print(f"Saved intrinsics: {path}")


def load_intrinsics(name):
    """
    Load previously calculated camera parameters.

    These are required for accurate pose estimation.
    """

    path = INTRINSIC_FILES[name]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing intrinsic calibration: {path}\n"
            "Run the intrinsics mode first."
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    K = np.array(
        data["camera_matrix"],
        dtype=np.float64
    )

    distortion = np.array(
        data["distortion"],
        dtype=np.float64
    )

    return K, distortion, data


# ============================================================
# SECTION 6: ROTATION MATRIX AND EULER ANGLES
# ============================================================

def rotation_matrix_to_euler(R):
    """
    Convert a rotation matrix into Euler angles.

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    The supplied matrix MUST represent camera-to-world
    orientation if camera angles relative to the field
    are desired.

    Returns:
        roll, pitch, yaw in degrees
    """

    sy = np.sqrt(
        R[0, 0] ** 2 + R[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        roll = np.arctan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy
        )

        yaw = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

    else:
        # Special case when Euler angles become ambiguous.
        roll = np.arctan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy
        )

        yaw = 0.0

    return np.degrees(
        [roll, pitch, yaw]
    )


# ============================================================
# SECTION 7: BOARD COORDINATE TRANSFORMATION
# ============================================================

def board_to_world_transform():
    """
    Define the world coordinate system.

    OpenCV's ChArUco board coordinates begin at a board
    corner, with the board lying in its local XY plane.

    Here we move the origin to the CENTER of the board.

    We also flip the board Y axis so that world Z points
    upward from the printed board surface.

    This is a rigid rotation with determinant +1.

    IMPORTANT:
        Place the printed board face-up with its orientation
        matching this coordinate convention.

        The world origin is at the printed surface,
        not necessarily at the physical home-plate surface.
    """

    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    # Board center in board-local coordinates.
    center_board = np.array([
        BOARD_SIZE_M / 2.0,
        BOARD_SIZE_M / 2.0,
        0.0
    ])

    # Translation makes the board center the world origin.
    t = -R @ center_board

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = t

    return T


BOARD_TO_WORLD = board_to_world_transform()


# ============================================================
# SECTION 8: ESTIMATE BOARD POSE
# ============================================================

def estimate_board_pose(corners, ids, K, distortion):
    """
    Calculate the board's position and orientation
    relative to a camera.

    solvePnP finds:
        R_cb = rotation from board coordinates to camera
        t_cb = translation from board origin to camera frame

    Equation:
        P_camera = R_cb @ P_board + t_cb
    """

    if not valid_detection(corners, ids):
        return None

    # Known 3D coordinates of all board chessboard corners.
    all_object_points = BOARD.getChessboardCorners()

    corner_ids = ids.flatten().astype(int)

    object_points = np.asarray(
        all_object_points[corner_ids],
        dtype=np.float64
    )

    image_points = np.asarray(
        corners.reshape(-1, 2),
        dtype=np.float64
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    R_cb, _ = cv2.Rodrigues(rvec)

    # Reproject the known corners to estimate fit quality.
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        distortion
    )

    projected = projected.reshape(-1, 2)

    error = projected - image_points

    rms = np.sqrt(
        np.mean(np.sum(error ** 2, axis=1))
    )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "R_cb": R_cb,
        "rms": float(rms),
        "corner_count": len(corner_ids),
    }


# ============================================================
# SECTION 9: CAMERA POSITION RELATIVE TO HOME PLATE
# ============================================================

def calculate_camera_world_pose(board_pose):
    """
    Convert the board pose into the camera's world pose.

    First invert the board-to-camera transformation:

        R_bc = R_cb.T
        C_board = -R_cb.T @ t_cb

    Then transform the camera position and orientation
    into the world coordinate system.

    The result gives:
        X, Y, Z
        Roll, Pitch, Yaw
        Camera-to-world rotation matrix
    """

    R_cb = board_pose["R_cb"]

    t_cb = board_pose["tvec"].reshape(3)

    # Invert board-to-camera transformation.
    R_bc = R_cb.T

    camera_position_board = -R_bc @ t_cb

    # Transform board coordinates into world coordinates.
    R_wb = BOARD_TO_WORLD[:3, :3]
    t_wb = BOARD_TO_WORLD[:3, 3]

    camera_position_world = (
        R_wb @ camera_position_board + t_wb
    )

    R_wc = R_wb @ R_bc

    roll, pitch, yaw = rotation_matrix_to_euler(
        R_wc
    )

    # OpenCV camera optical axis is +Z in camera coordinates.
    optical_axis_world = R_wc @ np.array(
        [0.0, 0.0, 1.0]
    )

    # Camera right axis is +X.
    right_axis_world = R_wc @ np.array(
        [1.0, 0.0, 0.0]
    )

    # Camera down axis is +Y.
    down_axis_world = R_wc @ np.array(
        [0.0, 1.0, 0.0]
    )

    return {
        "position_m": camera_position_world.tolist(),
        "roll_deg": float(roll),
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "rotation_camera_to_world": R_wc.tolist(),
        "optical_axis_world": optical_axis_world.tolist(),
        "right_axis_world": right_axis_world.tolist(),
        "down_axis_world": down_axis_world.tolist(),
        "reprojection_rms_px": board_pose["rms"],
        "detected_corners": board_pose["corner_count"],
    }


# ============================================================
# SECTION 10: 3D VISUALIZATION HELPERS
# ============================================================

def setup_3d_plot(title):
    """
    Create an interactive Matplotlib 3D window.

    You can rotate the plot using the mouse.
    """

    plt.ion()

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    fig.suptitle(title)

    return fig, ax


def draw_coordinate_axes(ax, origin, R, length=0.25):
    """
    Draw three local coordinate axes.

    Red   = X
    Green = Y
    Blue  = Z

    R defines the orientation of those axes.
    """

    origin = np.asarray(origin).reshape(3)

    colors = ["red", "green", "blue"]

    for i in range(3):

        direction = R[:, i] * length

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=colors[i],
            linewidth=2
        )


def draw_board(ax, R=None, t=None):
    """
    Draw the calibration board in 3D.

    R and t transform board-local coordinates into
    whichever coordinate system the plot uses.
    """

    if R is None:
        R = np.eye(3)

    if t is None:
        t = np.zeros(3)

    # Board corners in local coordinates.
    points = np.array([
        [0, 0, 0],
        [BOARD_SIZE_M, 0, 0],
        [BOARD_SIZE_M, BOARD_SIZE_M, 0],
        [0, BOARD_SIZE_M, 0],
        [0, 0, 0]
    ], dtype=float)

    transformed = (
        R @ points.T
    ).T + np.asarray(t).reshape(3)

    ax.plot(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        color="black",
        linewidth=2
    )

    # Draw the internal chessboard grid.
    for i in range(SQUARES_X + 1):

        x = i * SQUARE_LENGTH_M

        line = np.array([
            [x, 0, 0],
            [x, BOARD_SIZE_M, 0]
        ])

        line_world = (
            R @ line.T
        ).T + t

        ax.plot(
            line_world[:, 0],
            line_world[:, 1],
            line_world[:, 2],
            color="gray",
            linewidth=0.5
        )

    for j in range(SQUARES_Y + 1):

        y = j * SQUARE_LENGTH_M

        line = np.array([
            [0, y, 0],
            [BOARD_SIZE_M, y, 0]
        ])

        line_world = (
            R @ line.T
        ).T + t

        ax.plot(
            line_world[:, 0],
            line_world[:, 1],
            line_world[:, 2],
            color="gray",
            linewidth=0.5
        )


def set_equal_3d_axes(ax, points, minimum_radius=0.5):
    """
    Make X, Y, and Z use comparable visual scales.

    This prevents the plot from visually exaggerating
    one dimension relative to another.
    """

    points = np.asarray(points, dtype=float)

    center = np.mean(points, axis=0)

    spans = np.ptp(points, axis=0)

    radius = max(
        float(np.max(spans)) / 2.0,
        minimum_radius
    ) * 1.25

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_box_aspect((1, 1, 1))


# ============================================================
# SECTION 11: LIVE 3D BOARD VIEW FOR INTRINSICS
# ============================================================

def update_intrinsics_plot(ax, poses):
    """
    Show the board's 3D position relative to each camera.

    IMPORTANT:
        Each board pose is measured in its own camera
        coordinate system.

        Therefore, these are separate local views,
        not a common world-coordinate reconstruction.

    The plot uses the first available pose.
    """

    ax.clear()

    if not poses:
        ax.set_title(
            "Waiting for a valid board pose..."
        )
        return

    name = next(iter(poses))

    pose = poses[name]

    R = pose["R_cb"]
    t = pose["tvec"].reshape(3)

    # Draw the board relative to the camera.
    draw_board(ax, R, t)

    # Camera is the origin in this coordinate system.
    ax.scatter(
        [0], [0], [0],
        color="blue",
        s=90,
        label=name
    )

    draw_coordinate_axes(
        ax,
        np.zeros(3),
        np.eye(3),
        length=0.2
    )

    # Draw the board's own coordinate axes.
    draw_coordinate_axes(
        ax,
        t,
        R,
        length=0.2
    )

    ax.plot(
        [0, t[0]],
        [0, t[1]],
        [0, t[2]],
        linestyle="--",
        color="orange"
    )

    ax.set_title(
        f"Board relative to {name}\n"
        f"Distance: {np.linalg.norm(t):.2f} m"
    )

    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Y (m)")
    ax.set_zlabel("Camera Z (m)")

    points = np.vstack([
        np.zeros(3),
        t
    ])

    set_equal_3d_axes(ax, points)

    ax.legend(loc="upper right")


# ============================================================
# SECTION 12: 3D CAMERA VISUALIZATION
# ============================================================

def update_camera_plot(ax, results):
    """
    Draw the calibration board and both cameras.

    The board center is the world origin.

    Each camera is shown at its calculated X, Y, Z.

    Colored coordinate axes show orientation.
    Dashed lines show the camera optical axes.
    """

    ax.clear()

    # Convert the board's local coordinates into world.
    R_wb = BOARD_TO_WORLD[:3, :3]
    t_wb = BOARD_TO_WORLD[:3, 3]

    draw_board(ax, R_wb, t_wb)

    draw_coordinate_axes(
        ax,
        np.zeros(3),
        np.eye(3),
        length=0.3
    )

    all_points = [np.zeros(3)]

    colors = {
        "cam1": "blue",
        "cam2": "orange"
    }

    for name, pose in results.items():

        position = np.array(
            pose["position_m"]
        )

        R_wc = np.array(
            pose["rotation_camera_to_world"]
        )

        optical_axis = np.array(
            pose["optical_axis_world"]
        )

        all_points.append(position)

        color = colors.get(name, "purple")

        # Camera location.
        ax.scatter(
            position[0],
            position[1],
            position[2],
            color=color,
            s=100,
            label=name
        )

        # Camera coordinate axes.
        draw_coordinate_axes(
            ax,
            position,
            R_wc,
            length=0.4
        )

        # Optical axis line.
        end = position + optical_axis * 1.0

        ax.plot(
            [position[0], end[0]],
            [position[1], end[1]],
            [position[2], end[2]],
            linestyle="--",
            color=color,
            linewidth=2
        )

        # Line from camera to home plate.
        ax.plot(
            [position[0], 0],
            [position[1], 0],
            [position[2], 0],
            linestyle=":",
            color="gray"
        )

        # Label camera with XYZ and angles.
        label = (
            f"{name}\n"
            f"X={position[0]:.2f} m\n"
            f"Y={position[1]:.2f} m\n"
            f"Z={position[2]:.2f} m\n"
            f"R={pose['roll_deg']:.1f}° "
            f"P={pose['pitch_deg']:.1f}° "
            f"Y={pose['yaw_deg']:.1f}°"
        )

        ax.text(
            position[0],
            position[1],
            position[2] + 0.2,
            label,
            fontsize=8
        )

    ax.set_title(
        "ABS Camera Positions Relative to Home Plate"
    )

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_zlabel("World Z (m)")

    set_equal_3d_axes(
        ax,
        np.array(all_points),
        minimum_radius=1.0
    )

    ax.legend(loc="upper right")


# ============================================================
# SECTION 13: INTRINSIC CALIBRATION
# ============================================================

def calibrate_intrinsics():
    """
    Calibrate both camera lenses.

    Procedure:
        1. Open both cameras.
        2. Show live video feeds.
        3. Detect ChArUco corners.
        4. Press SPACE to save a sample.
        5. Move/tilt the board between samples.
        6. Collect at least 20 samples per camera.
        7. Press Q to finish and calculate intrinsics.

    If existing intrinsics are available, they are used
    to estimate a live 3D board pose.

    On the first calibration, the 3D pose becomes
    available after intrinsic parameters are estimated.
    """

    caps = open_both_cameras()

    samples = {
        name: {
            "corners": [],
            "ids": [],
            "size": None
        }
        for name in CAMERAS
    }

    # Existing calibration allows live pose estimation.
    existing_intrinsics = {}

    for name in CAMERAS:
        try:
            K, dist, _ = load_intrinsics(name)
            existing_intrinsics[name] = (K, dist)
            print(f"Loaded existing intrinsics for {name}")
        except FileNotFoundError:
            print(
                f"No existing intrinsics for {name}. "
                "Collect samples first."
            )

    fig = None
    ax = None

    if ENABLE_3D_VISUALIZATION:
        fig, ax = setup_3d_plot(
            "Live ChArUco Board Pose"
        )

    last_plot_update = 0

    try:

        while True:

            previews = {}
            detections = {}
            live_poses = {}

            for name, cap in caps.items():

                ok, frame = cap.read()

                if not ok:
                    print(f"Frame read failed: {name}")
                    continue

                corners, ids, preview = detect_board(
                    frame
                )

                detections[name] = (
                    corners,
                    ids,
                    frame.shape
                )

                if valid_detection(corners, ids):

                    # Display detected corner count.
                    cv2.putText(
                        preview,
                        f"Corners: {len(ids)}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # Estimate live board pose if intrinsics
                    # have already been calibrated.
                    if name in existing_intrinsics:

                        K, dist = existing_intrinsics[name]

                        pose = estimate_board_pose(
                            corners,
                            ids,
                            K,
                            dist
                        )

                        if pose is not None:
                            live_poses[name] = pose

                            cv2.drawFrameAxes(
                                preview,
                                K,
                                dist,
                                pose["rvec"],
                                pose["tvec"],
                                SQUARE_LENGTH_M * 2
                            )

                count = len(
                    samples[name]["corners"]
                )

                cv2.putText(
                    preview,
                    f"{name}: {count} samples",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                previews[name] = preview

            # Show both camera windows.
            for name, preview in previews.items():
                cv2.imshow(
                    f"Intrinsics - {name}",
                    preview
                )

            # Update the 3D plot periodically.
            now = time.time()

            if (
                fig is not None
                and now - last_plot_update
                >= PLOT_UPDATE_SECONDS
            ):

                update_intrinsics_plot(
                    ax,
                    live_poses
                )

                fig.canvas.draw_idle()
                plt.pause(0.001)

                last_plot_update = now

            key = cv2.waitKey(1) & 0xFF

            # SPACE: capture one sample per camera.
            if key == 32:

                for name in CAMERAS:

                    if name not in detections:
                        continue

                    corners, ids, shape = detections[name]

                    if not valid_detection(corners, ids):
                        print(
                            f"{name}: board not detected"
                        )
                        continue

                    samples[name]["corners"].append(
                        corners.copy()
                    )

                    samples[name]["ids"].append(
                        ids.copy()
                    )

                    samples[name]["size"] = (
                        shape[1],
                        shape[0]
                    )

                    print(
                        f"{name}: captured sample "
                        f"{len(samples[name]['corners'])}"
                    )

            # R: clear all captured samples.
            elif key == ord("r"):

                for name in CAMERAS:
                    samples[name]["corners"].clear()
                    samples[name]["ids"].clear()

                print("Intrinsic samples reset.")

            # Q: finish collecting samples.
            elif key == ord("q"):
                break

    finally:
        release_cameras(caps)

        if fig is not None:
            plt.close(fig)

    # Calculate intrinsics for each camera.
    for name in CAMERAS:

        data = samples[name]

        count = len(data["corners"])

        if count < MIN_INTRINSIC_SAMPLES:

            print(
                f"{name}: only {count} samples. "
                f"Need at least {MIN_INTRINSIC_SAMPLES}."
            )

            continue

        print(f"\nCalibrating {name}...")

        # Convert ChArUco IDs into known 3D points.
        object_points = []
        image_points = []

        all_board_points = BOARD.getChessboardCorners()

        for corners, ids in zip(
            data["corners"],
            data["ids"]
        ):

            selected_ids = ids.flatten().astype(int)

            obj = np.asarray(
                all_board_points[selected_ids],
                dtype=np.float32
            )

            img = np.asarray(
                corners.reshape(-1, 2),
                dtype=np.float32
            )

            object_points.append(obj)
            image_points.append(img)

        # Estimate camera matrix and lens distortion.
        rms, K, dist, rvecs, tvecs = (
            cv2.calibrateCamera(
                object_points,
                image_points,
                data["size"],
                None,
                None
            )
        )

        save_intrinsics(
            name,
            K,
            dist,
            data["size"],
            rms
        )

        print(f"\n{name} intrinsic calibration:")
        print("Camera matrix:")
        print(K)
        print("Distortion:")
        print(dist.reshape(-1))
        print(f"RMS: {rms:.3f} pixels")


# ============================================================
# SECTION 14: POSE CALIBRATION
# ============================================================

def calibrate_camera_poses():
    """
    Calculate both cameras' positions and orientations.

    Procedure:
        1. Place the board flat over home plate.
        2. Keep the board aligned with the field.
        3. Open both cameras.
        4. Detect the ChArUco corners.
        5. Calculate camera positions and rotations.
        6. Show both cameras in a 3D plot.
        7. Press SPACE to save the calibration.

    The cameras must not move after calibration.
    """

    # Load lens calibration for both cameras.
    intrinsics = {}

    for name in CAMERAS:
        K, dist, data = load_intrinsics(name)
        intrinsics[name] = (K, dist, data)

    caps = open_both_cameras()

    fig = None
    ax = None

    if ENABLE_3D_VISUALIZATION:
        fig, ax = setup_3d_plot(
            "ABS 3D Camera Pose Calibration"
        )

    last_plot_update = 0

    print("\nPOSE CALIBRATION")
    print("----------------")
    print("Place the 17 x 17-inch board over home plate.")
    print("Make sure the board is flat and correctly aligned.")
    print("Press SPACE when both cameras detect the board.")
    print("Press Q to quit.")

    try:

        while True:

            results = {}

            for name, cap in caps.items():

                ok, frame = cap.read()

                if not ok:
                    continue

                K, dist, intrinsic_data = intrinsics[name]

                expected_size = tuple(
                    intrinsic_data["image_size"]
                )

                actual_size = (
                    frame.shape[1],
                    frame.shape[0]
                )

                if actual_size != expected_size:
                    raise RuntimeError(
                        f"{name}: image resolution changed. "
                        f"Expected {expected_size}, "
                        f"received {actual_size}. "
                        "Recalibrate or restore resolution."
                    )

                corners, ids, preview = detect_board(
                    frame
                )

                if valid_detection(corners, ids):

                    board_pose = estimate_board_pose(
                        corners,
                        ids,
                        K,
                        dist
                    )

                    if board_pose is not None:

                        cv2.drawFrameAxes(
                            preview,
                            K,
                            dist,
                            board_pose["rvec"],
                            board_pose["tvec"],
                            SQUARE_LENGTH_M * 2
                        )

                        camera_pose = (
                            calculate_camera_world_pose(
                                board_pose
                            )
                        )

                        results[name] = camera_pose

                        rms = camera_pose[
                            "reprojection_rms_px"
                        ]

                        status = (
                            "GOOD"
                            if rms <= MAX_POSE_RMS_PX
                            else "HIGH ERROR"
                        )

                        cv2.putText(
                            preview,
                            f"Pose: {status} RMS={rms:.2f}px",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (
                                (0, 255, 0)
                                if status == "GOOD"
                                else (0, 0, 255)
                            ),
                            2
                        )

                        position = camera_pose["position_m"]

                        cv2.putText(
                            preview,
                            (
                                f"XYZ: "
                                f"{position[0]:.2f}, "
                                f"{position[1]:.2f}, "
                                f"{position[2]:.2f} m"
                            ),
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                cv2.imshow(
                    f"Pose - {name}",
                    preview
                )

            # Live 3D visualization.
            now = time.time()

            if (
                fig is not None
                and now - last_plot_update
                >= PLOT_UPDATE_SECONDS
            ):

                update_camera_plot(
                    ax,
                    results
                )

                fig.canvas.draw_idle()
                plt.pause(0.001)

                last_plot_update = now

            key = cv2.waitKey(1) & 0xFF

            if key == 32:

                # Both cameras must have valid poses.
                if len(results) != len(CAMERAS):

                    print(
                        "Both cameras must detect "
                        "the calibration board."
                    )

                    continue

                # Reject obviously poor reprojection fits.
                bad = [
                    name
                    for name, pose in results.items()
                    if pose["reprojection_rms_px"]
                    > MAX_POSE_RMS_PX
                ]

                if bad:
                    print(
                        "Calibration rejected due to "
                        f"high reprojection error: {bad}"
                    )
                    continue

                save_camera_poses(results)

                # Print camera results.
                print_camera_results(results)

                # Leave a final 3D plot open.
                if fig is not None:
                    update_camera_plot(ax, results)
                    fig.canvas.draw_idle()
                    plt.pause(0.1)

                break

            elif key == ord("q"):
                break

    finally:
        release_cameras(caps)

    # Keep the final plot available for inspection.
    if fig is not None:
        plt.ioff()
        plt.show()


# ============================================================
# SECTION 15: SAVE CAMERA POSITIONS AND ANGLES
# ============================================================

def save_camera_poses(results):
    """
    Save all camera pose information to JSON.

    This includes:
        X, Y, Z position
        Roll, pitch, yaw
        Camera-to-world rotation matrix
        Optical-axis direction
        Reprojection error

    The original intrinsic calibration files also remain
    available for the ball-tracking program.
    """

    OUTPUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "units": "meters",
        "angle_units": "degrees",
        "euler_convention": "ZYX_camera_to_world",
        "world_origin": "center_of_printed_board",
        "board_size_inches": BOARD_SIZE_IN,
        "board_to_world": BOARD_TO_WORLD.tolist(),
        "expected_distance_ft_advisory":
            EXPECTED_DISTANCE_FT,
        "cameras": results
    }

    POSE_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved camera poses to: {POSE_FILE}")


def print_camera_results(results):
    """
    Print human-readable camera positions and angles.
    """

    print("\n" + "=" * 55)
    print("ABS CAMERA CALIBRATION RESULTS")
    print("=" * 55)

    for name, pose in results.items():

        x, y, z = pose["position_m"]

        print(f"\n{name.upper()}")
        print("-" * 35)

        print(f"X position: {x:.3f} m")
        print(f"Y position: {y:.3f} m")
        print(f"Z position: {z:.3f} m")

        print(
            f"Roll:  {pose['roll_deg']:.2f} degrees"
        )

        print(
            f"Pitch: {pose['pitch_deg']:.2f} degrees"
        )

        print(
            f"Yaw:   {pose['yaw_deg']:.2f} degrees"
        )

        distance = np.linalg.norm(
            pose["position_m"]
        )

        print(
            f"Distance from board center: "
            f"{distance:.3f} m"
        )

        print(
            f"Distance: "
            f"{distance / 0.3048:.2f} ft"
        )

        print(
            f"Reprojection RMS: "
            f"{pose['reprojection_rms_px']:.3f} px"
        )

        print("\nCamera-to-world rotation matrix:")

        print(
            np.array(
                pose["rotation_camera_to_world"]
            )
        )

    print("\n" + "=" * 55)


# ============================================================
# SECTION 16: MAIN PROGRAM
# ============================================================

def main():
    """
    Select which calibration mode to run.
    """

    parser = argparse.ArgumentParser(
        description="ABS Two-Camera Calibration"
    )

    parser.add_argument(
        "mode",
        choices=[
            "board",
            "intrinsics",
            "pose"
        ],
        help="Calibration mode"
    )

    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\nABS CAMERA CALIBRATION")
    print("======================")
    print(f"Board: {BOARD_SIZE_IN} x {BOARD_SIZE_IN} inches")
    print(f"Expected distance: {EXPECTED_DISTANCE_FT} ft")
    print(f"Camera indices: {CAMERAS}")

    if args.mode == "board":

        generate_board_image()

    elif args.mode == "intrinsics":

        calibrate_intrinsics()

    elif args.mode == "pose":

        calibrate_camera_poses()


if __name__ == "__main__":
    main()
