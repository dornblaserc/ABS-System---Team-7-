import os
import re
import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def natural_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def prep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)  # suppress sensor noise

def draw_callout(img, text, target_pt, text_anchor, align="left", color=(0, 255, 255), scale=0.55, thickness=2):
    """Draws a coordinate label aligned to text_anchor with a leader line pointing to target_pt."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = img.shape[:2]

    tx, ty = text_anchor
    if align == "center":
        tx -= tw // 2
    elif align == "right":
        tx -= tw

    # Clamp inside image bounds
    tx = max(8, min(tx, w - tw - 8))
    ty = max(th + 8, min(ty, h - baseline - 8))

    # Leader line from point to the closest point on the text box
    line_end_x = max(tx, min(target_pt[0], tx + tw))
    line_end_y = ty - th - 4 if target_pt[1] < ty else ty + baseline + 4
    cv2.line(img, target_pt, (line_end_x, line_end_y), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, target_pt, (line_end_x, line_end_y), color, 1, cv2.LINE_AA)

    # Dark background pad + text for readability
    cv2.rectangle(img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + baseline + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

def annotate_blur(mask, min_area):
    sub_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)

        # 1. Draw high-visibility contour outline
        cv2.drawContours(sub_view, [c], -1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.drawContours(sub_view, [c], -1, (0, 255, 0), 3, cv2.LINE_AA)

        # 2. Fit line and orient vector consistently (left-to-right or top-to-bottom)
        vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if (abs(vx) >= abs(vy) and vx < 0) or (abs(vy) > abs(vx) and vy < 0):
            vx, vy = -vx, -vy

        pts = c.reshape(-1, 2)
        projections = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        t_min, t_max = projections.min(), projections.max()

        pt_start = (int(x0 + t_min * vx), int(y0 + t_min * vy))
        pt_end   = (int(x0 + t_max * vx), int(y0 + t_max * vy))
        center   = (int(x0), int(y0))

        # Draw motion axis line + markers
        cv2.line(sub_view, pt_start, pt_end, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(sub_view, pt_start, pt_end, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(sub_view, pt_start, 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(sub_view, pt_end, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(sub_view, center, 5, (255, 0, 255), -1, cv2.LINE_AA)

        # 3. Place labels in separated zones around the streak
        _, _, bw, bh = cv2.boundingRect(c)
        pad_y = max(45, bh // 2 + 30)

        if abs(vx) >= abs(vy):  # Mostly horizontal / diagonal motion
            draw_callout(sub_view, f"C: {center}", center, (center[0], center[1] - pad_y), align="center", color=(0, 255, 0))
            draw_callout(sub_view, f"S: {pt_start}", pt_start, (pt_start[0] - 25, pt_start[1] + pad_y), align="right", color=(255, 255, 0))
            draw_callout(sub_view, f"E: {pt_end}", pt_end, (pt_end[0] + 25, pt_end[1] + pad_y), align="left", color=(0, 255, 255))
        else:  # Mostly vertical motion
            draw_callout(sub_view, f"S: {pt_start}", pt_start, (pt_start[0], pt_start[1] - 40), align="center", color=(255, 255, 0))
            draw_callout(sub_view, f"C: {center}", center, (center[0] + bw // 2 + 35, center[1]), align="left", color=(0, 255, 0))
            draw_callout(sub_view, f"E: {pt_end}", pt_end, (pt_end[0], pt_end[1] + 55), align="center", color=(0, 255, 255))

    return sub_view

def detect_motion(source, output_dir, thresh=25, min_area=500):
    os.makedirs(output_dir, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def diff_and_save(gray, prev, out_name):
        diff = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=2)  # merge fragmented blobs
        out_img = annotate_blur(mask, min_area)
        cv2.imwrite(os.path.join(output_dir, out_name), out_img)

    # Case 1: Folder of images or list/tuple of image paths
    if isinstance(source, (list, tuple)) or os.path.isdir(source):
        if isinstance(source, (list, tuple)):
            paths = list(source)
        else:
            files = sorted(
                [f for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTS)],
                key=natural_key
            )
            paths = [os.path.join(source, f) for f in files]

        if not paths:
            raise ValueError("No valid images found.")

        if len(paths) == 1:
            source = paths[0]
        else:
            first = cv2.imread(paths[0])
            if first is None:
                raise IOError(f"Could not read {paths[0]}")
            prev = prep(first)

            for p in paths[1:]:
                frame = cv2.imread(p)
                if frame is None:
                    continue
                gray = prep(frame)
                diff_and_save(gray, prev, os.path.basename(p))
                prev = gray
            return

    # Case 2: Single image file
    if os.path.isfile(source) and source.lower().endswith(IMAGE_EXTS):
        frame = cv2.imread(source)
        if frame is None:
            raise IOError(f"Cannot read image {source}")
        gray = prep(frame)
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=2)
        out_img = annotate_blur(mask, min_area)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(source)), out_img)
        return

    # Case 3: Video file
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open {source}")

    try:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Could not read first frame")
        prev = prep(frame)
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = prep(frame)
            diff_and_save(gray, prev, f"frame_{frame_idx:04d}.png")
            prev = gray
            frame_idx += 1
    finally:
        cap.release()

detect_motion(
    source="C:/Users/Student/Downloads/ABS-System---Team-7-/test.mp4",
    output_dir="C:/Users/Student/Downloads/ABS-System---Team-7-/frames_out"
)