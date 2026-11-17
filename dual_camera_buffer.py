"""
DUAL CAMERA CAPTURE WITH LED SYNC  (Windows, Python + OpenCV)
=============================================================

WHAT THIS PROGRAM DOES (the big picture)
----------------------------------------
You have two cameras pointed at the same scene (for example, a pitch heading
toward home plate). To work out where a ball is in 3D, you need to compare
what both cameras saw at the SAME MOMENT. That is hard because:

  * The two cameras are separate devices with no shared clock, so they do
    not take their pictures at exactly the same instant.
  * Windows and USB add small, random delays before the pictures reach us.

This program deals with that in four steps:

  1. RECORD CONTINUOUSLY. Both cameras run non-stop. We keep only the most
     recent couple of seconds of pictures in memory (a "ring buffer"), so
     we never have to start or stop recording at the right moment.

  2. TIMESTAMP EVERY PICTURE. The instant each picture arrives we write down
     the time, using the computer's clock. Now every picture from both
     cameras can be placed on one shared timeline.

  3. MEASURE THE LEFTOVER DIFFERENCE WITH AN LED. If we flash a light that
     both cameras can see, we can compare WHEN each camera reports seeing
     it. Any difference tells us how far apart the two cameras' timestamps
     are. That number is called the "offset".

  4. SAVE PAIRED PICTURES. When you press a key, we save the last
     second or so from both cameras. For each picture from camera 0 we find
     the picture from camera 1 that is closest in time (after correcting by
     the offset) and save them together as a "pair".

HOW TO USE IT
-------------
Click on the preview window first (so it receives your key presses), then:

    r  -  Pick the LED region: drag a box around the LED in each camera's
          picture, then press ENTER.
    c  -  Calibrate: flash the LED on and off 10+ times during the countdown.
    s  -  Save the last SAVE_SECONDS of pictures from both cameras (paired).
    t  -  Timing check: shows how steady the frame timing is for each camera.
    q  -  Quit.

INSTALL (one time, in a Windows terminal)
-----------------------------------------
    pip install opencv-python numpy
"""

# ---------------------------------------------------------------------------
# IMPORTS
# "Importing" means loading a toolbox of ready-made code written by others.
# ---------------------------------------------------------------------------
import os                      # work with folders and file paths
import time                    # read the computer's clock
import csv                     # write spreadsheet-style (.csv) files
import bisect                  # quickly find the nearest number in a sorted list
import json                    # save/load settings in a simple text format
import threading               # run several jobs at the same time
from collections import deque  # a list with a fixed maximum length (the ring buffer)

import cv2                     # OpenCV: talks to cameras and handles images
import numpy as np             # NumPy: fast math on lists of numbers
import frame_subtraction as frame_sub
# ---------------------------------------------------------------------------
# CONFIG - the settings you are most likely to change.
# Everything else in the file reads these values.
# ---------------------------------------------------------------------------

# Windows numbers every camera, starting at 0. Index 0 is often the laptop's
# built-in webcam, so your two ELP cameras are probably 1 and 2. If the wrong
# cameras open, change these numbers.
CAMERA_INDEXES = (1, 2)

# Picture size in pixels. These are 16:9 sizes the ELP camera really supports.
# Bigger pictures show more detail but need more USB bandwidth and memory.
WIDTH, HEIGHT = 640, 360

# Frames per second (fps) we ASK the camera for. The camera may deliver a
# little less, depending on lighting, USB speed and exposure.
TARGET_FPS = 230

# How many frames per second to assume when sizing the memory buffer.
# Keep this about equal to the fps you actually get.
BUFFER_FPS = 60

# How many seconds of recent video to keep in memory at all times.
BUFFER_SECONDS = 2

# How many seconds of video to write to disk when you press 's'.
# Must be no longer than BUFFER_SECONDS.
SAVE_SECONDS = 1.5

# "Exposure" is how long the camera collects light for each picture.
#  - If left on AUTOMATIC, the camera makes pictures slower in dim rooms,
#    which wrecks the frame rate. So we use MANUAL exposure instead.
#  - EXPOSURE_VALUE is on a strange scale used by Windows (DirectShow):
#    each step down HALVES the time. Examples:
#        -5 ~ 31 ms    -6 ~ 16 ms    -7 ~ 8 ms    -8 ~ 4 ms
#    A frame can't be shorter than its exposure, so for high fps you need a
#    small number. Smaller exposure = darker picture, but less motion blur.
MANUAL_EXPOSURE = True
EXPOSURE_VALUE = -6

# "Backend" = which Windows camera system OpenCV uses. DSHOW (DirectShow)
# works with these cameras. If it ever misbehaves, try cv2.CAP_MSMF instead.
BACKEND = cv2.CAP_DSHOW

# MJPG is a compressed video format. These cameras can only reach high frame
# rates in MJPG; the uncompressed alternative is limited to about 10 fps.
USE_MJPG = True

# How big the preview window is, compared to the real picture (0.8 = 80%).
PREVIEW_SCALE = 0.8

# Folder where saved clips go (created automatically if it doesn't exist).
OUTPUT_DIR = "frames_in"

# ---- LED calibration settings ----
CAL_SECONDS = 5.0       # length of the calibration countdown (10 is better)
MIN_CONTRAST = 15.0     # how much brighter (0-255 scale) the LED must make the
                        # box before we believe it was a real flash
CAL_FILE = "calibration.json"   # where the LED boxes and offset are remembered


# ===========================================================================
# CAMERA READER
# ===========================================================================

class CamThread(threading.Thread):
    """
    One of these objects looks after ONE camera.

    A "thread" is a separate worker that runs at the same time as the rest of
    the program. We give each camera its own worker so that:
      * each camera is read as fast as it can deliver pictures, and
      * the preview window (drawn by the main program) never slows capture.

    Each worker keeps its own "ring buffer" of recent pictures. A ring
    buffer is a list with a fixed size: once it is full, adding a new
    picture automatically pushes out the oldest one. So it always holds
    "the last few seconds".
    """

    def __init__(self, name, index):
        super().__init__(daemon=True)   # daemon=True: worker stops when program exits
        self.name = name                # a label like "cam0", used in messages
        self.index = index              # the Windows camera number

        # Open the camera.
        self.cap = cv2.VideoCapture(index, BACKEND)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        # ---- Tell the camera what mode we want ----
        # ORDER MATTERS here! On Windows, if we ask for MJPG first, the camera
        # silently ignores us and drops to a slow 10 fps mode. Setting the
        # size and fps first and the MJPG format LAST makes it work.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        if USE_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if MANUAL_EXPOSURE:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 means "manual" on DirectShow
            self.cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)

        # Ask the camera what it actually accepted (it may refuse some
        # settings). These get printed at startup so you can check them.
        self.auto_exp_readback = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        self.exp_readback = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # ---- The ring buffer ----
        # Each entry will be a pair: (time the picture arrived, the picture).
        # maxlen is how many pictures fit: seconds x frames-per-second.
        maxlen = int(BUFFER_SECONDS * BUFFER_FPS)
        self.buf = deque(maxlen=maxlen)

        # A "lock" stops two workers touching the buffer at the same instant
        # (for example, the camera adding a picture while we copy the list).
        self.lock = threading.Lock()

        self.running = True        # set to False to tell the worker to stop
        self.frame_count = 0       # total pictures received so far
        self.measured_fps = 0.0    # fps we are really getting (updated each second)
        self.latest = None         # the newest picture (used for the preview)

        # ---- LED calibration state ----
        # roi = "region of interest": the box around the LED, as
        # (x, y, width, height) in pixels. None means not chosen yet.
        self.roi = None
        # While calibrating, this holds (time, brightness-in-the-box) readings.
        # None means "not calibrating right now".
        self.cal_samples = None

    def run(self):
        """
        This is the worker's main loop. It starts when we call .start() and
        repeats until .running becomes False.
        """
        fps_t0 = time.perf_counter()   # start of the current 1-second fps measurement
        fps_n = 0                      # pictures received in this second

        while self.running:
            # read() waits until the camera delivers its next picture.
            # (We use read() instead of grab()+retrieve() because on this
            # Windows driver grab() returns immediately, which would make the
            # timestamps meaningless.)
            ok, frame = self.cap.read()

            # Write down the time RIGHT AFTER the picture arrives. We use
            # perf_counter(), a very precise stopwatch clock that is the same
            # for both cameras, so their timestamps can be compared.
            t = time.perf_counter()

            if not ok:
                time.sleep(0.001)      # nothing arrived; wait a moment, try again
                continue

            # Store (time, picture) in the ring buffer. The lock makes sure
            # nobody is reading the list while we change it.
            with self.lock:
                self.buf.append((t, frame))
            self.latest = frame
            self.frame_count += 1

            # If we are calibrating, also measure how bright the LED box is
            # in this picture, and remember it together with the time.
            roi, samples = self.roi, self.cal_samples
            if roi is not None and samples is not None:
                x, y, w, h = roi
                # frame[y:y+h, x:x+w] cuts out just the box; .mean() averages
                # all its pixel values into one brightness number (0-255).
                samples.append((t, float(frame[y:y + h, x:x + w].mean())))

            # Work out real frames-per-second, refreshed once per second.
            fps_n += 1
            if t - fps_t0 >= 1.0:
                self.measured_fps = fps_n / (t - fps_t0)
                fps_t0, fps_n = t, 0

    def snapshot(self, seconds):
        """
        Return a copy of the pictures received in the last `seconds` seconds,
        as a list of (time, picture) pairs, oldest first.
        """
        now = time.perf_counter()
        with self.lock:
            items = list(self.buf)   # copy the buffer while nobody changes it
        return [(t, f) for (t, f) in items if now - t <= seconds]

    def stop(self):
        """Stop the worker and release the camera so other programs can use it."""
        self.running = False
        self.join(timeout=1.0)   # wait up to 1 second for the worker to finish
        self.cap.release()


# ===========================================================================
# LED FLASH DETECTION
# ===========================================================================
def find_rising_edges(samples):
    """
    Find the moments the LED turned ON.

    Input:  samples = a list of (time, brightness) readings from the LED box.
    Output: (list of "turn-on" times, contrast)

    How it works, in plain terms:
      1. Work out the dim level (LED off) and the bright level (LED on).
      2. Rescale every reading so dim = 0.0 and bright = 1.0.
      3. Each time the reading climbs past 0.5 (halfway), that's a turn-on.
      4. The cameras only take a picture every ~16 ms, so the true moment
         lies BETWEEN two pictures. We estimate it by drawing a straight line
         between the last "dim" reading and the first "bright" reading and
         finding where that line crosses the halfway mark. This gives a
         time more precise than one frame.
    """
    # Need a reasonable number of readings to say anything.
    if len(samples) < 10:
        return [], 0.0

    t = np.array([s[0] for s in samples])   # times
    b = np.array([s[1] for s in samples])   # brightness values

    lo = np.percentile(b, 10)   # a typical "LED off" brightness
    hi = b.max()                # the brightest reading = "LED on"
    contrast = hi - lo          # how big the flash was

    # If the box barely changed brightness, there was no real flash.
    if contrast < MIN_CONTRAST:
        return [], float(contrast)

    norm = (b - lo) / contrast  # rescale so off ~ 0.0 and on ~ 1.0
    edges = []

    # "on" remembers whether the LED is currently considered lit.
    # If it's already lit in the very first reading, we skip that one,
    # because we missed its true turn-on moment.
    on = norm[0] >= 0.5

    for i in range(1, len(norm)):
        if not on and norm[i] >= 0.5:
            # The light just crossed halfway. Estimate exactly when.
            d = norm[i] - norm[i - 1]
            frac = (0.5 - norm[i - 1]) / d if d > 0 else 1.0
            edges.append(float(t[i - 1] + frac * (t[i] - t[i - 1])))
            on = True
        elif on and norm[i] < 0.3:
            # The light dropped well below halfway, so it's off again.
            # Using 0.3 here (not 0.5) stops small noise from counting as
            # extra flashes.
            on = False

    return edges, float(contrast)


def finish_calibration(cams, state):
    """
    Called when the calibration countdown ends. Looks at the brightness
    readings from both cameras, finds the flashes, and works out the offset
    between the two cameras.
    """
    edges = []   # one list of flash times per camera
    for c in cams:
        samples = c.cal_samples or []
        c.cal_samples = None          # stop recording brightness
        e, contrast = find_rising_edges(samples)
        print(f"  {c.name}: {len(samples)} samples, contrast {contrast:.0f}, "
              f"{len(e)} flash(es) detected")
        edges.append(e)

    # Check 1: both cameras must have seen at least one flash.
    if not edges[0] or not edges[1]:
        print("Calibration FAILED: a camera saw no flash. Make sure the LED is "
              "inside the red box, bright enough (contrast >= "
              f"{MIN_CONTRAST:.0f}), and flashes on/off clearly.")
        return

    # Check 2: both cameras must have seen the SAME NUMBER of flashes,
    # otherwise we can't tell which flash matches which.
    if len(edges[0]) != len(edges[1]):
        print("Calibration FAILED: the cameras saw a different number of flashes "
              f"({len(edges[0])} vs {len(edges[1])}). Try again with clean, "
              "well-separated flashes.")
        return

    # For each flash, compare the time camera 1 reported with the time
    # camera 0 reported. Positive = camera 1's clock reads later.
    offsets = [tb - ta for ta, tb in zip(edges[0], edges[1])]

    # Use the MEDIAN (middle value) so one bad flash can't skew the result.
    med = float(np.median(offsets))

    # Spread = biggest minus smallest. A small spread means consistent results.
    spread = (max(offsets) - min(offsets)) * 1000
    frame_ms = 1000.0 / max(cams[0].measured_fps, 1.0)   # length of one frame

    print(f"  offsets per flash (ms): {', '.join(f'{o * 1000:.1f}' for o in offsets)}")
    print(f"Calibrated: cam1 is {med * 1000:+.1f} ms relative to cam0 "
          f"(spread {spread:.1f} ms, frame interval ~{frame_ms:.1f} ms)")
    if spread > frame_ms:
        print("  Warning: spread is larger than one frame. Results are shaky - "
              "check lighting/ROI, or repeat the calibration.")

    # Remember the result and save it to disk.
    state["offset"] = med
    state["history_ms"].append(med * 1000)
    save_calibration(cams, state)


def save_calibration(cams, state):
    """Write the LED boxes and the offset to calibration.json."""
    data = {
        "rois": {c.name: list(c.roi) if c.roi else None for c in cams},
        "offset_cam1_minus_cam0_s": state["offset"],
        "history_ms": state["history_ms"],
    }
    with open(CAL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(cams, state):
    """If a calibration.json exists from last time, load it."""
    if not os.path.exists(CAL_FILE):
        return
    try:
        with open(CAL_FILE) as f:
            data = json.load(f)
        for c in cams:
            r = data.get("rois", {}).get(c.name)
            if r:
                c.roi = tuple(r)
        state["offset"] = float(data.get("offset_cam1_minus_cam0_s", 0.0))
        state["history_ms"] = list(data.get("history_ms", []))
        print(f"Loaded {CAL_FILE}: offset {state['offset'] * 1000:+.1f} ms "
              "(drifts over time - recalibrate before important recordings)")
    except Exception as e:
        print(f"Could not read {CAL_FILE}: {e}")


def timing_report(cams):
    """
    Check how steady each camera's frame timing is.

    We look at the gaps between consecutive pictures. A healthy 60 fps camera
    should show gaps of about 16.7 ms, all very similar. Big gaps mean frames
    were skipped; tiny gaps followed by big ones mean frames arrived in bursts.
    """
    print("Timing check (gaps between consecutive frame timestamps):")
    for c in cams:
        ts = np.array([t for t, _ in c.snapshot(BUFFER_SECONDS)])
        if len(ts) < 10:
            print(f"  {c.name}: not enough frames yet")
            continue
        gaps = np.diff(ts) * 1000.0        # time between pictures, in milliseconds
        nominal = float(np.median(gaps))   # the typical gap
        late = int(np.sum(gaps > 1.5 * nominal))    # gaps much longer than normal
        early = int(np.sum(gaps < 0.5 * nominal))   # gaps much shorter than normal
        print(f"  {c.name}: {len(ts)} frames | median gap {nominal:.1f} ms | "
              f"std {gaps.std():.1f} ms | min {gaps.min():.1f} | max {gaps.max():.1f} | "
              f"late gaps (>1.5x) {late} | bunched gaps (<0.5x) {early}")


def select_rois(cams):
    """
    Let the user draw a box around the LED in each camera's picture.
    A small window opens for each camera: drag a rectangle with the mouse,
    then press ENTER to confirm (or press 'c' to cancel).
    """
    for c in cams:
        if c.latest is None:
            continue
        win = f"{c.name}: drag a box around the LED, ENTER to confirm"
        r = cv2.selectROI(win, c.latest.copy(), showCrosshair=False)
        cv2.destroyWindow(win)
        if r[2] > 0 and r[3] > 0:    # width and height both above zero = a real box
            c.roi = tuple(int(v) for v in r)
            print(f"{c.name}: LED region set to {c.roi}")
        else:
            print(f"{c.name}: no region selected")


# ===========================================================================
# PAIRING AND SAVING
# ===========================================================================
def pair_frames(a_items, b_items):
    """
    Match each picture from camera A with the picture from camera B that was
    taken closest in time.

    Both inputs are lists of (time, picture), sorted oldest first.
    Returns a list of (timeA, pictureA, timeB, pictureB).
    """
    if not a_items or not b_items:
        return []

    b_times = [t for t, _ in b_items]   # just the times from B, for searching
    pairs = []

    for ta, fa in a_items:
        # bisect finds where `ta` would slot into B's sorted list of times.
        # The closest match is either just before or just after that spot.
        i = bisect.bisect_left(b_times, ta)
        candidates = []
        if i > 0:
            candidates.append(i - 1)
        if i < len(b_times):
            candidates.append(i)

        # Pick whichever candidate is nearer in time.
        j = min(candidates, key=lambda k: abs(b_times[k] - ta))
        tb, fb = b_items[j]
        pairs.append((ta, fa, tb, fb))

    return pairs


def save_clip(cam_a, cam_b, offset):
    """
    Save the last SAVE_SECONDS of pictures from both cameras as matched pairs.
    `offset` is the calibration result: how far camera 1's clock is ahead of
    camera 0's. We subtract it so both cameras share one timeline.
    """
    a_items = cam_a.snapshot(SAVE_SECONDS)

    # Grab a little extra from camera B (0.3 s) so the edges of the clip
    # still find a partner after we shift B's times by the offset.
    b_raw = cam_b.snapshot(SAVE_SECONDS + 0.3)
    b_items = [(t - offset, f) for t, f in b_raw]   # shift B onto A's timeline

    pairs = pair_frames(a_items, b_items)
    if not pairs:
        print("Nothing to save (buffers empty).")
        return

    # Make a new folder named after the current date and time.
    folder = os.path.join(OUTPUT_DIR, time.strftime("pitch_%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    print(f"Saving {len(pairs)} frame pairs to {folder} "
          f"(cam1 shifted by {-offset * 1000:+.1f} ms) ...")

    t0 = pairs[0][0]   # the first picture's time; we show times relative to it

    # pairs.csv is a spreadsheet you can open in Excel. One row per pair.
    with open(os.path.join(folder, "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "cam0_time_s", "cam1_time_s_aligned",
                    "dt_ms_(cam1-cam0)", "cam1_time_s_raw"])
        for n, (ta, fa, tb, fb) in enumerate(pairs):
            # Save each picture as a JPEG: cam0_0000.jpg, cam1_0000.jpg, ...
            cv2.imwrite(os.path.join(folder, f"cam0_{n:04d}.jpg"), fa,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(folder, f"cam1_{n:04d}.jpg"), fb,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([n, f"{ta - t0:.6f}", f"{tb - t0:.6f}",
                        f"{(tb - ta) * 1000:.3f}", f"{tb + offset - t0:.6f}"])

    # Summary: how closely in time are the matched pictures?
    dts = [abs(tb - ta) * 1000 for ta, _, tb, _ in pairs]
    print(f"Done. Aligned pair time difference: mean {np.mean(dts):.2f} ms, "
          f"max {np.max(dts):.2f} ms")


# ===========================================================================
# MAIN PROGRAM
# ===========================================================================
def main():
    # Create and configure one worker per camera (but don't start them yet).
    cams = [CamThread("cam0", CAMERA_INDEXES[0]),
            CamThread("cam1", CAMERA_INDEXES[1])]

    # "state" holds values shared between the keys:
    #   offset    - the calibration result in seconds (0 until you calibrate)
    #   history_ms- every offset we've measured, for reference
    #   cal_end   - when the calibration countdown finishes (None = not running)
    state = {"offset": 0.0, "history_ms": [], "cal_end": None}

    # Print a rough estimate of memory use, so you aren't surprised.
    # Each pixel takes 3 bytes (red, green, blue) once decoded.
    mb_per_frame = WIDTH * HEIGHT * 3 / 1e6
    est_mb = mb_per_frame * BUFFER_SECONDS * BUFFER_FPS * len(cams)
    print(f"Estimated buffer RAM: ~{est_mb:.0f} MB "
          f"({BUFFER_SECONDS}s @ {BUFFER_FPS} fps, {WIDTH}x{HEIGHT}, 2 cameras)")

    # Show what each camera actually agreed to. Compare with the CONFIG above.
    for c in cams:
        print(f"{c.name}: got {c.actual_w}x{c.actual_h}, driver reports "
              f"{c.reported_fps:.0f} fps | auto_exposure={c.auto_exp_readback} "
              f"exposure={c.exp_readback}")

    load_calibration(cams, state)   # reuse the LED boxes/offset from last time, if any

    for c in cams:
        c.start()                   # start both camera workers running

    print("Keys: r = pick LED region, c = calibrate, s = save clip, t = timing check, q = quit")

    try:
        # ---- The main loop: draws the preview and reacts to key presses ----
        # It repeats many times per second until you press 'q'.
        while True:

            # --- Build the preview picture: one tile per camera ---
            tiles = []
            for c in cams:
                frame = c.latest
                if frame is None:
                    # Camera hasn't delivered anything yet: show a black tile.
                    tile = np.zeros((int(HEIGHT * PREVIEW_SCALE),
                                     int(WIDTH * PREVIEW_SCALE), 3), np.uint8)
                else:
                    tile = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)

                # Draw a red rectangle where the LED box is.
                if c.roi:
                    x, y, w, h = c.roi
                    cv2.rectangle(tile,
                                  (int(x * PREVIEW_SCALE), int(y * PREVIEW_SCALE)),
                                  (int((x + w) * PREVIEW_SCALE), int((y + h) * PREVIEW_SCALE)),
                                  (0, 0, 255), 2)   # colors here are (blue, green, red)

                # Write the camera name, real fps, and how full its buffer is.
                label = (f"{c.name}  {c.measured_fps:.0f} fps  "
                         f"buf {len(c.buf)}/{c.buf.maxlen}")
                cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                tiles.append(tile)

            # Put the two tiles side by side (cut to the same height first).
            h = min(t.shape[0] for t in tiles)
            img = np.hstack([t[:h] for t in tiles])

            # --- Status line along the bottom of the preview ---
            if state["cal_end"] is not None:
                left = max(0.0, state["cal_end"] - time.perf_counter())
                status = f"CALIBRATING - FLASH THE LED NOW ({left:.1f}s left)"
                color = (0, 0, 255)       # red
            else:
                have_roi = all(c.roi for c in cams)
                status = (f"offset {state['offset'] * 1000:+.1f} ms | "
                          f"LED regions: {'set' if have_roi else 'NOT set (press r)'}")
                color = (255, 255, 0)     # cyan
            cv2.putText(img, status, (10, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Dual camera preview (r/c/s/q)", img)

            # --- Has the calibration countdown just finished? ---
            if state["cal_end"] is not None and time.perf_counter() >= state["cal_end"]:
                state["cal_end"] = None
                print("Analyzing flashes...")
                finish_calibration(cams, state)

            # --- Check for a key press (waits up to 15 ms) ---
            # 255 means "no key was pressed".
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), ord("Q"), 27):          # q, Q, or the Esc key
                break

            elif key in (ord("r"), ord("R")):            # choose the LED boxes
                select_rois(cams)
                save_calibration(cams, state)

            elif key in (ord("c"), ord("C")):            # start calibration
                if not all(c.roi for c in cams):
                    print("Set the LED region first (press r).")
                elif state["cal_end"] is None:           # ignore if already running
                    print(f"Calibrating for {CAL_SECONDS:.0f}s - flash the LED "
                          "3-5 times, clean on/off, visible to both cameras.")
                    for c in cams:
                        c.cal_samples = []               # start recording brightness
                    state["cal_end"] = time.perf_counter() + CAL_SECONDS

            elif key in (ord("t"), ord("T")):            # timing check
                timing_report(cams)

            elif key in (ord("s"), ord("S"), 32):        # s, S, or the Spacebar
                save_clip(cams[0], cams[1], state["offset"])
                frame_sub.detect_motion()


    finally:
        # Whatever happens (quit, error, Ctrl+C), always shut things down
        # cleanly so the cameras are released for other programs.
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()


# This line means: "only run main() if this file is run directly"
# (not if another program imports it).
if __name__ == "__main__":
    main()