"""
Body Movement Detection

Calibration controls:
R = Start/stop CSV recording
Tab = Switch which tracked person is "active" (the one 0-9/S label)
0 = NO TRIAL
1 = STILL
2 = RIGHT_ARM
3 = LEFT_ARM
4 = HEAD
5 = RIGHT_LEG
6 = LEFT_LEG
7 = WHOLE_BODY
8 = SITTING_STILL
9 = LYING_STILL
S = LYING_SIDEWAYS_STILL
Q or Esc = Exit

Tracks up to MAX_PEOPLE people at once. Each tracked person gets a small
on-screen ID number ("#3") drawn above their head. Track IDs only hold
frame-to-frame, not for a whole session: if someone leaves the frame, is
fully occluded, or crosses paths closely with another tracked person, they
may come back under a new ID and need their trial re-selected. See
PersonTrackManager below for why.
"""

from collections import deque
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

# -----------------------------------------------------------------------------
# Calibration and detector settings
# -----------------------------------------------------------------------------

CAMERA_INDEX = 0
CALIBRATION_MODE = True

SMOOTHING_WINDOW = 8

# Per-region movement thresholds.
#
# STATIONARY_THRESHOLD must sit above a still person's natural landmark-noise
# floor, not just below MOVING_THRESHOLD. If it doesn't, hysteresis becomes a
# one-way trap: normal sensor noise eventually pushes the state into MOVING,
# and falling back requires MIN_STATIONARY_FRAMES straight frames *below*
# STATIONARY_THRESHOLD -- which almost never happens if that threshold is
# below what a still person's noise normally produces.
#
# A single shared threshold pair can't fit every region at once, though: eye
# and leg regions run 2-6x noisier than head/torso, since fewer landmarks per
# region means each one's jitter dominates the average more, and eye
# landmarks specifically get less precise localization from the pose model.
#
# Both sides of these are now calibrated from real recordings rather than
# modeled. STATIONARY_THRESHOLD is set just above each region's actual p99
# motion reading during data/movement_data_20260818_150031.csv (844 frames
# of genuinely holding still), data/movement_data_20260818_150937.csv's
# STILL/SITTING_STILL segments, and data/movement_data_20260818_152045.csv's
# STILL segment -- verified by replaying those recordings' real per-frame
# motion values back through this exact hysteresis logic: every region
# reaches 100% correct STATIONARY on all three, for both a standing and a
# seated person. MOVING_THRESHOLD is set from labeled RIGHT_ARM/LEFT_ARM/
# WHOLE_BODY/HEAD/RIGHT_LEG/LEFT_LEG movement segments across those same
# recordings -- verified frame by frame, e.g. a real arm-raise motion
# correctly starts reading MOVING right as the arm lifts and returns to
# STATIONARY the moment it settles, with no false triggers during the calm
# segments before or after. HEAD and the leg regions were recalibrated a
# second time after the first pass proved too conservative for real head
# turns and real leg movement (HEAD only registered MOVING 23% of a
# dedicated head-movement trial; the fix lowered its thresholds) while the
# legs' *stationary* noise ceiling turned out higher than the first
# recording suggested (the fix raised theirs instead) -- cross-checked
# against the other recordings afterward so this isn't overfit to one
# session. Eyes and torso/arms have not had a comparable second look.
#
# Different people/cameras/lighting will land somewhere near, not
# necessarily exactly on, these numbers.
REGION_MOVING_THRESHOLDS = {
    "HEAD": 0.0274,
    "LEFT EYE": 0.0745,
    "RIGHT EYE": 0.0777,
    "LEFT ARM": 0.0504,
    "RIGHT ARM": 0.0451,
    "TORSO": 0.0336,
    "LEFT LEG": 0.0564,
    "RIGHT LEG": 0.0482,
}

REGION_STATIONARY_THRESHOLDS = {
    "HEAD": 0.0261,
    "LEFT EYE": 0.071,
    "RIGHT EYE": 0.074,
    "LEFT ARM": 0.048,
    "RIGHT ARM": 0.043,
    "TORSO": 0.032,
    "LEFT LEG": 0.0537,
    "RIGHT LEG": 0.0459,
}

MIN_MOVING_FRAMES = 3
MIN_STATIONARY_FRAMES = 8
MIN_VISIBILITY = 0.50

# Do not divide by a very small body reference distance.
MIN_BODY_SCALE = 0.05

# Maximum number of people MediaPipe will detect and track at once.
MAX_PEOPLE = 2

# How far (as a multiple of that person's own body scale) a detected pose's
# centroid may have moved since the previous frame and still count as the
# same tracked person. Like the motion thresholds above, this is a starting
# point, not a calibrated value -- tune it after testing with real crossing
# and occlusion footage.
TRACK_MATCH_DISTANCE_MULTIPLIER = 3.0

# Consecutive frames a tracked person can go undetected (stepped out of
# frame, fully blocked by someone else) before their track is dropped.
# At roughly 30 fps this is about a third of a second.
TRACK_MISSING_GRACE_FRAMES = 10

# EMA weight applied to each landmark's raw (x, y) position before it is
# used for drawing or motion measurement. 1.0 = no smoothing at all; lower
# values damp more per-frame jitter but add more lag behind real movement.
# 0.5 is a reasonable starting point to feel out, not a calibrated value.
LANDMARK_SMOOTHING_ALPHA = 0.5


# -----------------------------------------------------------------------------
# EMG + joystick settings
# -----------------------------------------------------------------------------
#
# None = auto-detect by scanning available serial ports (same behavior as
# the original standalone script); set to e.g. "COM4" to force one.
EMG_SERIAL_PORT = None
EMG_BAUD_RATE = 115200

# The device sends 10 EMG samples per received line/row (see parse_emg_line).
EMG_SAMPLES_PER_ROW = 10

# Width, in raw samples, of the rolling waveform window shown for EMG Node
# 1/2 and the joystick -- this is a sample-count window, not a time window,
# because at ~1 kHz per the original code's own comments, EMG needs enough
# temporal resolution to see actual waveform shape; a time-based whole
# session axis (like the movement plot's) would compress it into a blur.
EMG_WINDOW_SAMPLES = 200

# How many raw samples get folded into each RMS envelope point. RMS
# (root-mean-square, after subtracting the batch's own mean so a fixed DC
# bias in the raw reading doesn't dominate it) turns the fast raw waveform
# into a slower "activation intensity" trend -- unlike the raw window
# above, the envelope is what actually belongs on the same whole-session
# time axis as the movement plot, since that's the comparison the "EMG vs
# camera movement" goal is actually about.
EMG_RMS_WINDOW_SAMPLES = 50


# -----------------------------------------------------------------------------
# MediaPipe Pose Landmarker indices
# -----------------------------------------------------------------------------

NOSE = 0

LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3

RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6

LEFT_EAR = 7
RIGHT_EAR = 8

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Knee/ankle landmarks get extra smoothing on top of LANDMARK_SMOOTHING_ALPHA.
# Real recordings show their motion noise floor running roughly 2-3x higher
# than every other region even while genuinely still (e.g. mean ~0.021-0.023
# vs ~0.008-0.011 for head/torso/arms) -- hip landmarks are not part of this
# (TORSO, which shares them, is clean), so the extra smoothing is scoped to
# just the lower-leg joints instead of applied globally, which would add
# unnecessary lag to the regions that are already working well.
LEG_EXTREMITY_SMOOTHING_ALPHA = 0.3
LEG_EXTREMITY_LANDMARKS = {LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}


# -----------------------------------------------------------------------------
# Region definitions
# -----------------------------------------------------------------------------

REGION_LANDMARKS = {
    "HEAD": [
        NOSE,
        LEFT_EYE_INNER,
        LEFT_EYE,
        LEFT_EYE_OUTER,
        LEFT_EAR,
        RIGHT_EYE_INNER,
        RIGHT_EYE,
        RIGHT_EYE_OUTER,
        RIGHT_EAR,
    ],
    "LEFT EYE": [
        LEFT_EYE_INNER,
        LEFT_EYE,
        LEFT_EYE_OUTER,
    ],
    "RIGHT EYE": [
        RIGHT_EYE_INNER,
        RIGHT_EYE,
        RIGHT_EYE_OUTER,
    ],
    "LEFT ARM": [
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
    ],
    "RIGHT ARM": [
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
    ],
    "TORSO": [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    ],
    "LEFT LEG": [
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
    ],
    "RIGHT LEG": [
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
    ],
}

REGION_CONNECTIONS = {
    "HEAD": [
        (NOSE, LEFT_EYE_INNER),
        (NOSE, RIGHT_EYE_INNER),
        (NOSE, LEFT_EAR),
        (NOSE, RIGHT_EAR),
        (LEFT_EYE_OUTER, LEFT_EAR),
        (RIGHT_EYE_OUTER, RIGHT_EAR),
        (LEFT_EAR, RIGHT_EAR),
    ],
    "LEFT EYE": [
        (LEFT_EYE_INNER, LEFT_EYE),
        (LEFT_EYE, LEFT_EYE_OUTER),
    ],
    "RIGHT EYE": [
        (RIGHT_EYE_INNER, RIGHT_EYE),
        (RIGHT_EYE, RIGHT_EYE_OUTER),
    ],
    "LEFT ARM": [
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
    ],
    "RIGHT ARM": [
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
    ],
    "TORSO": [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
    ],
    "LEFT LEG": [
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
    ],
    "RIGHT LEG": [
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ],
}


# OpenCV colors use BGR order.
MOVEMENT_COLORS = {
    "MOVING": (0, 0, 255),       # Red
    "STATIONARY": (0, 255, 0),   # Green
    "UNCERTAIN": (0, 165, 255),  # Orange
}

PARTIAL_COLOR = (255, 0, 255)  # Purple


# Calibration labels only. They never affect movement detection.
TRIAL_LABELS = {
    0: "NO TRIAL",
    1: "STILL",
    2: "RIGHT_ARM",
    3: "LEFT_ARM",
    4: "HEAD",
    5: "RIGHT_LEG",
    6: "LEFT_LEG",
    7: "WHOLE_BODY",
    8: "SITTING_STILL",
    9: "LYING_STILL",
    10: "LYING_SIDEWAYS_STILL",
}

TRIAL_KEY_BINDINGS = {
    ord("0"): 0,
    ord("1"): 1,
    ord("2"): 2,
    ord("3"): 3,
    ord("4"): 4,
    ord("5"): 5,
    ord("6"): 6,
    ord("7"): 7,
    ord("8"): 8,
    ord("9"): 9,
    ord("s"): 10,
    ord("S"): 10,
}


# -----------------------------------------------------------------------------
# Geometry and observability helpers
# -----------------------------------------------------------------------------

def is_landmark_reliable(landmark) -> bool:
    """Return True only when a landmark is visible and inside the image."""
    return (
        landmark.visibility >= MIN_VISIBILITY
        and 0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
    )


def get_landmark_position(landmarks, landmark_index):
    """Return (x, y) for a reliable landmark, otherwise None."""
    landmark = landmarks[landmark_index]

    if not is_landmark_reliable(landmark):
        return None

    return landmark.x, landmark.y


def distance_2d(first_position, second_position) -> float:
    """Return Euclidean distance between two normalized image positions."""
    first_x, first_y = first_position
    second_x, second_y = second_position

    return (
        (second_x - first_x) ** 2
        + (second_y - first_y) ** 2
    ) ** 0.5


def midpoint(first_position, second_position):
    """Return the midpoint between two normalized image positions."""
    first_x, first_y = first_position
    second_x, second_y = second_position

    return (
        (first_x + second_x) / 2,
        (first_y + second_y) / 2,
    )


def calculate_body_scale(landmarks):
    """Return a stable body-size reference for motion normalization.

    Shoulder width is preferred. Hip width and shoulder-to-hip torso lengths
    are fallbacks for partial body visibility.
    """
    left_shoulder = get_landmark_position(landmarks, LEFT_SHOULDER)
    right_shoulder = get_landmark_position(landmarks, RIGHT_SHOULDER)
    left_hip = get_landmark_position(landmarks, LEFT_HIP)
    right_hip = get_landmark_position(landmarks, RIGHT_HIP)

    candidate_scales = []

    if left_shoulder is not None and right_shoulder is not None:
        candidate_scales.append(distance_2d(left_shoulder, right_shoulder))

    if left_hip is not None and right_hip is not None:
        candidate_scales.append(distance_2d(left_hip, right_hip))

    if left_shoulder is not None and left_hip is not None:
        candidate_scales.append(distance_2d(left_shoulder, left_hip))

    if right_shoulder is not None and right_hip is not None:
        candidate_scales.append(distance_2d(right_shoulder, right_hip))

    for scale in candidate_scales:
        if scale >= MIN_BODY_SCALE:
            return scale

    return None


def aspect_correct_landmarks(landmarks, frame_width: int, frame_height: int):
    """Rescale whichever axis is wider so x and y end up on the same
    physical scale.

    MediaPipe normalizes a landmark's x to [0, 1] as a fraction of frame
    WIDTH and y as a fraction of frame HEIGHT -- two different physical
    scales whenever width != height, which is true of essentially every
    real camera. distance_2d() and everything built on it (raw_motion,
    calculate_body_scale) then mixes those two scales as if they were one,
    so the same physical arm movement measures differently depending on
    its direction: for a 1280x720 (landscape) frame, a purely vertical
    movement reads about 1.78x higher than the identical horizontal
    movement, because the 720-pixel height compresses less than the
    1280-pixel width does. That is why waving an arm side to side barely
    registered while raising it the same physical amount worked fine --
    horizontal was reading correctly the whole time; vertical was
    inflated.

    This shrinks the longer axis to match the shorter one (dividing y by
    width/height for a landscape frame, or x by height/width for a
    portrait one) rather than growing the shorter axis to match the
    longer one. Both are mathematically equivalent for the final
    normalized_motion ratio, but growing an axis can push it past 1.0 --
    e.g. multiplying x by ~1.78 sends anyone in roughly the right half of
    a landscape frame outside [0, 1] -- which is exactly the range
    is_landmark_reliable() checks. Shrinking only ever moves a coordinate
    closer to 0, so it can't trip that check the way growing did when
    this was first written.

    Must only be applied to the copy fed into motion tracking
    (RegionTracker, calculate_body_scale, calculate_person_centroid) --
    landmarks used for on-screen drawing still need genuine
    width-normalized x and height-normalized y to land on the correct
    pixel.
    """
    if frame_width >= frame_height:
        aspect_ratio = frame_width / frame_height
        return [
            SmoothedLandmark(
                landmark.x,
                landmark.y / aspect_ratio,
                landmark.visibility,
            )
            for landmark in landmarks
        ]

    aspect_ratio = frame_height / frame_width

    return [
        SmoothedLandmark(
            landmark.x / aspect_ratio,
            landmark.y,
            landmark.visibility,
        )
        for landmark in landmarks
    ]


def calculate_person_centroid(landmarks):
    """Return an approximate (x, y) center point for one detected pose.

    This is only used to match detections to tracks between frames -- it is
    a rough "where is this body roughly located" signal, not a precise
    measurement, so it simply averages every currently reliable landmark.
    """
    positions = [
        (landmark.x, landmark.y)
        for landmark in landmarks
        if is_landmark_reliable(landmark)
    ]

    if not positions:
        return None

    x_total = sum(position[0] for position in positions)
    y_total = sum(position[1] for position in positions)

    return x_total / len(positions), y_total / len(positions)


@dataclass
class SmoothedLandmark:
    """A landmark position after EMA smoothing -- same shape as a MediaPipe
    landmark (x, y, visibility) so it can be used as a drop-in replacement
    anywhere a raw landmark is expected."""

    x: float
    y: float
    visibility: float


def smooth_landmarks(raw_landmarks, previous_smoothed):
    """Exponentially smooth one person's landmark positions across frames.

    MediaPipe's raw per-frame output has natural landmark noise. Left alone,
    that noise shows up two ways: a visibly jittery skeleton, and small
    spurious blips in the motion signal that the SMOOTHING_WINDOW average
    does not fully remove because it smooths *motion*, not *position* -- a
    single noisy frame still contributes its (wrong) displacement before it
    gets averaged out. Smoothing position first, before displacement is ever
    measured, fixes both with one mechanism instead of two.

    Landmarks that are not currently reliable are passed through unsmoothed
    -- there is no reason to trust a low-confidence position, smoothed or not.
    """
    smoothed = []

    for index, landmark in enumerate(raw_landmarks):
        if not is_landmark_reliable(landmark):
            smoothed.append(
                SmoothedLandmark(landmark.x, landmark.y, landmark.visibility)
            )
            continue

        has_previous = (
            previous_smoothed is not None
            and index < len(previous_smoothed)
        )

        if not has_previous:
            smoothed.append(
                SmoothedLandmark(landmark.x, landmark.y, landmark.visibility)
            )
            continue

        previous = previous_smoothed[index]

        alpha = (
            LEG_EXTREMITY_SMOOTHING_ALPHA
            if index in LEG_EXTREMITY_LANDMARKS
            else LANDMARK_SMOOTHING_ALPHA
        )

        new_x = alpha * landmark.x + (1 - alpha) * previous.x
        new_y = alpha * landmark.y + (1 - alpha) * previous.y

        smoothed.append(SmoothedLandmark(new_x, new_y, landmark.visibility))

    return smoothed


def smooth_body_scale(raw_scale, previous_smoothed_scale):
    """Exponentially smooth body scale across frames, same idea as
    smooth_landmarks but for the normalization denominator itself.

    calculate_body_scale() can legitimately switch which pair of landmarks
    it measures between frames -- e.g. a shoulder landmark's visibility
    flickering across the reliability cutoff, which is common when sitting
    and partly occluded by a desk or chair. Shoulder width and torso length
    are different real distances, so an unsmoothed switch changes the
    normalization denominator for every region at once on a frame where
    nothing actually moved.
    """
    if raw_scale is None:
        return previous_smoothed_scale

    if previous_smoothed_scale is None:
        return raw_scale

    return (
        LANDMARK_SMOOTHING_ALPHA * raw_scale
        + (1 - LANDMARK_SMOOTHING_ALPHA) * previous_smoothed_scale
    )


def get_observability(region_name: str, visible_indices: set[int]) -> str:
    """Return RELIABLE, PARTIAL, or UNRELIABLE for a body region."""
    all_indices = set(REGION_LANDMARKS[region_name])

    if visible_indices == all_indices:
        return "RELIABLE"

    # A PARTIAL region must still contain a meaningful connected segment.
    for first_index, second_index in REGION_CONNECTIONS[region_name]:
        if first_index in visible_indices and second_index in visible_indices:
            return "PARTIAL"

    return "UNRELIABLE"


def calculate_head_raw_motion(current_positions, previous_positions):
    """Calculate head motion from displacement and simple head geometry.

    This combines normal visible head-landmark displacement with:
    - nose movement relative to the midpoint of both ears;
    - ear-to-ear vector change.

    These relative measurements help capture turning/rotation, rather than
    relying only on whole-head movement across the image.
    """
    motion_components = []

    landmark_displacements = []

    for index, current_position in current_positions.items():
        if index in previous_positions:
            landmark_displacements.append(
                distance_2d(
                    current_position,
                    previous_positions[index],
                )
            )

    if landmark_displacements:
        motion_components.append(
            sum(landmark_displacements) / len(landmark_displacements)
        )

    required_indices = {NOSE, LEFT_EAR, RIGHT_EAR}

    if (
        required_indices.issubset(current_positions)
        and required_indices.issubset(previous_positions)
    ):
        current_ear_midpoint = midpoint(
            current_positions[LEFT_EAR],
            current_positions[RIGHT_EAR],
        )

        previous_ear_midpoint = midpoint(
            previous_positions[LEFT_EAR],
            previous_positions[RIGHT_EAR],
        )

        current_nose_relative = (
            current_positions[NOSE][0] - current_ear_midpoint[0],
            current_positions[NOSE][1] - current_ear_midpoint[1],
        )

        previous_nose_relative = (
            previous_positions[NOSE][0] - previous_ear_midpoint[0],
            previous_positions[NOSE][1] - previous_ear_midpoint[1],
        )

        motion_components.append(
            distance_2d(
                current_nose_relative,
                previous_nose_relative,
            )
        )

        current_ear_vector = (
            current_positions[RIGHT_EAR][0]
            - current_positions[LEFT_EAR][0],
            current_positions[RIGHT_EAR][1]
            - current_positions[LEFT_EAR][1],
        )

        previous_ear_vector = (
            previous_positions[RIGHT_EAR][0]
            - previous_positions[LEFT_EAR][0],
            previous_positions[RIGHT_EAR][1]
            - previous_positions[LEFT_EAR][1],
        )

        motion_components.append(
            distance_2d(
                current_ear_vector,
                previous_ear_vector,
            )
        )

    if not motion_components:
        return None

    return sum(motion_components) / len(motion_components)


# -----------------------------------------------------------------------------
# Movement tracking
# -----------------------------------------------------------------------------

@dataclass
class RegionTracker:
    """Stores measurement history, observability, and stable state."""

    name: str

    previous_positions: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

    motion_history: deque = field(
        default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW)
    )

    state: str = "UNCERTAIN"
    observability: str = "UNRELIABLE"

    is_observable: bool = False
    motion_available: bool = False

    moving_frames: int = 0
    stationary_frames: int = 0

    # Current measurement before normalization.
    raw_motion: float = 0.0

    # Current frame motion after normalization.
    normalized_motion: float = 0.0

    # Smoothed normalized motion used by movement detection.
    smoothed_motion: float = 0.0

    def reset_measurement(self) -> None:
        """Clear temporal data when a safe movement decision is not possible."""
        self.previous_positions.clear()
        self.motion_history.clear()

        self.raw_motion = 0.0
        self.normalized_motion = 0.0
        self.smoothed_motion = 0.0

        self.moving_frames = 0
        self.stationary_frames = 0

        self.state = "UNCERTAIN"
        self.is_observable = False
        self.motion_available = False

    def mark_unreliable(self) -> None:
        """Mark a region as unobservable and reset its motion baseline."""
        self.reset_measurement()
        self.observability = "UNRELIABLE"

    def update(
        self,
        landmarks,
        landmark_indices: list[int],
        body_scale,
    ) -> str:
        """Update one body region using body-scale-normalized motion."""
        current_positions = {}

        for index in landmark_indices:
            landmark = landmarks[index]

            if is_landmark_reliable(landmark):
                current_positions[index] = (landmark.x, landmark.y)

        visible_indices = set(current_positions)
        self.observability = get_observability(
            self.name,
            visible_indices,
        )

        if self.observability == "UNRELIABLE":
            self.mark_unreliable()
            return self.state

        # Do not treat unnormalized displacement as detector motion.
        if body_scale is None or body_scale < MIN_BODY_SCALE:
            self.reset_measurement()
            return self.state

        # First frame after uncertainty establishes a clean baseline.
        if not self.is_observable:
            self.previous_positions = current_positions
            self.is_observable = True
            self.motion_available = False
            self.state = "UNCERTAIN"
            return self.state

        common_indices = set(current_positions).intersection(
            self.previous_positions
        )

        # Never calculate movement from only one common point.
        if (
            get_observability(self.name, common_indices)
            == "UNRELIABLE"
        ):
            self.previous_positions = current_positions
            self.motion_history.clear()
            self.raw_motion = 0.0
            self.normalized_motion = 0.0
            self.smoothed_motion = 0.0
            self.motion_available = False
            self.state = "UNCERTAIN"
            self.moving_frames = 0
            self.stationary_frames = 0
            return self.state

        if self.name == "HEAD":
            raw_motion = calculate_head_raw_motion(
                current_positions,
                self.previous_positions,
            )
        else:
            # The maximum, not the average, of the region's landmark
            # displacements. A whole-limb swing moves every landmark in
            # the region together, so the two are close either way -- but
            # a localized movement (shaking just the hand, wiggling just
            # the foot) only displaces the most distal landmark, and
            # averaging that against two near-stationary ones divides a
            # real signal by the region's landmark count instead of
            # reporting it. Whichever landmark is actually moving should
            # drive the reading.
            displacements = []

            for index in common_indices:
                displacements.append(
                    distance_2d(
                        current_positions[index],
                        self.previous_positions[index],
                    )
                )

            raw_motion = max(displacements) if displacements else None

        if raw_motion is None:
            self.previous_positions = current_positions
            self.motion_available = False
            self.state = "UNCERTAIN"
            return self.state

        self.raw_motion = raw_motion

        # Normalize by body size so distance from the webcam has less effect.
        self.normalized_motion = raw_motion / body_scale

        self.motion_history.append(self.normalized_motion)

        self.smoothed_motion = (
            sum(self.motion_history) / len(self.motion_history)
        )

        self.previous_positions = current_positions
        self.motion_available = True

        self._update_state()

        return self.state

    def _update_state(self) -> None:
        """Apply hysteresis and persistence using this region's thresholds."""
        moving_threshold = REGION_MOVING_THRESHOLDS[self.name]
        stationary_threshold = REGION_STATIONARY_THRESHOLDS[self.name]

        if self.state in ("UNCERTAIN", "STATIONARY"):
            if self.smoothed_motion >= moving_threshold:
                self.moving_frames += 1
            else:
                self.moving_frames = 0

            if self.moving_frames >= MIN_MOVING_FRAMES:
                self.state = "MOVING"
                self.moving_frames = 0
                self.stationary_frames = 0
            else:
                self.state = "STATIONARY"

        elif self.state == "MOVING":
            if self.smoothed_motion <= stationary_threshold:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0

            if self.stationary_frames >= MIN_STATIONARY_FRAMES:
                self.state = "STATIONARY"
                self.stationary_frames = 0
                self.moving_frames = 0


# -----------------------------------------------------------------------------
# Multi-person tracking
# -----------------------------------------------------------------------------

@dataclass
class PersonTrack:
    """One tracked person: their own region trackers, trial label, and
    the bookkeeping PersonTrackManager needs to keep matching them frame
    to frame."""

    track_id: int

    region_trackers: dict[str, RegionTracker] = field(
        default_factory=lambda: {
            region_name: RegionTracker(region_name)
            for region_name in REGION_LANDMARKS
        }
    )

    trial_number: int = 0

    # Frame-to-frame matching state only -- not a stable long-term identity.
    centroid: tuple[float, float] | None = None
    last_raw_body_scale: float | None = None
    missing_frames: int = 0

    # EMA smoothing state (see smooth_landmarks) and the body scale derived
    # from it -- this is the one actually used for motion detection/display,
    # kept separate from last_raw_body_scale above which only ever feeds the
    # identity-matching distance threshold.
    smoothed_landmarks: list | None = None
    motion_body_scale: float | None = None

    # Rolling (timestamp_seconds, smoothed_motion) history per region, for
    # the live movement plot. Unbounded deque, time-pruned each frame
    # (see record_plot_samples) rather than a fixed maxlen, so the window
    # covers a consistent number of *seconds* regardless of frame rate.
    plot_history: dict[str, deque] = field(
        default_factory=lambda: {
            region_name: deque() for region_name in REGION_LANDMARKS
        }
    )


class PersonTrackManager:
    """Matches each frame's detected poses to persistent-for-a-while tracks.

    MediaPipe's multi-pose detection has no built-in identity across frames:
    the order of poses in its result can change from one frame to the next.
    Left alone, that would make a region tracker compare one person's current
    position against a different person's previous position -- a large,
    spurious "movement" every time it happens. This class closes that gap
    with a simple nearest-centroid match: a detection links to whichever
    existing track has the closest last-known centroid, as long as it is
    within a body-size-scaled distance. Anything left over becomes a new
    track.

    A track that goes unmatched for a few frames (person left frame, brief
    occlusion) is retired. If that person reappears later they get a new
    track id rather than resuming the old one -- a deliberate simplification,
    since this project does not need identity to survive a long absence,
    only frame-to-frame continuity for the motion math to stay valid.
    """

    def __init__(self) -> None:
        self.tracks: dict[int, PersonTrack] = {}
        self.next_track_id = 1
        self.active_track_id: int | None = None

    def update(self, detected_landmarks_list) -> dict[int, list]:
        """Match this frame's detected poses to tracks.

        Returns {track_id: landmarks} for people actually detected this
        frame (tracks in a missing-frame grace period are not included,
        since there is nothing fresh to draw or measure for them).
        """
        detections = []

        for landmarks in detected_landmarks_list:
            detections.append(
                {
                    "landmarks": landmarks,
                    "centroid": calculate_person_centroid(landmarks),
                    "body_scale": calculate_body_scale(landmarks),
                }
            )

        candidates = []

        for detection_index, detection in enumerate(detections):
            if detection["centroid"] is None:
                continue

            for track_id, track in self.tracks.items():
                if track.centroid is None:
                    continue

                distance = distance_2d(detection["centroid"], track.centroid)

                reference_scale = (
                    detection["body_scale"]
                    or track.last_raw_body_scale
                    or MIN_BODY_SCALE
                )
                max_distance = reference_scale * TRACK_MATCH_DISTANCE_MULTIPLIER

                if distance <= max_distance:
                    candidates.append((distance, detection_index, track_id))

        # Greedy nearest-neighbor matching: closest pairs win first, and
        # each detection/track can only be used once. Simple, and plenty
        # for a handful of people -- an optimal assignment (e.g. Hungarian
        # algorithm) is not worth the complexity at this scale.
        candidates.sort(key=lambda candidate: candidate[0])

        detection_index_to_track_id: dict[int, int] = {}
        matched_detection_indices = set()
        matched_track_ids = set()

        for distance, detection_index, track_id in candidates:
            if detection_index in matched_detection_indices:
                continue
            if track_id in matched_track_ids:
                continue

            detection_index_to_track_id[detection_index] = track_id
            matched_detection_indices.add(detection_index)
            matched_track_ids.add(track_id)

        for detection_index in range(len(detections)):
            if detection_index in matched_detection_indices:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = PersonTrack(track_id=track_id)
            detection_index_to_track_id[detection_index] = track_id

        for detection_index, track_id in detection_index_to_track_id.items():
            detection = detections[detection_index]
            track = self.tracks[track_id]
            track.centroid = detection["centroid"]
            track.last_raw_body_scale = detection["body_scale"]
            track.missing_frames = 0

        matched_track_ids_this_frame = set(detection_index_to_track_id.values())

        # A track that isn't matched this exact frame is left alone here --
        # its region trackers keep their last known state frozen rather than
        # being reset. Earlier this called mark_unreliable() on every single
        # missed frame, which meant one momentary detection dip (multi-pose
        # inference is noisier per person than single-pose) threw away
        # accumulated motion history and forced MIN_MOVING_FRAMES worth of
        # fresh evidence to rebuild a MOVING state -- the track survived the
        # grace period, but its motion state did not, which read as random
        # flip-flopping. A track only loses its state now by actually being
        # retired below, at which point its RegionTracker instances are
        # discarded along with it.
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids_this_frame:
                continue

            track.missing_frames += 1

        for track_id in list(self.tracks):
            if self.tracks[track_id].missing_frames > TRACK_MISSING_GRACE_FRAMES:
                del self.tracks[track_id]

                if self.active_track_id == track_id:
                    self.active_track_id = None

        if self.active_track_id not in self.tracks:
            self.active_track_id = min(self.tracks) if self.tracks else None

        return {
            track_id: detections[detection_index]["landmarks"]
            for detection_index, track_id in detection_index_to_track_id.items()
        }

    def cycle_active_track(self) -> None:
        """Move the "active" (labelable) track to the next tracked person."""
        track_ids = sorted(self.tracks)

        if not track_ids:
            self.active_track_id = None
            return

        if self.active_track_id not in track_ids:
            self.active_track_id = track_ids[0]
            return

        current_index = track_ids.index(self.active_track_id)
        next_index = (current_index + 1) % len(track_ids)
        self.active_track_id = track_ids[next_index]


# -----------------------------------------------------------------------------
# EMG + joystick data pipeline
# -----------------------------------------------------------------------------

def parse_emg_line(line: str):
    """Parse one line of the device's wire format into a row dict, or None
    if the line is blank or malformed (a torn/partial line during startup
    or a baud-rate mismatch, not something worth crashing over).

    Expected format: seq,timestamp_ms,emg1_0..emg1_9,emg2_0..emg2_9,
    joy_x,joy_y,btn -- 25 comma-separated fields.
    """
    cleaned = line.strip()

    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(",")]

    if len(parts) < 25:
        return None

    try:
        seq_num = int(parts[0])
        device_timestamp_ms = int(parts[1])
        emg1_values = [int(value) for value in parts[2:12]]
        emg2_values = [int(value) for value in parts[12:22]]
        joy_x = int(parts[22])
        joy_y = int(parts[23])
        btn = int(parts[24])
    except ValueError:
        return None

    return {
        "seq": seq_num,
        "device_timestamp_ms": device_timestamp_ms,
        "emg1": emg1_values,
        "emg2": emg2_values,
        "joy_x": joy_x,
        "joy_y": joy_y,
        "btn": btn,
    }


def compute_rms(values) -> float:
    """RMS of the AC-varying component of a batch of raw samples (the
    batch's own mean is subtracted first). Plain RMS of the raw values
    would mostly just reflect whatever fixed DC bias the sensor's reading
    happens to sit at (this demo data centers on 300, a real device might
    center on something else entirely) rather than the actual muscle
    activity, which is the variation around that baseline.
    """
    if not values:
        return 0.0

    mean = sum(values) / len(values)
    mean_square = sum((value - mean) ** 2 for value in values) / len(values)

    return mean_square ** 0.5


class EmgReceiver:
    """Reads real EMG + joystick rows on a background thread so the ~30 fps
    video loop never blocks waiting on a slow or serial-timeout read.

    Each row is stamped with pc_timestamp_seconds at the moment it is
    actually received, measured from the same start_time origin as the
    video loop's own timestamp_ms -- not at the moment the main loop
    later happens to drain it. EMG rows can (and at ~20 rows/sec vs ~30
    fps, will) arrive faster than the video loop drains them; stamping at
    drain time would silently pile several rows onto one video frame's
    timestamp and lose their real spacing.
    """

    def __init__(
        self,
        port,
        baud_rate: int,
        start_time: float,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.start_time = start_time

        self.row_queue: queue.Queue = queue.Queue()
        # `connected` means valid EMG data is arriving, not merely that a
        # serial port happened to open successfully.
        self.connected = False
        self.port_open = False
        self.connected_port: str | None = None
        self.connection_error: str | None = None
        self.received_row_count = 0
        self.last_data_seconds: float | None = None
        self.last_data_monotonic: float | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def drain(self) -> list[dict]:
        """Return every row received since the last call, without blocking."""
        rows = []

        while True:
            try:
                rows.append(self.row_queue.get_nowait())
            except queue.Empty:
                break

        return rows

    def _run(self) -> None:
        self._run_serial()

    def _run_serial(self) -> None:
        try:
            import serial
            import serial.tools.list_ports
        except ImportError:
            self.connection_error = (
                "pyserial is not installed. Run: pip install pyserial"
            )
            return

        while not self._stop_event.is_set():
            available_ports = [
                device.device for device in serial.tools.list_ports.comports()
            ]
            # Auto-detect still supports every listed port, but tries the
            # common ESP32 assignments COM3/COM4 first when they exist.
            available_ports.sort(
                key=lambda name: (name.upper() not in {"COM3", "COM4"}, name)
            )
            candidates = [self.port] if self.port else available_ports

            if not candidates:
                self.connection_error = "COM NOT CONNECTED: no serial ports found."
                self._stop_event.wait(2.0)
                continue

            connection = None
            last_error = None

            for port_name in candidates:
                try:
                    connection = serial.Serial(
                        port_name, self.baud_rate, timeout=0.1
                    )
                    self.connected_port = port_name
                    self.port_open = True
                    self.connection_error = None
                    break
                except serial.SerialException as exc:
                    last_error = exc

            if connection is None:
                requested_ports = ", ".join(candidates)
                self.connection_error = (
                    f"COM NOT CONNECTED: could not open {requested_ports} "
                    f"({last_error})."
                )
                self._stop_event.wait(2.0)
                continue

            try:
                while not self._stop_event.is_set():
                    line = connection.readline()

                    if not line:
                        continue

                    elapsed_seconds = time.perf_counter() - self.start_time
                    row = parse_emg_line(line.decode("utf-8", errors="ignore"))

                    # Only correctly formed device rows reach the state or
                    # plots. Random serial text and malformed packets cannot
                    # create fake EMG traces.
                    if row is not None:
                        row["pc_timestamp_seconds"] = elapsed_seconds
                        self.row_queue.put(row)
                        self.received_row_count += 1
                        self.last_data_seconds = elapsed_seconds
                        self.last_data_monotonic = time.perf_counter()
                        self.connected = True
            except serial.SerialException as exc:
                self.connection_error = (
                    f"COM NOT CONNECTED: {self.connected_port} disconnected "
                    f"({exc})."
                )
            finally:
                connection.close()
                self.port_open = False
                self.connected = False

            # Keep trying so reconnecting an ESP32 does not require restarting.
            if not self._stop_event.is_set():
                self._stop_event.wait(1.0)


@dataclass
class EmgState:
    """Everything the display needs from the EMG/joystick stream: a short
    rolling window of the raw waveform (signal-quality viewing, matches
    the original standalone script's windowing) plus a whole-session RMS
    envelope trend for each EMG channel (the thing actually comparable to
    the movement plot's timescale).
    """

    sample_count: int = 0

    # Joystick/button update once per received row, not once per EMG
    # sample within it, so their rolling window is indexed by row_count
    # instead of sample_count.
    row_count: int = 0

    emg1_raw: deque = field(
        default_factory=lambda: deque(maxlen=EMG_WINDOW_SAMPLES)
    )
    emg2_raw: deque = field(
        default_factory=lambda: deque(maxlen=EMG_WINDOW_SAMPLES)
    )
    joy_x_raw: deque = field(
        default_factory=lambda: deque(maxlen=EMG_WINDOW_SAMPLES)
    )
    joy_y_raw: deque = field(
        default_factory=lambda: deque(maxlen=EMG_WINDOW_SAMPLES)
    )
    btn_raw: deque = field(
        default_factory=lambda: deque(maxlen=EMG_WINDOW_SAMPLES)
    )

    # (timestamp_seconds, rms_value) pairs, unbounded -- covers the whole
    # session, like PersonTrack.plot_history.
    emg1_envelope: deque = field(default_factory=deque)
    emg2_envelope: deque = field(default_factory=deque)

    # Accumulate raw samples here until there are enough for one RMS point.
    _emg1_rms_buffer: list = field(default_factory=list)
    _emg2_rms_buffer: list = field(default_factory=list)

    connected: bool = False
    last_row_seconds: float | None = None


def update_emg_state(state: EmgState, row: dict) -> None:
    """Fold one received row into both the rolling raw window and the
    session-long RMS envelope, using the row's own pc_timestamp_seconds
    (stamped by EmgReceiver at actual receipt time) rather than whatever
    video frame happens to be draining it."""
    timestamp_seconds = row["pc_timestamp_seconds"]

    state.emg1_raw.extend(row["emg1"])
    state.emg2_raw.extend(row["emg2"])
    state.joy_x_raw.append(row["joy_x"])
    state.joy_y_raw.append(row["joy_y"])
    state.btn_raw.append(row["btn"])
    state.sample_count += len(row["emg1"])
    state.row_count += 1
    state.last_row_seconds = timestamp_seconds

    state._emg1_rms_buffer.extend(row["emg1"])
    state._emg2_rms_buffer.extend(row["emg2"])

    if len(state._emg1_rms_buffer) >= EMG_RMS_WINDOW_SAMPLES:
        state.emg1_envelope.append(
            (timestamp_seconds, compute_rms(state._emg1_rms_buffer))
        )
        state._emg1_rms_buffer.clear()

    if len(state._emg2_rms_buffer) >= EMG_RMS_WINDOW_SAMPLES:
        state.emg2_envelope.append(
            (timestamp_seconds, compute_rms(state._emg2_rms_buffer))
        )
        state._emg2_rms_buffer.clear()


# -----------------------------------------------------------------------------
# CSV logging
# -----------------------------------------------------------------------------

class MovementCsvLogger:
    """Writes one row per tracked person per recorded frame."""

    def __init__(self, project_directory: Path, session_timestamp: str) -> None:
        data_directory = project_directory / "data"
        data_directory.mkdir(exist_ok=True)

        timestamp = session_timestamp

        self.path = data_directory / f"movement_data_{timestamp}.csv"

        suffix = 1
        while self.path.exists():
            self.path = data_directory / (
                f"movement_data_{timestamp}_{suffix}.csv"
            )
            suffix += 1

        self.file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self._fieldnames(),
        )

        self.writer.writeheader()
        self.file.flush()

    @staticmethod
    def _fieldnames() -> list[str]:
        fieldnames = [
            "timestamp_ms",
            "frame_number",
            "track_id",
            "trial",
            "body_scale",
            "HEAD_raw_motion",
            "HEAD_normalized_motion",
        ]

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_motion")

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_observability")

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_state")

        fieldnames.append("OVERALL_state")

        return fieldnames

    def write_frame(
        self,
        timestamp_ms: int,
        frame_number: int,
        track_id: int,
        trial: str,
        body_scale,
        region_trackers: dict[str, RegionTracker],
        overall_state: str,
    ) -> None:
        """Write one detector row for one tracked person. Missing motion remains blank."""
        head_tracker = region_trackers["HEAD"]

        row = {
            "timestamp_ms": timestamp_ms,
            "frame_number": frame_number,
            "track_id": track_id,
            "trial": trial,
            "body_scale": (
                f"{body_scale:.8f}"
                if body_scale is not None
                else ""
            ),
            "HEAD_raw_motion": (
                f"{head_tracker.raw_motion:.8f}"
                if head_tracker.motion_available
                else ""
            ),
            "HEAD_normalized_motion": (
                f"{head_tracker.normalized_motion:.8f}"
                if head_tracker.motion_available
                else ""
            ),
            "OVERALL_state": overall_state,
        }

        for region_name, tracker in region_trackers.items():
            csv_name = region_name.replace(" ", "_")

            row[f"{csv_name}_motion"] = (
                f"{tracker.smoothed_motion:.8f}"
                if tracker.motion_available
                else ""
            )

            row[f"{csv_name}_observability"] = tracker.observability
            row[f"{csv_name}_state"] = tracker.state

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file when the program exits."""
        self.file.close()


class EmgCsvLogger:
    """Writes one row per received EMG/joystick row.

    A separate file from MovementCsvLogger, not a shared one: EMG rows and
    video frames arrive on fundamentally different, unrelated sampling
    grids (roughly 20 rows/sec of 10 packed samples each here, ~30 fps
    there), so forcing them into one row-per-something would mean either
    throwing away EMG resolution or padding out nonsense movement rows.
    Two files sharing the same session_timestamp and the same timestamp_ms
    clock (see main()) can still be joined by timestamp in analysis --
    e.g. pandas.merge_asof -- without either one being distorted to fit
    the other's rate.
    """

    def __init__(self, project_directory: Path, session_timestamp: str) -> None:
        data_directory = project_directory / "data"
        data_directory.mkdir(exist_ok=True)

        self.path = data_directory / f"emg_data_{session_timestamp}.csv"

        suffix = 1
        while self.path.exists():
            self.path = data_directory / (
                f"emg_data_{session_timestamp}_{suffix}.csv"
            )
            suffix += 1

        self.file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.file)

        header = (
            ["timestamp_ms", "device_timestamp_ms", "seq"]
            + [f"emg1_{i}" for i in range(EMG_SAMPLES_PER_ROW)]
            + [f"emg2_{i}" for i in range(EMG_SAMPLES_PER_ROW)]
            + ["joy_x", "joy_y", "btn"]
        )
        self.writer.writerow(header)
        self.file.flush()

    def write_row(self, timestamp_ms: int, row: dict) -> None:
        """Write one row. timestamp_ms is this program's own PC-side clock
        (the same one MovementCsvLogger uses), so it is what should be
        used to line the two files up; device_timestamp_ms is kept
        alongside it only as a reference for once real hardware exists."""
        csv_row = (
            [timestamp_ms, row["device_timestamp_ms"], row["seq"]]
            + row["emg1"]
            + row["emg2"]
            + [row["joy_x"], row["joy_y"], row["btn"]]
        )
        self.writer.writerow(csv_row)
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file when the program exits."""
        self.file.close()


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def mirror_landmarks_for_display(landmarks):
    """Mirror x-coordinates only, for drawing on the horizontally-flipped
    display frame. Detection runs on the unflipped camera frame so
    MediaPipe's LEFT_*/RIGHT_* labels are anatomically correct; the frame
    shown on screen is flipped afterward for natural mirror-style webcam
    behavior, so the skeleton drawn on it needs its x-coordinates mirrored
    to line up with it. y and visibility are untouched."""
    return [
        SmoothedLandmark(1.0 - landmark.x, landmark.y, landmark.visibility)
        for landmark in landmarks
    ]


def get_region_color(tracker: RegionTracker) -> tuple[int, int, int]:
    """Give PARTIAL stationary regions a separate visible color."""
    if (
        tracker.observability == "PARTIAL"
        and tracker.state == "STATIONARY"
    ):
        return PARTIAL_COLOR

    return MOVEMENT_COLORS[tracker.state]


def landmark_display_state(
    index: int,
    region_trackers: dict[str, RegionTracker],
) -> str:
    """Choose a state color for landmarks shared by multiple regions."""
    related_trackers = [
        tracker
        for region_name, tracker in region_trackers.items()
        if index in REGION_LANDMARKS[region_name]
    ]

    if any(tracker.state == "MOVING" for tracker in related_trackers):
        return "MOVING"

    if any(tracker.state == "STATIONARY" for tracker in related_trackers):
        return "STATIONARY"

    return "UNCERTAIN"


def draw_skeleton(
    frame,
    landmarks,
    region_trackers: dict[str, RegionTracker],
) -> None:
    """Draw visible skeleton connections and landmark points for one person."""
    height, width = frame.shape[:2]

    for region_name, connections in REGION_CONNECTIONS.items():
        tracker = region_trackers[region_name]

        if tracker.observability == "UNRELIABLE":
            continue

        color = get_region_color(tracker)

        thickness = (
            2
            if tracker.observability == "PARTIAL"
            else 3
        )

        for first_index, second_index in connections:
            first = landmarks[first_index]
            second = landmarks[second_index]

            if not (
                is_landmark_reliable(first)
                and is_landmark_reliable(second)
            ):
                continue

            first_point = (
                int(first.x * width),
                int(first.y * height),
            )

            second_point = (
                int(second.x * width),
                int(second.y * height),
            )

            cv2.line(
                frame,
                first_point,
                second_point,
                color,
                thickness,
                cv2.LINE_AA,
            )

    all_indices = set()

    for indices in REGION_LANDMARKS.values():
        all_indices.update(indices)

    for index in all_indices:
        landmark = landmarks[index]

        if not is_landmark_reliable(landmark):
            continue

        state = landmark_display_state(
            index,
            region_trackers,
        )

        color = MOVEMENT_COLORS[state]
        radius = 6 if state == "MOVING" else 4

        point = (
            int(landmark.x * width),
            int(landmark.y * height),
        )

        cv2.circle(frame, point, radius, color, -1)


def get_overall_state(
    region_trackers: dict[str, RegionTracker],
) -> str:
    """Use only regions with a valid motion measurement."""
    measurable_trackers = [
        tracker
        for tracker in region_trackers.values()
        if tracker.motion_available
    ]

    if not measurable_trackers:
        return "UNCERTAIN"

    if any(tracker.state == "MOVING" for tracker in measurable_trackers):
        return "MOVING"

    return "STATIONARY"


MOVEMENT_SUMMARY_LINE_HEIGHT = 28


def draw_movement_summary(
    frame,
    person_tracks: dict[int, PersonTrack],
) -> None:
    """Draw a minimal top-left readout: overall state, and which regions
    are moving, for each currently tracked person. This is the only
    on-screen overlay -- no skeleton, no panel -- so the feed stays clean
    for recording or demonstration."""
    show_track_id = len(person_tracks) > 1
    y = 30

    for track_id in sorted(person_tracks):
        track = person_tracks[track_id]
        overall_state = get_overall_state(track.region_trackers)

        prefix = f"Person {track_id}: " if show_track_id else ""

        if overall_state == "MOVING":
            moving_regions = [
                region_name.title()
                for region_name, tracker in track.region_trackers.items()
                if tracker.state == "MOVING"
            ]
            text = f"{prefix}MOVING - {', '.join(moving_regions)}"
        elif overall_state == "STATIONARY":
            text = f"{prefix}STILL"
        else:
            text = f"{prefix}..."

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            MOVEMENT_COLORS[overall_state],
            2,
            cv2.LINE_AA,
        )

        y += MOVEMENT_SUMMARY_LINE_HEIGHT


def draw_datetime_label(frame, current_datetime: datetime) -> None:
    """Burn the wall-clock date and time into the top-right corner of the
    video frame, so a viewer (or a saved recording) can read exactly when
    a given moment happened without needing the CSV alongside it. Kept in
    the opposite corner from draw_movement_summary's top-left readout so
    the two never overlap."""
    text = current_datetime.strftime("%Y-%m-%d  %H:%M:%S")

    height, width = frame.shape[:2]

    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1,
    )

    point = (width - text_width - 15, text_height + 15)

    cv2.putText(
        frame,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_emg_status_indicator(frame, receiver: EmgReceiver | None) -> None:
    """Show the real COM/data state below the date/time label."""
    if receiver is None:
        return

    if receiver.connection_error:
        text, color = "COM NOT CONNECTED", (0, 0, 255)
    elif not receiver.port_open:
        text, color = "COM NOT CONNECTED", (0, 0, 255)
    elif receiver.received_row_count == 0:
        text, color = f"{receiver.connected_port}: WAITING FOR DATA", (0, 165, 255)
    elif (
        receiver.last_data_monotonic is None
        or time.perf_counter() - receiver.last_data_monotonic > 2.0
    ):
        text, color = f"{receiver.connected_port}: DATA STOPPED", (0, 0, 255)
    else:
        text, color = (
            f"{receiver.connected_port}: RECEIVING DATA "
            f"({receiver.received_row_count} rows)",
            (0, 255, 0),
        )

    height, width = frame.shape[:2]
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
    )
    x = width - text_width - 15
    y = text_height + 42
    cv2.rectangle(frame, (x - 7, y - text_height - 6), (width - 8, y + 6), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )


# The plot's x-axis covers the whole session, from 0:00 at the moment
# recording started to the current elapsed time -- not a fixed-width
# rolling window. It grows as the session goes on rather than scrolling,
# which means a long session compresses earlier movements into less and
# less of the panel's width over time; that's an accepted trade-off for
# "the whole session is always visible," not an oversight.
PLOT_TICK_COUNT = 5

# Fixed y-axis ceiling rather than auto-scaling. Auto-scaling would rescale
# the whole plot every time the peak changes, which makes a "spike then
# settle" motion hard to read at a glance -- a fixed axis keeps the same
# movement always look like the same size spike. Set from real data this
# time: data/movement_data_20260820_161804.csv's arm/leg/head/torso trials
# peaked at 0.223 (LEFT_ARM), so 0.25 gives real headroom without being the
# arbitrary 2x margin used previously. Because aspect_correct_landmarks
# (see the geometry helpers) can only ever lower a region's recorded
# motion for a given real movement, not raise it, this ceiling stays valid
# after that fix -- it does not need re-deriving alongside it.
PLOT_Y_MAX = 0.25

PLOT_PANEL_WIDTH = 440
PLOT_MARGIN_LEFT = 55
PLOT_MARGIN_RIGHT = 15
PLOT_MARGIN_TOP = 20
PLOT_MARGIN_BOTTOM = 30

# Which regions get their own line on the plot. HEAD alone represents the
# whole head -- its RegionTracker already aggregates displacement across
# all 9 head landmarks (see calculate_head_raw_motion), and LEFT EYE/RIGHT
# EYE mostly just mirror HEAD's own motion (per the caveat where those
# regions were added: MediaPipe Pose has no independent gaze/blink signal),
# so plotting them separately was three near-duplicate lines instead of
# one. They're still tracked, colored on the skeleton, and logged to CSV
# as their own regions -- only the plot collapses them into HEAD.
PLOT_REGIONS = [
    region_name
    for region_name in REGION_LANDMARKS
    if region_name not in ("LEFT EYE", "RIGHT EYE")
]

# One distinguishable color per plotted region, cycled if there are ever
# more regions than colors. Deliberately separate from MOVEMENT_COLORS,
# which encodes *state* (moving/stationary/uncertain) elsewhere in the UI
# -- here each color instead identifies *which region* a line belongs to.
_PLOT_LINE_COLOR_CYCLE = [
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (255, 128, 0),    # azure
    (128, 255, 0),    # spring green
    (255, 255, 255),  # white
    (0, 128, 255),    # amber
]
PLOT_REGION_COLORS = {
    region_name: _PLOT_LINE_COLOR_CYCLE[index % len(_PLOT_LINE_COLOR_CYCLE)]
    for index, region_name in enumerate(PLOT_REGIONS)
}


def record_plot_samples(track: PersonTrack, timestamp_seconds: float) -> None:
    """Append this frame's motion reading to each region's plot history.

    The plot covers the whole session rather than a rolling window, so
    nothing is pruned here -- every sample stays for the life of the
    program.

    A region with no current motion reading (UNRELIABLE, or the track's
    first frame) gets no sample appended rather than a synthetic zero --
    the line simply has a gap there, which is a more honest picture than
    pretending the region was known to be still.
    """
    for region_name in PLOT_REGIONS:
        tracker = track.region_trackers[region_name]
        history = track.plot_history[region_name]

        if tracker.motion_available:
            history.append((timestamp_seconds, tracker.smoothed_motion))


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as M:SS, e.g. 75.3 -> "1:15"."""
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}:{secs:02d}"


def indexed_samples(base_index: int, values) -> list[tuple[float, float]]:
    """Pair a fixed-length rolling-window deque's values with the sample
    index each one actually corresponds to. base_index is the index of
    the *next* sample that will be added (EmgState.sample_count or
    .row_count), so the oldest value currently in `values` is
    base_index - len(values), and each one after it is +1.
    """
    start_index = base_index - len(values)
    return [
        (start_index + offset, value) for offset, value in enumerate(values)
    ]


def auto_range(
    values,
    fallback: tuple[float, float] = (0.0, 1.0),
    padding_fraction: float = 0.1,
) -> tuple[float, float]:
    """Return (min, max) over values with a little padding, or `fallback`
    if there is nothing to range over yet. For panels showing raw sensor
    units with no known fixed scale -- unlike the movement plot's
    calibrated PLOT_Y_MAX, there is nothing to calibrate an EMG axis
    against before real hardware exists.
    """
    if not values:
        return fallback

    low = min(values)
    high = max(values)

    if low == high:
        margin = max(abs(low) * padding_fraction, 1.0)
        return low - margin, high + margin

    span = high - low
    return low - span * padding_fraction, high + span * padding_fraction


def draw_time_series_panel(
    panel_width: int,
    panel_height: int,
    series,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    format_x_label,
    format_y_label=lambda value: f"{value:.2f}",
) -> np.ndarray:
    """Generic line-graph panel: PLOT_TICK_COUNT marks + gridlines on each
    axis, one colored line + legend entry per (label, color, points) entry
    in `series` (points is a list of (x, y) pairs). Shared by the movement
    plot, the EMG RMS envelope, the raw EMG waveform, and the joystick
    panel -- what differs between those is only the axis ranges and label
    formatting, passed in here rather than duplicated per panel.
    """
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    plot_width = panel_width - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT
    plot_height = panel_height - PLOT_MARGIN_TOP - PLOT_MARGIN_BOTTOM
    plot_bottom = PLOT_MARGIN_TOP + plot_height
    plot_right = PLOT_MARGIN_LEFT + plot_width

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    # Axes.
    cv2.line(
        panel,
        (PLOT_MARGIN_LEFT, PLOT_MARGIN_TOP),
        (PLOT_MARGIN_LEFT, plot_bottom),
        (120, 120, 120),
        1,
    )
    cv2.line(
        panel,
        (PLOT_MARGIN_LEFT, plot_bottom),
        (plot_right, plot_bottom),
        (120, 120, 120),
        1,
    )

    def to_pixel(x_value: float, y_value: float) -> tuple[int, int]:
        x = PLOT_MARGIN_LEFT + int((x_value - x_min) / x_span * plot_width)
        clamped_y = min(max(y_value, y_min), y_max)
        y = plot_bottom - int((clamped_y - y_min) / y_span * plot_height)
        return x, y

    # Y-axis: PLOT_TICK_COUNT marks from y_min to y_max, with a light
    # horizontal gridline at each (skipping the bottom one -- that's
    # already the axis line itself).
    for tick_index in range(PLOT_TICK_COUNT):
        fraction = tick_index / (PLOT_TICK_COUNT - 1)
        value = y_min + fraction * y_span
        y = plot_bottom - int(fraction * plot_height)

        if tick_index > 0:
            cv2.line(panel, (PLOT_MARGIN_LEFT, y), (plot_right, y), (50, 50, 50), 1)

        cv2.putText(
            panel, format_y_label(value), (5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA,
        )

    # X-axis: PLOT_TICK_COUNT marks from x_min to x_max, with a light
    # vertical gridline at each interior tick.
    for tick_index in range(PLOT_TICK_COUNT):
        fraction = tick_index / (PLOT_TICK_COUNT - 1)
        value = x_min + fraction * x_span
        x = PLOT_MARGIN_LEFT + int(fraction * plot_width)

        if 0 < tick_index < PLOT_TICK_COUNT - 1:
            cv2.line(panel, (x, PLOT_MARGIN_TOP), (x, plot_bottom), (50, 50, 50), 1)

        label = format_x_label(value)
        (label_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1,
        )
        label_x = min(max(x - label_width // 2, 0), panel_width - label_width)

        cv2.putText(
            panel, label, (label_x, panel_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA,
        )

    legend_y = PLOT_MARGIN_TOP + 4

    for label, color, points in series:
        pixel_points = [to_pixel(x_value, y_value) for x_value, y_value in points]

        for start_point, end_point in zip(pixel_points, pixel_points[1:]):
            cv2.line(panel, start_point, end_point, color, 2, cv2.LINE_AA)

        (label_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1,
        )
        legend_width = 18 + 5 + label_width
        legend_x = panel_width - legend_width - 8

        cv2.line(
            panel,
            (legend_x, legend_y),
            (legend_x + 18, legend_y),
            color,
            2,
        )
        cv2.putText(
            panel, label, (legend_x + 23, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )
        legend_y += 15

    return panel


def draw_movement_plot(
    panel_height: int,
    plot_history: dict[str, deque],
    current_time_seconds: float,
) -> np.ndarray:
    """The movement plot: elapsed time (M:SS, whole session) on x, smoothed
    motion on y, one line per PLOT_REGIONS entry, fixed 0..PLOT_Y_MAX so
    the same size movement always looks like the same size spike."""
    time_span = max(current_time_seconds, 5.0)

    series = [
        (
            region_name.title(),
            PLOT_REGION_COLORS[region_name],
            list(plot_history[region_name]),
        )
        for region_name in PLOT_REGIONS
    ]

    return draw_time_series_panel(
        panel_width=PLOT_PANEL_WIDTH,
        panel_height=panel_height,
        series=series,
        x_min=0.0,
        x_max=time_span,
        y_min=0.0,
        y_max=PLOT_Y_MAX,
        format_x_label=format_elapsed_time,
    )


def draw_emg_envelope_panel(
    panel_height: int,
    emg_state: EmgState,
    current_time_seconds: float,
) -> np.ndarray:
    """EMG activation-intensity trend (RMS envelope, see compute_rms) for
    both nodes across the whole session, on the same time-axis convention
    as draw_movement_plot -- this is the panel meant to be read side by
    side with it, since it is the one actually comparable to detected
    camera movement. Auto-ranged rather than a fixed ceiling like
    PLOT_Y_MAX, since there's no real EMG amplitude yet to calibrate a
    fixed one against.
    """
    time_span = max(current_time_seconds, 5.0)

    envelope_values = [value for _, value in emg_state.emg1_envelope] + [
        value for _, value in emg_state.emg2_envelope
    ]
    y_min, y_max = auto_range(envelope_values, fallback=(0.0, 100.0))
    y_min = min(y_min, 0.0)  # RMS is non-negative; always show the 0 baseline.

    series = [
        ("EMG 1 RMS", _PLOT_LINE_COLOR_CYCLE[0], list(emg_state.emg1_envelope)),
        ("EMG 2 RMS", _PLOT_LINE_COLOR_CYCLE[1], list(emg_state.emg2_envelope)),
    ]

    return draw_time_series_panel(
        panel_width=PLOT_PANEL_WIDTH,
        panel_height=panel_height,
        series=series,
        x_min=0.0,
        x_max=time_span,
        y_min=y_min,
        y_max=y_max,
        format_x_label=format_elapsed_time,
    )


def draw_emg_raw_panel(panel_height: int, emg_state: EmgState) -> np.ndarray:
    """Raw EMG waveform for both nodes, most recent EMG_WINDOW_SAMPLES
    samples only -- a sliding sample-index window, not the whole-session
    time axis the other two panels use, because at ~1 kHz the raw
    waveform needs real temporal resolution to be readable; the whole
    session would just be a dense blur here. This is the signal-quality /
    "is the sensor actually working" view, not the correlation view.
    """
    x_max = max(emg_state.sample_count, EMG_WINDOW_SAMPLES)
    x_min = x_max - EMG_WINDOW_SAMPLES

    emg1_points = indexed_samples(emg_state.sample_count, emg_state.emg1_raw)
    emg2_points = indexed_samples(emg_state.sample_count, emg_state.emg2_raw)

    y_min, y_max = auto_range(
        list(emg_state.emg1_raw) + list(emg_state.emg2_raw),
    )

    series = [
        ("EMG 1 (raw)", _PLOT_LINE_COLOR_CYCLE[0], emg1_points),
        ("EMG 2 (raw)", _PLOT_LINE_COLOR_CYCLE[1], emg2_points),
    ]

    return draw_time_series_panel(
        panel_width=PLOT_PANEL_WIDTH,
        panel_height=panel_height,
        series=series,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        format_x_label=lambda value: f"{int(value)}",
    )


def draw_emg_channel_panel(
    panel_height: int,
    sample_count: int,
    samples: deque,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw one received EMG channel as a rolling raw waveform."""
    x_max = max(sample_count, EMG_WINDOW_SAMPLES)
    x_min = x_max - EMG_WINDOW_SAMPLES
    values = list(samples)
    y_min, y_max = auto_range(values)

    return draw_time_series_panel(
        panel_width=PLOT_PANEL_WIDTH,
        panel_height=panel_height,
        series=[(label, color, indexed_samples(sample_count, samples))],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        format_x_label=lambda value: f"{int(value)}",
    )


def draw_live_emg_monitor(emg_state: EmgState, panel_height: int) -> np.ndarray:
    """Show raw EMG 1, raw EMG 2, and joystick data from real COM rows."""
    emg1_height = panel_height // 3
    emg2_height = panel_height // 3
    joystick_height = panel_height - emg1_height - emg2_height

    emg1_panel = draw_emg_channel_panel(
        emg1_height,
        emg_state.sample_count,
        emg_state.emg1_raw,
        "EMG 1 raw",
        _PLOT_LINE_COLOR_CYCLE[0],
    )
    emg2_panel = draw_emg_channel_panel(
        emg2_height,
        emg_state.sample_count,
        emg_state.emg2_raw,
        "EMG 2 raw",
        _PLOT_LINE_COLOR_CYCLE[1],
    )
    joystick_panel = draw_joystick_panel(joystick_height, emg_state)
    return np.vstack((emg1_panel, emg2_panel, joystick_panel))


def draw_joystick_panel(panel_height: int, emg_state: EmgState) -> np.ndarray:
    """Joystick X/Y and button, most recent EMG_WINDOW_SAMPLES rows --
    same sliding-window idea as draw_emg_raw_panel, but indexed by
    row_count rather than sample_count since the joystick only updates
    once per received row, not once per individual EMG sample."""
    x_max = max(emg_state.row_count, EMG_WINDOW_SAMPLES)
    x_min = x_max - EMG_WINDOW_SAMPLES

    joy_x_points = indexed_samples(emg_state.row_count, emg_state.joy_x_raw)
    joy_y_points = indexed_samples(emg_state.row_count, emg_state.joy_y_raw)
    btn_points = indexed_samples(emg_state.row_count, emg_state.btn_raw)

    y_min, y_max = auto_range(
        list(emg_state.joy_x_raw)
        + list(emg_state.joy_y_raw)
        + list(emg_state.btn_raw),
    )

    series = [
        ("Joy X", _PLOT_LINE_COLOR_CYCLE[2], joy_x_points),
        ("Joy Y", _PLOT_LINE_COLOR_CYCLE[3], joy_y_points),
        ("Button", _PLOT_LINE_COLOR_CYCLE[4], btn_points),
    ]

    return draw_time_series_panel(
        panel_width=PLOT_PANEL_WIDTH,
        panel_height=panel_height,
        series=series,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        format_x_label=lambda value: f"{int(value)}",
    )


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

def choose_capture_mode() -> tuple[bool, str | None, int]:
    """Show clickable startup buttons for visual-only or visual + EMG mode."""
    window_name = "Choose Capture Mode"
    canvas = np.zeros((420, 820, 3), dtype=np.uint8)
    selection = {"use_emg": None}
    visual_button = (80, 180, 740, 255)
    emg_button = (80, 285, 740, 360)

    def select_mode(event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return

        if visual_button[0] <= x <= visual_button[2] and visual_button[1] <= y <= visual_button[3]:
            selection["use_emg"] = False
        elif emg_button[0] <= x <= emg_button[2] and emg_button[1] <= y <= emg_button[3]:
            selection["use_emg"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, select_mode)

    while selection["use_emg"] is None:
        canvas[:] = (28, 28, 28)
        cv2.putText(canvas, "Body Movement Detection", (145, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Choose how you want to run this session", (185, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, visual_button[:2], visual_button[2:], (54, 123, 54), -1)
        cv2.rectangle(canvas, emg_button[:2], emg_button[2:], (120, 72, 35), -1)
        cv2.putText(canvas, "Visual Motion Only", (265, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Visual Motion + EMG Hardware", (175, 331), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press Esc to exit", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyWindow(window_name)
            raise SystemExit("Cancelled before starting capture.")

    cv2.destroyWindow(window_name)

    if not selection["use_emg"]:
        return False, None, EMG_BAUD_RATE

    port = input(
        "Serial port (for example COM4; press Enter to auto-detect): "
    ).strip() or None
    baud_text = input(f"Baud rate [{EMG_BAUD_RATE}]: ").strip()

    try:
        baud_rate = int(baud_text) if baud_text else EMG_BAUD_RATE
    except ValueError:
        print(f"Invalid baud rate; using {EMG_BAUD_RATE}.")
        baud_rate = EMG_BAUD_RATE

    return True, port, baud_rate


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    use_emg, emg_port, emg_baud_rate = choose_capture_mode()

    model_path = (
        project_directory
        / "models"
        / "pose_landmarker_full.task"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}"
        )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    track_manager = PersonTrackManager()

    # Loggers are deliberately created only when R starts recording. This
    # prevents empty CSV files from appearing merely because the program ran.
    csv_logger = None
    emg_csv_logger = None

    if CALIBRATION_MODE:
        print("R = start/stop recording (CSV files are created when recording starts)")
        print("Tab = switch which tracked person you're labeling")
        print("0-9 = select trial for the active person")
        print("S = LYING_SIDEWAYS_STILL")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {CAMERA_INDEX}."
        )

    start_time = time.perf_counter()
    start_datetime = datetime.now()
    frame_number = 0

    emg_receiver = None
    emg_state = EmgState() if use_emg else None

    if use_emg:
        emg_receiver = EmgReceiver(emg_port, emg_baud_rate, start_time)
        emg_receiver.start()
        print(
            f"EMG: connecting to real hardware "
            f"(port={emg_port or 'auto-detect'}, baud={emg_baud_rate})..."
        )
    else:
        print("EMG: disabled (visual motion only).")

    print(
        f"Camera started. Tracking up to {MAX_PEOPLE} people. "
        "Press Q or Esc to close."
    )

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame_number += 1

                # Detection runs on the frame exactly as the camera captured
                # it -- not mirrored. MediaPipe's LEFT_*/RIGHT_* landmarks
                # are only correct for anatomical left/right when the input
                # isn't flipped; mirroring first (as this used to do) feeds
                # the model an image indistinguishable from a person whose
                # left and right are swapped, which silently swaps every
                # LEFT_ARM/RIGHT_ARM/LEFT_LEG/RIGHT_LEG reading. The frame
                # is mirrored afterward, for display only, once detection
                # has already produced correctly-labeled landmarks.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                frame = cv2.flip(frame, 1)

                matched_landmarks = track_manager.update(result.pose_landmarks)

                for track_id, raw_landmarks in matched_landmarks.items():
                    track = track_manager.tracks[track_id]

                    landmarks = smooth_landmarks(
                        raw_landmarks,
                        track.smoothed_landmarks,
                    )
                    track.smoothed_landmarks = landmarks

                    # Motion tracking (but not drawing -- see
                    # aspect_correct_landmarks) works on an aspect-ratio-
                    # corrected copy so horizontal and vertical movement
                    # are measured on the same scale.
                    frame_height, frame_width = frame.shape[:2]
                    tracking_landmarks = aspect_correct_landmarks(
                        landmarks,
                        frame_width,
                        frame_height,
                    )

                    raw_body_scale = calculate_body_scale(tracking_landmarks)
                    body_scale = smooth_body_scale(
                        raw_body_scale,
                        track.motion_body_scale,
                    )
                    track.motion_body_scale = body_scale

                    for region_name, landmark_indices in (
                        REGION_LANDMARKS.items()
                    ):
                        track.region_trackers[region_name].update(
                            tracking_landmarks,
                            landmark_indices,
                            body_scale,
                        )

                    record_plot_samples(track, timestamp_ms / 1000)

                    draw_skeleton(
                        frame,
                        mirror_landmarks_for_display(landmarks),
                        track.region_trackers,
                    )

                    if csv_logger is not None:
                        overall_state = get_overall_state(track.region_trackers)

                        csv_logger.write_frame(
                            timestamp_ms,
                            frame_number,
                            track_id,
                            TRIAL_LABELS[track.trial_number],
                            body_scale,
                            track.region_trackers,
                            overall_state,
                        )

                if emg_receiver is not None:
                    # EMG rows arrive independently of video frames, so the
                    # receiver is drained without blocking the camera loop.
                    for emg_row in emg_receiver.drain():
                        update_emg_state(emg_state, emg_row)

                        if emg_csv_logger is not None:
                            emg_csv_logger.write_row(
                                int(emg_row["pc_timestamp_seconds"] * 1000),
                                emg_row,
                            )

                draw_movement_summary(frame, track_manager.tracks)

                current_datetime = start_datetime + timedelta(
                    seconds=timestamp_ms / 1000
                )
                draw_datetime_label(frame, current_datetime)
                draw_emg_status_indicator(frame, emg_receiver)

                active_track = track_manager.tracks.get(
                    track_manager.active_track_id
                )

                # Keep the display focused: one movement plot, plus one EMG
                # activation plot only when real EMG hardware is selected.
                # Stacking the plots keeps the total window narrow enough for
                # the full EMG legend to remain visible on typical screens.
                top_panel_height = frame.shape[0] // 2
                bottom_panel_height = frame.shape[0] - top_panel_height
                movement_panel = draw_movement_plot(
                    panel_height=(top_panel_height if emg_state is not None else frame.shape[0]),
                    plot_history=(
                        active_track.plot_history
                        if active_track is not None
                        else {
                            region_name: deque()
                            for region_name in PLOT_REGIONS
                        }
                    ),
                    current_time_seconds=timestamp_ms / 1000,
                )

                if emg_state is not None:
                    emg_panel = draw_emg_envelope_panel(
                        panel_height=bottom_panel_height,
                        emg_state=emg_state,
                        current_time_seconds=timestamp_ms / 1000,
                    )
                    combined_display = np.hstack(
                        (frame, np.vstack((movement_panel, emg_panel)))
                    )
                else:
                    combined_display = np.hstack((frame, movement_panel))

                cv2.imshow(
                    "Body Movement Detection",
                    combined_display,
                )

                if emg_state is not None:
                    cv2.imshow(
                        "Live COM EMG + Joystick",
                        draw_live_emg_monitor(emg_state, frame.shape[0]),
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("r"), ord("R")) and CALIBRATION_MODE:
                    if csv_logger is None:
                        session_timestamp = datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        csv_logger = MovementCsvLogger(
                            project_directory, session_timestamp
                        )
                        if use_emg:
                            emg_csv_logger = EmgCsvLogger(
                                project_directory, session_timestamp
                            )

                        print(f"Recording ON. Movement CSV: {csv_logger.path}")
                        if emg_csv_logger is not None:
                            print(f"EMG CSV: {emg_csv_logger.path}")
                    else:
                        csv_logger.close()
                        csv_logger = None

                        if emg_csv_logger is not None:
                            emg_csv_logger.close()
                            emg_csv_logger = None

                        print("Recording OFF.")

                elif key == 9:  # Tab
                    track_manager.cycle_active_track()
                    print(f"Active track: {track_manager.active_track_id}")

                elif (
                    key in TRIAL_KEY_BINDINGS
                    and track_manager.active_track_id is not None
                ):
                    trial_number = TRIAL_KEY_BINDINGS[key]
                    active_track = track_manager.tracks[
                        track_manager.active_track_id
                    ]
                    active_track.trial_number = trial_number

                    print(
                        f"Track {active_track.track_id} trial: "
                        f"{TRIAL_LABELS[trial_number]}"
                    )

                elif key in (ord("q"), 27):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        if emg_receiver is not None:
            emg_receiver.stop()

        if csv_logger is not None:
            csv_logger.close()

        if emg_csv_logger is not None:
            emg_csv_logger.close()


if __name__ == "__main__":
    main()
