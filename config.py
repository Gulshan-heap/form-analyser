"""All tunable numbers live here. On the Pi you will mostly change width/height/smoothing."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- model / capture ---
    model_path: str = "models/pose_landmarker_lite.task"
    width: int = 640                 # frames are resized to this width (keep aspect)
    min_visibility: float = 0.5      # ignore joints MediaPipe is unsure about
    smooth_alpha: float = 0.55       # EMA on landmarks: 1.0 = no smoothing, lower = smoother

    # --- view detection (shoulder width / torso length) ---
    view_front_above: float = 0.75
    view_side_below: float = 0.30

    # --- rep counting on knee angle (degrees, 180 = straight leg) ---
    stand_angle: float = 160         # above this = standing
    descend_angle: float = 150       # below this = rep has started
    count_angle: float = 130         # a rep only counts if it gets below this
    max_missing_frames: int = 30     # drop a half-finished rep if the person vanishes

    # --- squat fault thresholds (CALIBRATE on your own footage) ---
    depth_ok_angle: float = 105      # knee angle at bottom must be <= this
    lean_max: float = 55             # torso lean from vertical (side/diagonal view)
    knee_fwd_max: float = 0.12       # knee past toe, as a fraction of shin length
    valgus_min: float = 0.85         # knee gap / ankle gap vs standing (front view)

    # --- feedback ---
    fault_consecutive: int = 2       # speak only if same fault in N reps in a row
    voice_cooldown_s: float = 4.0
    message_show_s: float = 3.0
    voice: bool = True
