"""Pose estimation (MediaPipe PoseLandmarker, Tasks API) + landmark smoothing."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class PoseEstimator:
    def __init__(self, model_path, conf=0.5):
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=conf,
            min_pose_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, ts_ms):
        """Returns (33,4) array [x, y, z, visibility], x/y normalised 0..1, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, int(ts_ms))
        if not res.pose_landmarks:
            return None
        pts = res.pose_landmarks[0]
        return np.array([[p.x, p.y, p.z, p.visibility] for p in pts], dtype=np.float32)

    def close(self):
        self._lm.close()


class Smoother:
    """Exponential moving average on landmarks; low-visibility joints keep their old position."""

    def __init__(self, alpha=0.55, min_vis=0.5):
        self.alpha, self.min_vis, self.prev = alpha, min_vis, None

    def reset(self):
        self.prev = None

    def __call__(self, lm):
        if lm is None:
            return None
        if self.prev is None:
            self.prev = lm.copy()
            return self.prev
        out = lm.copy()
        good = lm[:, 3] >= self.min_vis
        a = self.alpha
        out[good, :3] = a * lm[good, :3] + (1 - a) * self.prev[good, :3]
        out[~good, :3] = self.prev[~good, :3]
        self.prev = out
        return out
