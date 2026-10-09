"""Turn 33 keypoints into the few numbers the analysers need.
Angles are computed in PIXEL space (x*w, y*h) so a non-square frame doesn't distort them."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# MediaPipe landmark indices
LEFT = dict(sh=11, hip=23, knee=25, ankle=27, heel=29, toe=31)
RIGHT = dict(sh=12, hip=24, knee=26, ankle=28, heel=30, toe=32)


@dataclass
class PoseFeatures:
    side: str                    # which leg chain was used: "L" or "R"
    knee_angle: float            # hip-knee-ankle, 180 = straight
    hip_angle: float             # shoulder-hip-knee
    torso_lean: float            # shoulder-hip line vs vertical, 0 = upright
    knee_fwd: float              # (knee past toe) / shin length, >0 = knee ahead of toe
    knee_ankle_ratio: Optional[float]   # knee gap / ankle gap (front view signal)
    view_ratio: float            # shoulder width / torso length
    view: str = "side"           # filled in later: front / diagonal / side
    vis: float = 1.0


def _p(lm, i, w, h):
    return np.array([lm[i, 0] * w, lm[i, 1] * h], dtype=np.float64)


def angle_3pt(a, b, c):
    ba, bc = a - b, c - b
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))))


def _chain_vis(lm, ids):
    return float(np.mean([lm[ids[k], 3] for k in ("sh", "hip", "knee", "ankle")]))


def compute(lm, w, h, min_vis=0.5) -> Optional[PoseFeatures]:
    if lm is None:
        return None
    vl, vr = _chain_vis(lm, LEFT), _chain_vis(lm, RIGHT)
    side, ids, vis = ("L", LEFT, vl) if vl >= vr else ("R", RIGHT, vr)
    if vis < min_vis:
        return None

    sh, hip, knee, ankle = (_p(lm, ids[k], w, h) for k in ("sh", "hip", "knee", "ankle"))
    heel, toe = _p(lm, ids["heel"], w, h), _p(lm, ids["toe"], w, h)

    knee_angle = angle_3pt(hip, knee, ankle)
    hip_angle = angle_3pt(sh, hip, knee)
    if knee_angle is None or hip_angle is None:
        return None

    # torso lean from vertical (image y points down, so "up" is (0,-1))
    v = sh - hip
    n = np.linalg.norm(v)
    lean = float(np.degrees(np.arccos(np.clip(-v[1] / n, -1, 1)))) if n > 1e-6 else 0.0

    # knee ahead of toe, in the direction the foot points
    facing = 1.0 if toe[0] >= heel[0] else -1.0
    shin = np.linalg.norm(knee - ankle)
    knee_fwd = float((knee[0] - toe[0]) * facing / shin) if shin > 1e-6 else 0.0

    # front-view signal: how far apart are knees vs ankles
    ratio = None
    if min(lm[25, 3], lm[26, 3], lm[27, 3], lm[28, 3]) >= min_vis:
        ankle_gap = abs(lm[27, 0] - lm[28, 0]) * w
        if ankle_gap > 0.02 * w:
            ratio = float(abs(lm[25, 0] - lm[26, 0]) * w / ankle_gap)

    # view signal: shoulders wide apart on screen = facing camera
    sh_l, sh_r = _p(lm, 11, w, h), _p(lm, 12, w, h)
    hip_l, hip_r = _p(lm, 23, w, h), _p(lm, 24, w, h)
    torso = np.linalg.norm((sh_l + sh_r) / 2 - (hip_l + hip_r) / 2)
    view_ratio = float(np.linalg.norm(sh_l - sh_r) / torso) if torso > 1e-6 else 0.0

    return PoseFeatures(side, knee_angle, hip_angle, lean, knee_fwd, ratio, view_ratio, vis=vis)


class ViewEstimator:
    """Smooths the shoulder/torso ratio and labels the camera view."""

    def __init__(self, front_above=0.75, side_below=0.30, alpha=0.1):
        self.fa, self.sb, self.alpha, self.val = front_above, side_below, alpha, None

    def update(self, ratio):
        self.val = ratio if self.val is None else self.alpha * ratio + (1 - self.alpha) * self.val
        if self.val > self.fa:
            return "front"
        if self.val < self.sb:
            return "side"
        return "diagonal"
