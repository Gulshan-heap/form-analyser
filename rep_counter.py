"""Rep counting: a small up/down state machine on the knee angle (with hysteresis)."""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Rep:
    index: int
    t_start: float
    t_end: float
    duration: float            # whole rep, seconds
    descent_time: float        # start -> bottom, seconds
    min_knee: float            # deepest knee angle (smaller = deeper)
    hip_at_bottom: float
    lean_at_bottom: float
    max_lean: float
    knee_fwd_bottom: float
    valgus_idx: Optional[float]   # knee/ankle gap at bottom vs at start (<1 = knees collapsing in)
    view: str
    side: str

    def to_dict(self):
        return asdict(self)


class RepCounter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.count = 0
        self.state = "UP"
        self.knee_angle = None
        self._missing = 0
        self._cur = None

    def update(self, f, t) -> Optional[Rep]:
        """Feed one frame's features (or None). Returns a Rep when one is completed."""
        c = self.cfg
        if f is None:
            self._missing += 1
            if self._missing > c.max_missing_frames and self.state == "DOWN":
                self.state, self._cur = "UP", None
            return None
        self._missing = 0
        self.knee_angle = f.knee_angle

        if self.state == "UP":
            if f.knee_angle < c.descend_angle:
                self.state = "DOWN"
                self._cur = dict(t0=t, min=f.knee_angle, t_bottom=t, bottom=f,
                                 max_lean=f.torso_lean, ratio0=f.knee_ankle_ratio)
            return None

        cur = self._cur
        cur["max_lean"] = max(cur["max_lean"], f.torso_lean)
        if f.knee_angle < cur["min"]:
            cur.update(min=f.knee_angle, t_bottom=t, bottom=f)

        if f.knee_angle > c.stand_angle:                 # back up -> rep finished
            self.state, self._cur = "UP", None
            if cur["min"] >= c.count_angle:              # barely bent: wiggle, not a rep
                return None
            self.count += 1
            b = cur["bottom"]
            valgus = None
            if cur["ratio0"] and b.knee_ankle_ratio:
                valgus = b.knee_ankle_ratio / cur["ratio0"]
            return Rep(self.count, cur["t0"], t, t - cur["t0"], cur["t_bottom"] - cur["t0"],
                       cur["min"], b.hip_angle, b.torso_lean, cur["max_lean"],
                       b.knee_fwd, valgus, b.view, b.side)
        return None
