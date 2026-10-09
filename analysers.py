"""Analyser A: rule-based squat faults. Each rule only runs from camera views where it is
trustworthy (a 2D camera can't see knee cave from the side, or depth well from the front)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Fault:
    code: str
    label: str      # short, shown on screen
    cue: str        # what we say to the user
    penalty: int


@dataclass
class Analysis:
    faults: List[Fault] = field(default_factory=list)
    assessed: List[str] = field(default_factory=list)   # which checks were possible from this view
    score: int = 100


class RuleAnalyser:
    # most important first: safety before range of motion
    PRIORITY = ["knees_inward", "excess_lean", "knees_forward", "shallow_depth"]

    def __init__(self, cfg):
        self.c = cfg

    def analyse(self, rep) -> Analysis:
        c, v = self.c, rep.view
        side_ok = v in ("side", "diagonal")
        front_ok = v in ("front", "diagonal")
        a = Analysis()

        if side_ok:
            a.assessed += ["shallow_depth", "excess_lean", "knees_forward"]
            if rep.min_knee > c.depth_ok_angle:
                a.faults.append(Fault("shallow_depth", "Too shallow", "Go a little deeper", 20))
            if rep.max_lean > c.lean_max:
                a.faults.append(Fault("excess_lean", "Leaning forward", "Keep your chest up", 25))
            if rep.knee_fwd_bottom > c.knee_fwd_max:
                a.faults.append(Fault("knees_forward", "Knees too far forward",
                                      "Sit back into your hips", 20))
        if front_ok and rep.valgus_idx is not None:
            a.assessed.append("knees_inward")
            if rep.valgus_idx < c.valgus_min:
                a.faults.append(Fault("knees_inward", "Knees caving in",
                                      "Push your knees out", 30))

        a.faults.sort(key=lambda f: self.PRIORITY.index(f.code))
        a.score = max(0, 100 - sum(f.penalty for f in a.faults))
        return a
