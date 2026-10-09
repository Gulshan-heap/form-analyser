"""Writes one CSV row per rep. The empty 'label' column is for hand-labelling on Day 2."""
import csv
import os


class RepLogger:
    COLS = ["person", "index", "view", "side", "duration", "descent_time", "min_knee", "hip_at_bottom",
            "lean_at_bottom", "max_lean", "knee_fwd_bottom", "valgus_idx", "rule_faults", "rule_score", "label"]

    def __init__(self, path, person="unknown"):
        self.f, self.person = None, person
        if path:
            new = not os.path.exists(path)
            self.f = open(path, "a", newline="")
            self.w = csv.writer(self.f)
            if new:
                self.w.writerow(self.COLS)

    def write(self, rep, analysis):
        if not self.f:
            return
        d = rep.to_dict()
        r = lambda x: "" if x is None else round(x, 3)
        self.w.writerow([self.person, d["index"], d["view"], d["side"], r(d["duration"]),
                         r(d["descent_time"]), r(d["min_knee"]), r(d["hip_at_bottom"]),
                         r(d["lean_at_bottom"]), r(d["max_lean"]), r(d["knee_fwd_bottom"]),
                         r(d["valgus_idx"]), "|".join(f.code for f in analysis.faults),
                         analysis.score, ""])
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
