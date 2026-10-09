"""Synthetic-skeleton tests: no camera, no model. Run:  python tests/test_logic.py"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import Config
from features import compute, ViewEstimator
from rep_counter import RepCounter
from analysers import RuleAnalyser
from overlay import draw
from feedback import FeedbackManager, Speaker

W = H = 1000


def dirv(phi):  # angle from "up", positive toward +x; image y points down
    r = math.radians(phi)
    return np.array([math.sin(r), -math.cos(r)])


def side_pose(knee_deg, lean_deg=10, shin_fwd_deg=15, shin=0.2, thigh=0.22, torso=0.25):
    ankle = np.array([0.5, 0.9]); knee = ankle + shin * dirv(shin_fwd_deg)
    hip = knee + thigh * dirv(shin_fwd_deg + 180 + knee_deg)
    sh = hip + torso * dirv(lean_deg)
    lm = np.zeros((33, 4), np.float32); lm[:, 3] = 0.99
    for i in (11, 12): lm[i, :2] = sh
    for i in (23, 24): lm[i, :2] = hip
    for i in (25, 26): lm[i, :2] = knee
    for i in (27, 28): lm[i, :2] = ankle
    for i in (29, 30): lm[i, :2] = ankle + [-0.03, 0.0]
    for i in (31, 32): lm[i, :2] = ankle + [0.10, 0.0]
    return lm


def front_pose(knee_gap, ankle_gap=0.2):
    lm = np.zeros((33, 4), np.float32); lm[:, 3] = 0.99
    def put(ids, xs, y):
        for i, x in zip(ids, xs): lm[i, :2] = [x, y]
    put((11, 12), (0.38, 0.62), 0.30); put((23, 24), (0.42, 0.58), 0.55)
    put((25, 26), (0.5 - knee_gap / 2, 0.5 + knee_gap / 2), 0.72)
    put((27, 28), (0.5 - ankle_gap / 2, 0.5 + ankle_gap / 2), 0.90)
    put((29, 30), (0.5 - ankle_gap / 2, 0.5 + ankle_gap / 2), 0.91)
    put((31, 32), (0.5 - ankle_gap / 2, 0.5 + ankle_gap / 2), 0.93)
    return lm


def run(poses, view_override=None):
    cfg = Config(); c = RepCounter(cfg); an = RuleAnalyser(cfg); ve = ViewEstimator()
    reps = []
    for i, lm in enumerate(poses):
        f = compute(lm, W, H)
        f.view = view_override or ve.update(f.view_ratio)
        r = c.update(f, i / 30)
        if r: reps.append((r, an.analyse(r)))
    return reps


def squat_cycle(min_knee, n=40, **kw):
    ks = [175 - (175 - min_knee) * math.sin(math.pi * k / (n - 1)) for k in range(n)]
    return [side_pose(k, **kw) for k in ks]


# 1. features on a standing side-view pose
f = compute(side_pose(178, lean_deg=0, shin_fwd_deg=0), W, H)
assert 175 < f.knee_angle <= 180 and f.torso_lean < 2, f
assert ViewEstimator().update(f.view_ratio) == "side"
print("ok  standing features:", f"knee {f.knee_angle:.0f}, lean {f.torso_lean:.1f}, view_ratio {f.view_ratio:.2f}")

# 2. rep counting: 5 good squats, standing pauses between
poses = []
for _ in range(5):
    poses += [side_pose(176)] * 10 + squat_cycle(90)
poses += [side_pose(176)] * 10
reps = run(poses)
assert len(reps) == 5, len(reps)
assert all(abs(r.min_knee - 90) < 3 for r, _ in reps)
assert all(not a.faults for _, a in reps), [[x.code for x in a.faults] for _, a in reps]
print("ok  5 good squats -> 5 reps, no faults, depth", round(reps[0][0].min_knee))

# 3. wiggles don't count
assert len(run([side_pose(176)] * 10 + squat_cycle(145) + [side_pose(176)] * 10)) == 0
print("ok  half-bend wiggle not counted")

# 4. shallow depth flagged
r, a = run([side_pose(176)] * 5 + squat_cycle(120) + [side_pose(176)] * 5)[0]
assert [x.code for x in a.faults] == ["shallow_depth"], [x.code for x in a.faults]
print("ok  shallow squat flagged:", a.faults[0].label, "score", a.score)

# 5. forward lean flagged
r, a = run([side_pose(176, lean_deg=10)] * 5 + squat_cycle(90, lean_deg=70) + [side_pose(176)] * 5)[0]
assert "excess_lean" in [x.code for x in a.faults]
print("ok  forward lean flagged: max lean", round(r.max_lean), "->", [x.code for x in a.faults])

# 6. knees too far forward flagged
r, a = run([side_pose(176)] * 5 + squat_cycle(90, shin_fwd_deg=40) + [side_pose(176)] * 5)[0]
assert "knees_forward" in [x.code for x in a.faults], r.knee_fwd_bottom
print("ok  knees forward flagged: knee_fwd", round(r.knee_fwd_bottom, 2))

# 7. front view: knees caving in (squat in front view = knee angle shrinks, knee gap shrinks)
def front_cycle(cave):
    out = []
    for k in range(40):
        s = math.sin(math.pi * k / 39)
        p = front_pose(0.2 * (1 - cave * s))
        # fake knee angle by moving hips down (same chain geometry as side pose is not needed):
        out.append(p)
    return out
# knee angle in a pure front pose stays ~180 -> not counted; so feed side-style chain + front gaps
def hybrid(knee_deg, cave_frac):
    lm = side_pose(knee_deg)
    # add front-view width: shoulders/hips apart, knees/ankles gap
    for i, dx in ((11, -.12), (12, .12), (23, -.06), (24, .06)): lm[i, 0] += dx
    kx = lm[25, 0]; ax = lm[27, 0]
    lm[25, 0] = kx - 0.1 * cave_frac; lm[26, 0] = kx + 0.1 * cave_frac
    lm[27, 0] = ax - 0.1; lm[28, 0] = ax + 0.1
    return lm
good = [hybrid(175 - 85 * math.sin(math.pi * k / 39), 1.0) for k in range(40)]
bad = [hybrid(175 - 85 * math.sin(math.pi * k / 39), 1.0 - 0.6 * math.sin(math.pi * k / 39)) for k in range(40)]
rg, ag = run([hybrid(176, 1.0)] * 5 + good + [hybrid(176, 1.0)] * 5, "front")[0]
rb, ab = run([hybrid(176, 1.0)] * 5 + bad + [hybrid(176, 1.0)] * 5, "front")[0]
assert "knees_inward" not in [x.code for x in ag.faults], (rg.valgus_idx, ag.faults)
assert "knees_inward" in [x.code for x in ab.faults], rb.valgus_idx
assert "shallow_depth" not in ag.assessed      # front view must NOT judge depth
print(f"ok  front view: good valgus_idx {rg.valgus_idx:.2f}, caving {rb.valgus_idx:.2f} flagged; depth not assessed")

# 8. dropped detections mid-rep are handled
cfg = Config(); c = RepCounter(cfg)
for i, lm in enumerate([side_pose(176)] * 3 + squat_cycle(90)[:20]): c.update(compute(lm, W, H), i / 30)
for i in range(40): c.update(None, 1 + i / 30)
assert c.state == "UP"
print("ok  lost person mid-rep resets safely")

# 9. feedback manager + overlay render without crashing
import time
cfg = Config(); fb = FeedbackManager(cfg, Speaker(False))
r, a = run([side_pose(176)] * 5 + squat_cycle(120) + [side_pose(176)] * 5)[0]
fb.on_rep(r, a); assert fb.current_message() == "Too shallow"
c = RepCounter(cfg); c.count = 3
frame = np.zeros((480, 640, 3), np.uint8)
draw(frame, side_pose(100), compute(side_pose(100), 640, 480), c, r, a, fb.current_message(), 28.4, cfg)
import cv2; cv2.imwrite("/tmp/overlay_test.png", frame)
print("ok  overlay + feedback render")

# 10. MediaPipe options construct with the right argument names
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="x.task"), running_mode=vision.RunningMode.VIDEO,
    num_poses=1, min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5)
print("ok  PoseLandmarkerOptions arguments valid")
print("\nALL TESTS PASSED")
