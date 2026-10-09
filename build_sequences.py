"""Turn cached keypoints + per-rep labels into fixed-length sequences for the 1D-CNN / LSTM.

  python build_sequences.py cfrep        # needs data/keypoints/cfrep_squat/*.npz and data/cfrep/video_config.json
  python build_sequences.py mendeley     # needs data/keypoints/squat/*.npz; clip-level good/bad, reps found by the rep counter

Every rep becomes X[c, t]: SEQ_LEN time steps x len(CHANNELS) joint-angle signals, so the network sees the
whole movement, not a single number. Output: datasets/sequences_<name>.npz (small, committed to git so teammates can train without the videos)
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from config import Config
from features import compute
from pose import Smoother
from rep_counter import RepCounter

SEQ_LEN = 64
CHANNELS = ["knee_angle", "hip_angle", "torso_lean", "knee_fwd", "hip_below_knee"]
SCALE = np.array([180.0, 180.0, 90.0, 1.0, 1.0], dtype=np.float32)      # keeps every channel roughly in -1..1
VIEWS = {"diag": "diagonal", "diagonal": "diagonal", "front": "front", "side": "side"}


def frame_signals(npz, cfg, view=None):
    """Per-frame values of CHANNELS from cached landmarks, smoothed like the live pipeline (NaN = not found).
    If `view` is given the live rep counter is run too. Returns (signals, [(start_frame, end_frame), ...])."""
    lm, w, h = npz["lm"], int(npz["w"]), int(npz["h"])
    fps = float(npz["fps"])
    smooth = Smoother(cfg.smooth_alpha, cfg.min_visibility)
    counter = RepCounter(cfg) if view else None
    out = np.full((len(lm), len(CHANNELS)), np.nan, np.float32)
    segs = []
    for i, frame in enumerate(lm):
        f = compute(smooth(None if np.isnan(frame[0, 0]) else frame), w, h, cfg.min_visibility)
        if f:
            out[i] = [f.knee_angle, f.hip_angle, f.torso_lean, f.knee_fwd, f.hip_below_knee]
            f.view = view
        if counter:
            rep = counter.update(f, i / fps)
            if rep:
                segs.append((int(round(rep.t_start * fps)), int(round(rep.t_end * fps))))
    return out, segs


def resample(sig):
    """(frames, channels) -> (channels, SEQ_LEN); gaps are bridged by interpolation, ends are held."""
    t = np.arange(len(sig), dtype=np.float32)
    ok = ~np.isnan(sig).any(axis=1)
    if ok.sum() < 3:
        return None
    filled = np.stack([np.interp(t, t[ok], sig[ok, c]) for c in range(sig.shape[1])], axis=1)
    grid = np.linspace(0, len(sig) - 1, SEQ_LEN)
    seq = np.stack([np.interp(grid, t, filled[:, c]) for c in range(sig.shape[1])])
    return (seq / SCALE[:, None]).astype(np.float32), float(ok.mean())


def parse_cfrep_name(name):
    m = re.match(r"squat_(diag|front|side)_([mw])(\d)\d?_", name)       # m32 is treated as a second take of m3
    return VIEWS[m.group(1)], m.group(2) + m.group(3)


def build_cfrep(args, cfg):
    cfg_json = {v["filename"]: v for v in json.load(open(os.path.join(args.cfrep, "video_config.json")))
                if v["exercise"] == "squat"}
    rows = []
    for name, v in sorted(cfg_json.items()):
        path = os.path.join(args.keypoints, os.path.splitext(name)[0] + ".npz")
        if not os.path.exists(path):
            print("missing keypoints, skipped:", name)
            continue
        npz = np.load(path)
        sig, _ = frame_signals(npz, cfg)
        view, person = parse_cfrep_name(name)
        fps = float(npz["fps"])
        for k, seg in enumerate(s for s in v["segments"] if s["label"] in ("rep", "no-rep")):
            part = sig[seg["start"]:seg["end"] + 1]
            res = resample(part)
            if res is None:
                print(f"  unusable rep (pose lost): {name} #{k}")
                continue
            seq, found = res
            rows.append(dict(x=seq, y=int(seg["label"] == "no-rep"), person=person, view=view, clip=name,
                             idx=k, dur=(seg["end"] - seg["start"] + 1) / fps, found=found,
                             min_knee=float(np.nanmin(part[:, 0])), max_hbk=float(np.nanmax(part[:, 4]))))
    return rows


def build_mendeley(args, cfg):
    """Clip-level labels: the rep counter finds the reps, every rep inherits its clip's good/bad label."""
    skip = set()
    dup = os.path.join(args.videos, "DUPLICATES.txt")
    if os.path.exists(dup):
        skip = {os.path.splitext(line.strip())[0] for line in open(dup) if line.strip()}
    rx = re.compile(r"subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)$", re.I)
    rows, clips = [], 0
    for path in sorted(glob.glob(os.path.join(args.keypoints, "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = rx.match(stem)
        if not m or stem in skip:
            continue
        person, _, quality, view = m.groups()
        view, clips = view.lower(), clips + 1
        npz = np.load(path)
        sig, segs = frame_signals(npz, cfg, view)
        fps = float(npz["fps"])
        for k, (a, b) in enumerate(segs):
            if not 0.5 <= (b - a + 1) / fps <= 8.0:                      # not a plausible squat
                continue
            part = sig[a:b + 1]
            res = resample(part)
            if res is None:
                continue
            seq, found = res
            rows.append(dict(x=seq, y=int(quality.lower() == "bad"), person=person, view=view, clip=stem + ".mp4",
                             idx=k, dur=(b - a + 1) / fps, found=found,
                             min_knee=float(np.nanmin(part[:, 0])), max_hbk=float(np.nanmax(part[:, 4]))))
    print(f"{clips} clips -> {len(rows)} reps (excluded as duplicate/mislabelled: {len(skip)})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["cfrep", "mendeley"])
    ap.add_argument("--cfrep", default="data/cfrep")
    ap.add_argument("--keypoints", default="")
    ap.add_argument("--videos", default="data/videos/squat", help="mendeley: folder with DUPLICATES.txt")
    args = ap.parse_args()
    args.keypoints = args.keypoints or ("data/keypoints/cfrep_squat" if args.dataset == "cfrep" else "data/keypoints/squat")
    rows = (build_cfrep if args.dataset == "cfrep" else build_mendeley)(args, Config())
    if not rows:
        raise SystemExit("No sequences built.")
    os.makedirs("datasets", exist_ok=True)
    out = f"datasets/sequences_{args.dataset}.npz"
    np.savez_compressed(
        out, x=np.stack([r["x"] for r in rows]), y=np.array([r["y"] for r in rows]),
        person=np.array([r["person"] for r in rows]), view=np.array([r["view"] for r in rows]),
        clip=np.array([r["clip"] for r in rows]), idx=np.array([r["idx"] for r in rows]),
        dur=np.array([r["dur"] for r in rows], np.float32), found=np.array([r["found"] for r in rows], np.float32),
        min_knee=np.array([r["min_knee"] for r in rows], np.float32),
        max_hbk=np.array([r["max_hbk"] for r in rows], np.float32), channels=np.array(CHANNELS))
    y = np.array([r["y"] for r in rows])
    print(f"{len(rows)} reps ({int((y == 0).sum())} valid/good, {int(y.sum())} invalid/bad) from "
          f"{len({r['person'] for r in rows})} people -> {out}")


if __name__ == "__main__":
    main()
