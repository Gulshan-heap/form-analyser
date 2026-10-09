"""Run MediaPipe once over every video in a folder and cache the raw landmarks, so features and rep
logic can be re-run in seconds. Resumable: clips that already have a .npz are skipped.

  python extract_keypoints.py data/videos/squat                 # -> data/keypoints/squat/<clip>.npz
  python extract_keypoints.py data/videos/squat --workers 2

Each .npz holds lm (frames, 33, 4) = x, y, z, visibility (NaN where no person was found, unsmoothed),
ts (ms), and the resized frame width/height that the x/y values refer to.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

from config import Config
from capture import VideoSource
from pose import PoseEstimator


def process(job):
    path, dest, model, width = job
    src = VideoSource(path, width)
    pose = PoseEstimator(model)
    lms, tss, h = [], [], 0
    try:
        while True:
            frame, ts = src.read()
            if frame is None:
                break
            h = frame.shape[0]
            lm = pose.detect(frame, ts)
            lms.append(np.full((33, 4), np.nan, np.float32) if lm is None else lm)
            tss.append(ts)
    finally:
        src.release()
        pose.close()
    np.savez_compressed(dest, lm=np.array(lms), ts=np.array(tss), w=width, h=h, fps=src.fps)
    found = float(np.mean([not np.isnan(x[0, 0]) for x in lms])) if lms else 0.0
    return os.path.basename(path), len(lms), found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = Config()
    out = args.out or os.path.join("data", "keypoints", os.path.basename(os.path.normpath(args.folder)))
    os.makedirs(out, exist_ok=True)
    jobs = []
    for p in sorted(glob.glob(os.path.join(args.folder, "*.mp4"))):
        dest = os.path.join(out, os.path.splitext(os.path.basename(p))[0] + ".npz")
        if not os.path.exists(dest):
            jobs.append((p, dest, cfg.model_path, cfg.width))
    print(f"{len(jobs)} clips to process ({args.workers} workers) -> {out}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (name, frames, found) in enumerate(pool.imap_unordered(process, jobs), 1):
            print(f"[{n}/{len(jobs)}] {name}: {frames} frames, person found in {found:.0%}, "
                  f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)


if __name__ == "__main__":
    main()
