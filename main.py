"""FormCoach - squat analyser (laptop version).

  python main.py                          # webcam
  python main.py --source clip.mp4        # video file
  python main.py --source http://192.168.1.5:8080/video   # phone IP-camera app
Keys: q quit | r reset reps | space pause
"""
import argparse
import cv2

from config import Config
from capture import VideoSource
from pose import PoseEstimator, Smoother
from features import compute, ViewEstimator
from rep_counter import RepCounter
from analysers import RuleAnalyser
from feedback import FeedbackManager, Speaker
from overlay import draw
from logger import RepLogger
from bench import Bench


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--model", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--mirror", action="store_true", help="flip webcam like a mirror")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--headless", action="store_true", help="no window (for benchmarks)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--view", choices=["front", "diagonal", "side"], help="override auto view detection")
    p.add_argument("--person", default="unknown")
    p.add_argument("--log", default="reps.csv", help="per-rep CSV ('' to disable)")
    p.add_argument("--record", default="", help="save the annotated video to this .mp4")
    return p.parse_args()


def main():
    args = parse()
    cfg = Config()
    if args.model:
        cfg.model_path = args.model
    if args.width:
        cfg.width = args.width
    cfg.voice = not args.no_voice

    src = VideoSource(args.source, cfg.width)
    pose = PoseEstimator(cfg.model_path)
    smoother = Smoother(cfg.smooth_alpha, cfg.min_visibility)
    view_est = ViewEstimator(cfg.view_front_above, cfg.view_side_below)
    counter, analyser = RepCounter(cfg), RuleAnalyser(cfg)
    feedback = FeedbackManager(cfg, Speaker(cfg.voice))
    logger = RepLogger(args.log, args.person)
    bench, writer = Bench(), None
    last_rep = last_analysis = None
    paused = False

    try:
        while True:
            if not paused:
                with bench.stage("capture"):
                    frame, ts = src.read()
                if frame is None:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                with bench.stage("pose"):
                    lm = smoother(pose.detect(frame, ts))

                with bench.stage("logic"):
                    feat = compute(lm, w, h, cfg.min_visibility)
                    if feat:
                        feat.view = args.view or view_est.update(feat.view_ratio)
                    rep = counter.update(feat, ts / 1000.0)
                    if rep:
                        last_rep, last_analysis = rep, analyser.analyse(rep)
                        feedback.on_rep(rep, last_analysis)
                        logger.write(rep, last_analysis)
                        print(f"rep {rep.index}: depth {rep.min_knee:.0f} deg, view {rep.view}, "
                              f"score {last_analysis.score}, "
                              f"faults {[f.code for f in last_analysis.faults] or 'none'}")

                with bench.stage("draw"):
                    draw(frame, lm, feat, counter, last_rep, last_analysis,
                         feedback.current_message(), bench.fps(), cfg)
                bench.frame_done()

                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src.fps if not src.is_live else 25, (w, h))
                    writer.write(frame)
                if args.max_frames and len(bench.frame_ms) >= args.max_frames:
                    break

            if not args.headless:
                cv2.imshow("FormCoach", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    counter.reset(); smoother.reset(); last_rep = last_analysis = None
                if k == ord(" "):
                    paused = not paused
    finally:
        src.release(); pose.close(); logger.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("\n--- benchmark ---\n" + bench.summary())


if __name__ == "__main__":
    main()
