# FormCoach — squat form analyser (laptop build, Pi-ready)

## Setup (once)
    pip install -r requirements.txt
    python get_model.py          # downloads pose_landmarker_lite.task into models/

## Run
    python main.py                                   # webcam
    python main.py --source clip.mp4 --person asha   # video file, tags the CSV rows
    python main.py --source http://PHONE_IP:8080/video   # phone as IP camera
    python main.py --view side                       # force view (viewpoint study)
    python main.py --headless --max-frames 300       # benchmark only
Keys: q quit | r reset | space pause

## Files
config.py      every threshold (calibrate these on real footage)
capture.py     webcam / file / phone URL  (only file that knows the source)
pose.py        MediaPipe PoseLandmarker + smoothing
features.py    angles, torso lean, knee-over-toe, view detection
rep_counter.py up/down state machine -> one Rep object per squat
analysers.py   Analyser A: rule-based faults (view-aware)
feedback.py    what to say, when (priority + repeat + cooldown)
overlay.py     skeleton, counter, gauge, banner
logger.py      reps.csv  (fill the 'label' column by hand on Day 2)
bench.py       FPS / per-stage latency  -> your Pi-performance numbers

## Test without camera/model
    python tests/test_logic.py

## Porting to the Pi
Copy the folder, `pip install` the same packages, run get_model.py, then lower
`--width` (e.g. 480 or 320) and compare bench output with the laptop.
