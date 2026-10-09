"""Download the MediaPipe pose model once:  python get_model.py"""
import os
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
DEST = "models/pose_landmarker_lite.task"

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    if os.path.exists(DEST):
        print("Model already present:", DEST)
    else:
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, DEST)
        print("Saved to", DEST, f"({os.path.getsize(DEST)/1e6:.1f} MB)")
