"""Frame source. The ONLY file that knows where video comes from:
webcam index (0), video file, or phone stream URL (http://... / rtsp://...)."""
import time
import cv2


class VideoSource:
    def __init__(self, source, width=640):
        src = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.is_live = isinstance(src, int) or str(src).startswith(("http", "rtsp"))
        self.width = width
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.idx = 0
        self._t0 = time.monotonic()
        self._last_ts = -1

    def read(self):
        """Returns (frame_bgr, timestamp_ms) or (None, 0) at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None, 0
        h, w = frame.shape[:2]
        if w != self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)))
        if self.is_live:
            ts = int((time.monotonic() - self._t0) * 1000)
        else:
            ts = int(self.idx * 1000 / self.fps)
        ts = max(ts, self._last_ts + 1)          # MediaPipe needs strictly increasing
        self._last_ts = ts
        self.idx += 1
        return frame, ts

    def release(self):
        self.cap.release()
