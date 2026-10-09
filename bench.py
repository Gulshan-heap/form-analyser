"""Tiny profiler. This is your 'Pi performance' experiment: run it on laptop now, Pi later."""
import time
from contextlib import contextmanager
import numpy as np


class Bench:
    def __init__(self):
        self.stages, self.frame_ms, self._t = {}, [], time.perf_counter()
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        t = time.perf_counter()
        yield
        self.stages.setdefault(name, []).append((time.perf_counter() - t) * 1000)

    def frame_done(self):
        now = time.perf_counter()
        self.frame_ms.append((now - self._t) * 1000)
        self._t = now

    def fps(self, n=30):
        recent = self.frame_ms[-n:]
        return 1000 / np.mean(recent) if recent else 0.0

    def summary(self):
        if not self.frame_ms:
            return "no frames processed"
        lines = [f"frames: {len(self.frame_ms)}   mean FPS: {1000/np.mean(self.frame_ms):.1f}   "
                 f"frame latency p50/p95: {np.percentile(self.frame_ms,50):.1f}/"
                 f"{np.percentile(self.frame_ms,95):.1f} ms"]
        for k, v in self.stages.items():
            lines.append(f"  {k:<8} mean {np.mean(v):6.2f} ms   p95 {np.percentile(v,95):6.2f} ms")
        return "\n".join(lines)
