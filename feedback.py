"""Decides WHAT to say and WHEN: most important fault first, only if it repeats,
with a cooldown so the app never nags."""
import queue
import threading
import time


class Speaker:
    """Non-blocking text-to-speech. Falls back to silence if pyttsx3 isn't installed."""

    def __init__(self, enabled=True):
        self.q = queue.Queue()
        self.ok = False
        if enabled:
            try:
                import pyttsx3  # noqa: F401
                self.ok = True
                threading.Thread(target=self._run, daemon=True).start()
            except Exception:
                print("[voice] pyttsx3 not available, voice disabled")

    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            engine.say(self.q.get())
            engine.runAndWait()

    def say(self, text):
        if self.ok:
            self.q.put(text)


class FeedbackManager:
    def __init__(self, cfg, speaker):
        self.c, self.speaker = cfg, speaker
        self.streak = {}              # fault code -> consecutive reps with that fault
        self.message, self.message_until = "", 0.0
        self._last_spoken = 0.0

    def on_rep(self, rep, analysis):
        now = time.monotonic()
        codes = [f.code for f in analysis.faults]
        for code in list(self.streak):
            if code not in codes:
                self.streak[code] = 0
        for code in codes:
            self.streak[code] = self.streak.get(code, 0) + 1

        if analysis.faults:
            top = analysis.faults[0]
            self._show(top.label, now)
            if (self.streak[top.code] >= self.c.fault_consecutive
                    and now - self._last_spoken > self.c.voice_cooldown_s):
                self.speaker.say(top.cue)
                self._last_spoken = now
        else:
            self._show("Good rep!", now)
            if rep.index % 5 == 0 and now - self._last_spoken > self.c.voice_cooldown_s:
                self.speaker.say("Nice form")
                self._last_spoken = now

    def _show(self, text, now):
        self.message, self.message_until = text, now + self.c.message_show_s

    def current_message(self):
        return self.message if time.monotonic() < self.message_until else ""
