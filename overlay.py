"""Everything drawn on screen: skeleton, rep counter, knee-angle gauge, feedback banner."""
import cv2
import numpy as np

BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
         (27, 31), (28, 32), (27, 29), (28, 30), (11, 13), (13, 15), (12, 14), (14, 16)]
GREEN, AMBER, RED, WHITE, DARK = (110, 220, 90), (0, 190, 255), (70, 70, 235), (245, 245, 245), (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, DARK), alpha, roi, 1 - alpha, 0, roi)


def _text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _score_color(score):
    return GREEN if score >= 80 else AMBER if score >= 55 else RED


def draw(frame, lm, feat, counter, last_rep, last_analysis, message, fps, cfg):
    h, w = frame.shape[:2]
    col = _score_color(last_analysis.score) if last_analysis else GREEN

    # skeleton
    if lm is not None:
        pts = {i: (int(lm[i, 0] * w), int(lm[i, 1] * h)) for i in range(33) if lm[i, 3] >= cfg.min_visibility}
        for a, b in BONES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], col, 3, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame, p, 4, WHITE, -1, cv2.LINE_AA)

    # top-left panel
    _panel(frame, 10, 10, 230, 138)
    _text(frame, "REPS", (22, 36), 0.55, (200, 200, 200))
    _text(frame, str(counter.count), (22, 98), 2.2, WHITE, 3)
    _text(frame, counter.state, (150, 36), 0.55, AMBER if counter.state == "DOWN" else GREEN, 2)
    view = feat.view if feat else "-"
    _text(frame, f"view: {view}", (22, 124), 0.5, (200, 200, 200))
    _text(frame, f"{fps:4.1f} fps", (150, 124), 0.5, (200, 200, 200))

    # last-rep score card
    if last_rep and last_analysis:
        _panel(frame, 10, 146, 230, 206)
        _text(frame, f"Rep {last_rep.index} score", (22, 170), 0.55, (200, 200, 200))
        _text(frame, str(last_analysis.score), (22, 200), 0.9, _score_color(last_analysis.score), 2)
        _text(frame, f"depth {last_rep.min_knee:.0f} deg", (95, 200), 0.5, WHITE)

    # knee-angle gauge on the right edge (60..180 deg)
    gx0, gx1, gy0, gy1 = w - 34, w - 16, 60, h - 60
    _panel(frame, gx0 - 6, gy0 - 28, gx1 + 6, gy1 + 10, 0.45)
    _text(frame, "KNEE", (gx0 - 4, gy0 - 10), 0.4, (200, 200, 200))
    cv2.rectangle(frame, (gx0, gy0), (gx1, gy1), (90, 90, 90), 1)
    to_y = lambda a: int(gy0 + (np.clip(a, 60, 180) - 60) / 120 * (gy1 - gy0))   # 60=top, 180=bottom
    ty = to_y(cfg.depth_ok_angle)
    cv2.line(frame, (gx0 - 4, ty), (gx1 + 4, ty), GREEN, 2)
    if counter.knee_angle is not None:
        cy = to_y(counter.knee_angle)
        cv2.rectangle(frame, (gx0 + 2, cy), (gx1 - 2, gy1 - 1), AMBER if counter.knee_angle > cfg.depth_ok_angle else GREEN, -1)

    # feedback banner
    if message:
        bad = message != "Good rep!"
        _panel(frame, 0, h - 52, w, h, 0.65)
        (tw, _), _ = cv2.getTextSize(message, FONT, 0.95, 2)
        _text(frame, message, ((w - tw) // 2, h - 17), 0.95, AMBER if bad else GREEN, 2)

    if lm is None:
        _text(frame, "No person detected - step into frame", (w // 2 - 190, h // 2), 0.7, RED, 2)
    return frame
