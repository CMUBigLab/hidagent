"""Visual cursor positioning.

Instead of converting screen pixels to HID units with a fixed calibration
(which drifts with pointer acceleration and differs per device), find the
pointer in the HDMI capture and steer it onto the target:

1. Every firmware "move" starts from the top-left corner, and the same
   command lands on the same pixel every time (measured within ~1 px).
2. CursorTracker finds the pointer by comparing each frame with a
   cursor-free background using a disc-shaped matched filter, so it works
   over icons, text and screen edges, and ignores pixels that keep changing
   on their own (e.g. the Clock icon's hands).
3. CursorPositioner tries a command, sees where the cursor landed, and
   corrects with a bracketed Newton search until it is within a fraction of
   a pixel. The winning command is then reused for the click.

Coordinates are in screenshot pixels (the image returned by get_screenshot).
Every landing is remembered in cursor_map.json, keyed by screen size, so
later moves usually start very close to the right command.
"""
import json
import math
import os
import time

import cv2
import numpy as np

from .hidagent import _get_usb_capture, get_screen_rect, send_serial_message

_MAP_FILE = os.path.join(os.path.dirname(__file__), "cursor_map.json")

NOISE = 6.0               # per-pixel color-diff noise floor to subtract
CLIP = 60.0               # cap per-pixel diff so sharp edges don't dominate
SETTLE_PX = 25            # frame is settled when fewer pixels change than this
SETTLE_TIMEOUT = 0.8      # s; iOS animates the pointer for ~100-150 ms
MIN_SCORE = 24.0          # real cursor scores 30-60; clock-hand noise 14-22
MIN_SCORE_NEAR = 12.0     # ...but over dark icons it can score ~15-28, so
NEAR_RADIUS = 8.0         # accept weaker peaks right where we expected it
VOLATILE_DECAY = 0.8      # per background update
VOLATILE_THRESH = 15.0    # bg-to-bg change (after noise floor) that counts
VOLATILE_MIN_COUNT = 1.6  # ~2 recent changes before a pixel is ignored
SCENE_CHANGE_FRAC = 0.01  # large non-cursor change over this -> rebuild bg
DEFAULT_RADIUS = 10.7     # iOS pointer radius in screenshot pixels
DEFAULT_PX_PER_HID = 0.764


def hid_move(hx, hy):
    """Firmware move: jump to the top-left corner, then (hx, hy) HID units."""
    return send_serial_message(json.dumps(
        {"type": "move", "x": int(hx), "y": int(hy)}) + "\n")


def hid_click(hx, hy):
    """Firmware click: same path as hid_move(hx, hy), then a left click."""
    return send_serial_message(json.dumps(
        {"type": "click", "x": int(hx), "y": int(hy)}) + "\n")


def _disc(r):
    k = int(math.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (np.hypot(xx, yy) <= r).astype(np.float32)


def _ring(r_in, r_out):
    k = int(math.ceil(r_out))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    d = np.hypot(xx, yy)
    return ((d > r_in) & (d <= r_out)).astype(np.float32)


class CursorTracker:
    """Finds the pointer in settled capture frames."""

    def __init__(self, radius=DEFAULT_RADIUS):
        self.cap = _get_usb_capture()
        self.rect = get_screen_rect()
        self.radius = radius
        self.bg = None          # cursor-free screen image (float32 BGR)
        self.pos = None
        self.volatile = None    # 1 where pixels keep changing on their own
        self._vol_count = None
        self.scene_change = 0.0
        d, r = _disc(radius), _ring(radius + 1.5, radius + 5)
        self._disc, self._ring = d / d.sum(), r / r.sum()

    # ---------------------------------------------------------------- frames
    def _read(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("capture read failed")
        return f

    def _crop(self, f):
        x0, y0, x1, y1 = self.rect
        return f[y0:y1, x0:x1].astype(np.float32)

    def grab(self):
        """Return the first frame after the screen stops changing."""
        prev = self._crop(self._read())
        t0 = time.perf_counter()
        while True:
            cur = self._crop(self._read())
            changed = int((np.abs(cur - prev).max(axis=2) > 14).sum())
            if changed < SETTLE_PX or time.perf_counter() - t0 > SETTLE_TIMEOUT:
                return cur
            prev = cur

    # ------------------------------------------------------------- detection
    def _diff(self, a, b):
        d = np.sqrt(((b - a) ** 2).sum(axis=2))
        return np.clip(d - NOISE, 0, CLIP)

    def _score_map(self, d):
        """Mean change inside a pointer-sized disc minus a ring around it."""
        s_in = cv2.filter2D(d, -1, self._disc, borderType=cv2.BORDER_CONSTANT)
        s_out = cv2.filter2D(d, -1, self._ring, borderType=cv2.BORDER_CONSTANT)
        return s_in - s_out

    def _peaks(self, score, k=4):
        s = score.copy()
        sep = int(2 * self.radius)
        out = []
        for _ in range(k):
            y, x = np.unravel_index(np.argmax(s), s.shape)
            if s[y, x] <= 0:
                break
            # Parabolic sub-pixel refinement.
            fx = fy = 0.0
            if 0 < x < s.shape[1] - 1:
                l, c, r = score[y, x - 1], score[y, x], score[y, x + 1]
                den = l - 2 * c + r
                fx = 0.5 * (l - r) / den if den else 0.0
            if 0 < y < s.shape[0] - 1:
                u, c, dn = score[y - 1, x], score[y, x], score[y + 1, x]
                den = u - 2 * c + dn
                fy = 0.5 * (u - dn) / den if den else 0.0
            out.append((x + fx, y + fy, float(score[y, x])))
            s[max(0, y - sep):y + sep + 1, max(0, x - sep):x + sep + 1] = -1e9
        return out

    def _patch(self, dst, src, x, y, pad=5):
        r = int(math.ceil(self.radius)) + pad
        H, W = dst.shape[:2]
        x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
        dst[y0:y1, x0:x1] = src[y0:y1, x0:x1]

    def init_background(self, a=(100, 100), b=(300, 600), attempts=3):
        """Build a cursor-free background from two well-separated landings."""
        for _ in range(attempts):
            hid_move(*a)
            fa = self.grab()
            hid_move(*b)
            fb = self.grab()
            pk = self._peaks(self._score_map(self._diff(fa, fb)), k=2)
            if len(pk) == 2 and pk[1][2] >= MIN_SCORE:
                break
        else:
            raise RuntimeError(f"could not find the cursor to start tracking: {pk}")

        def lum_change(p):
            x, y = int(round(p[0])), int(round(p[1]))
            r = int(self.radius * 0.6)
            sl = (slice(max(0, y - r), y + r + 1), slice(max(0, x - r), x + r + 1))
            return float((fb[sl] - fa[sl]).mean())

        # The translucent pointer darkens these (light) spots, so the peak
        # that got darker is where it is now.
        new, _ = sorted(pk, key=lum_change)
        self.bg = fb.copy()
        self._patch(self.bg, fa, new[0], new[1])
        self.pos = (new[0], new[1])
        return self.pos

    def locate(self, expected=None, search_radius=60):
        """Find the cursor in a newly settled frame.

        expected: optional predicted (x, y); peaks farther than
        search_radius are ignored and the nearest acceptable one wins.
        Returns (x, y) or None. If the screen changed a lot (notification
        banner, app transition) the background is dropped (self.bg = None)
        and the caller should call init_background() and retry.
        """
        frame = self.grab()
        d_raw = self._diff(self.bg, frame)
        d = d_raw if self.volatile is None else d_raw * (self.volatile < 0.5)
        pk = self._peaks(self._score_map(d))
        if expected is None:
            pk = [p for p in pk if p[2] >= MIN_SCORE]
        else:
            dist = lambda p: math.hypot(p[0] - expected[0], p[1] - expected[1])
            pk = sorted((p for p in pk if dist(p) <= search_radius and (
                p[2] >= MIN_SCORE or (p[2] >= MIN_SCORE_NEAR and dist(p) <= NEAR_RADIUS))),
                key=dist)
        self.last_peaks = pk
        found = pk[0][:2] if pk else None
        self.scene_change = self._large_change(d_raw, [self.pos, found])
        if self.scene_change > SCENE_CHANGE_FRAC:
            self.bg = None
            return None
        if found is None:
            return None
        new_bg = frame.copy()
        self._patch(new_bg, self.bg, *found)
        # Both backgrounds are cursor-free, so whatever differs between them
        # changed on its own (clock hands, status bar).
        self._learn_volatile(self._diff(self.bg, new_bg))
        self.bg = new_bg
        self.pos = found
        return found

    def _learn_volatile(self, bg_d):
        # Capture noise at sharp edges flickers randomly, so a pixel only
        # counts once it has changed in repeated updates.
        changed = (bg_d > VOLATILE_THRESH).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        changed = cv2.dilate(changed, np.ones((3, 3), np.uint8)).astype(np.float32)
        if self._vol_count is None:
            self._vol_count = changed
        else:
            self._vol_count = self._vol_count * VOLATILE_DECAY + changed
        self.volatile = (self._vol_count >= VOLATILE_MIN_COUNT).astype(np.float32)

    def _large_change(self, d, keep_out):
        """Fraction of the screen covered by large non-cursor changes."""
        changed = (d > 8).astype(np.uint8)
        r = int(math.ceil(self.radius)) + 6
        H, W = changed.shape
        for p in keep_out:
            if p is not None:
                x, y = int(p[0]), int(p[1])
                changed[max(0, y - r):min(H, y + r + 1),
                        max(0, x - r):min(W, x + r + 1)] = 0
        changed = cv2.dilate(changed, np.ones((5, 5), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        big = 6 * math.pi * self.radius ** 2
        return sum(int(s[4]) for s in stats[1:] if s[4] > big) / float(H * W)


class CursorPositioner:
    """Steers the cursor onto screenshot coordinates by watching it."""

    def __init__(self, tracker=None):
        self.t = tracker or CursorTracker()
        x0, y0, x1, y1 = self.t.rect
        self._key = f"{x1 - x0}x{y1 - y0}"
        self.obs = self._load()  # [(hx, hy, px, py)]

    # ---------------------------------------------------------- persistence
    def _load(self):
        if not os.path.exists(_MAP_FILE):
            return []
        with open(_MAP_FILE) as f:
            return [tuple(o) for o in json.load(f).get(self._key, [])]

    def save(self, max_obs=2000):
        data = {}
        if os.path.exists(_MAP_FILE):
            with open(_MAP_FILE) as f:
                data = json.load(f)
        data[self._key] = [list(o) for o in self.obs[-max_obs:]]
        with open(_MAP_FILE, "w") as f:
            json.dump(data, f)

    # ---------------------------------------------------------------- model
    def _fit(self, key, by, k=8, pts=None):
        """Locally weighted affine map from the k observations nearest key."""
        o = np.array(self.obs if pts is None else pts, float)
        if len(o) < 3:
            return None
        s, t = ([2, 3], [0, 1]) if by == "pos" else ([0, 1], [2, 3])
        d = np.hypot(o[:, s[0]] - key[0], o[:, s[1]] - key[1])
        idx = np.argsort(d)[:k]
        X = np.c_[o[idx][:, s], np.ones(len(idx))]
        if np.linalg.matrix_rank(X) < 3:
            return None
        w = 1.0 / (1.0 + d[idx] / 20.0)
        coef, *_ = np.linalg.lstsq(X * w[:, None], o[idx][:, t] * w[:, None],
                                   rcond=None)
        return coef

    def predict_cmd(self, target):
        coef = self._fit(target, "pos")
        if coef is None:
            return np.array(target, float) / DEFAULT_PX_PER_HID
        return np.array([*target, 1.0]) @ coef

    def predict_pos(self, cmd):
        coef = self._fit(cmd, "cmd")
        return None if coef is None else np.array([*cmd, 1.0]) @ coef

    def _jacobian(self, cmd, here):
        """d(landing pixel)/d(command) near cmd, as a 2x2 matrix A[pos][cmd].

        The firmware moves both axes together until the shorter one is done,
        then finishes the longer ("dominant") axis alone at constant speed.
        So extra units on the dominant axis land exactly DEFAULT_PX_PER_HID
        further along it and don't move the other axis; only the other
        column (diagonal phase, accelerated and coupled) is estimated from
        nearby landings.
        """
        A = np.eye(2) * DEFAULT_PX_PER_HID
        coef = self._fit(cmd, "cmd", pts=here) if len(here) >= 3 else None
        if coef is None:
            coef = self._fit(cmd, "cmd")
        if coef is not None:
            A = coef[:2].T.copy()
        mag = np.abs(np.asarray(cmd, float))
        if abs(mag[0] - mag[1]) > 15:
            D = int(np.argmax(mag))
            N = 1 - D
            A[D, D], A[N, D] = DEFAULT_PX_PER_HID, 0.0
            A[N, N] = np.clip(A[N, N], 0.5, 2.0)
        if abs(np.linalg.det(A)) < 0.1:
            A = np.eye(2) * DEFAULT_PX_PER_HID
        return A

    # ------------------------------------------------------------- hardware
    def _locate_after(self, cmd):
        exp = self.predict_pos(cmd)
        if exp is None:
            return self.t.locate()
        pos = self.t.locate(expected=exp, search_radius=60)
        if pos is None and self.t.bg is not None:
            pos = self.t.locate()
        return pos

    def try_cmd(self, cmd, rebuilds=2):
        """Send a move command and return where the cursor landed (or None)."""
        cmd = (int(cmd[0]), int(cmd[1]))
        if self.t.bg is None:
            self.t.init_background()
        hid_move(*cmd)
        pos = self._locate_after(cmd)
        while pos is None and self.t.bg is None and rebuilds > 0:
            # The screen changed under us: rebuild the background, redo.
            rebuilds -= 1
            self.t.init_background()
            hid_move(*cmd)
            pos = self._locate_after(cmd)
        if pos is not None:
            self.obs.append((cmd[0], cmd[1], float(pos[0]), float(pos[1])))
        return pos

    # -------------------------------------------------------------- control
    def goto(self, tx, ty, tol=1.0, max_iter=8):
        """Steer the cursor onto (tx, ty).

        Returns {"cmd", "pos", "err", "iters", "verified"} for the closest
        landing (the cursor is left there), or None if the cursor was never
        found. verified is False when the pointer was invisible at the final
        spot (e.g. over a black icon) and pos was estimated from the visible
        landings around it.
        """
        target = np.array([tx, ty], float)
        cmd = np.round(self.predict_cmd(target)).astype(int)
        lo, hi = [None, None], [None, None]  # per-axis bracket on the command
        here, tried = [], {}
        best, misses, it = None, 0, 0
        while it < max_iter:
            it += 1
            key = (int(cmd[0]), int(cmd[1]))
            pos = self.try_cmd(key)
            tried[key] = pos
            verified = pos is not None
            if pos is None:
                misses += 1
                if not here:
                    if misses > 3:
                        break
                    cmd = cmd + 2  # never seen yet: step off the blind spot
                    continue
                # The pointer is invisible here, but landings are smooth and
                # repeatable, so estimate where it went from nearby visible
                # landings (this target's included) and keep steering.
                pred = self.predict_pos(key)
                if pred is None or misses > 3:
                    break
                pos = (float(pred[0]), float(pred[1]))
            else:
                here.append((*key, *pos))
            err = target - np.array(pos)
            dist = float(np.hypot(*err))
            if best is None or dist < best["err"]:
                best = {"cmd": key, "pos": pos, "err": dist, "verified": verified}
            if dist <= tol:
                break
            # Landing position increases with the command on each axis, so
            # remember the tightest commands known to land short / long. The
            # axes are coupled, so a bound only holds while the other axis's
            # command is unchanged: store it as (value, other axis command).
            for a in (0, 1) if verified else ():
                if err[a] > 0 and (lo[a] is None or lo[a][1] != key[1 - a]
                                   or key[a] > lo[a][0]):
                    lo[a] = (key[a], key[1 - a])
                if err[a] < 0 and (hi[a] is None or hi[a][1] != key[1 - a]
                                   or key[a] < hi[a][0]):
                    hi[a] = (key[a], key[1 - a])
            new = cmd + np.linalg.solve(self._jacobian(key, here), err)
            new_r = np.round(new).astype(int)
            for a in (0, 1):
                other = new_r[1 - a]
                l = lo[a][0] if lo[a] is not None and lo[a][1] == other else -np.inf
                h = hi[a][0] if hi[a] is not None and hi[a][1] == other else np.inf
                if not l < new[a] < h:  # Newton left the bracket: bisect
                    new[a] = (l + h) / 2 if np.isfinite(l) and np.isfinite(h) \
                        else np.clip(new[a], l + 1, h - 1)
            new = np.round(new).astype(int)
            if tuple(new) in tried:
                # Integer commands: try the untried neighbour predicted closest.
                scored = []
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (-1, -1), (1, -1), (-1, 1)):
                    c = (key[0] + dx, key[1] + dy)
                    p = None if c in tried else self.predict_pos(c)
                    if p is not None:
                        scored.append((float(np.hypot(*(p - target))), c))
                if not scored or min(scored)[0] >= best["err"]:
                    break
                new = np.array(min(scored)[1])
            cmd = new
        if best is None:
            return None
        if tuple(int(c) for c in cmd) != best["cmd"]:
            hid_move(*best["cmd"])  # landings repeat, so this returns to best
            if best["verified"]:
                self.t.locate(expected=best["pos"], search_radius=15)
        best["iters"] = it
        self.save()
        return best


_positioner = None


def get_positioner():
    """Shared CursorPositioner (one per process)."""
    global _positioner
    if _positioner is None:
        _positioner = CursorPositioner()
    return _positioner
