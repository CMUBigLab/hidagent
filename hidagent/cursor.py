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

from .hidagent import (_get_usb_capture, check_device_change, get_screen_rect,
                       send_serial_message)

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
# Nudge speeds as (units per report, delay, px per unit). Each is linear, but
# iOS acceleration gives each its own gain. FINE handles the last few px;
# FAST (5x quicker) is unreliable for very short hops, so it's only used for
# hops over NUDGE_FAST_PX.
NUDGE_FINE = (1, 0.02, 0.30)
NUDGE_FAST = (3, 0.01, 0.41)
NUDGE_FAST_PX = 12.0
NUDGE_MAX_PX = 100.0      # farther targets get an absolute move first
NUDGE_INIT_PX = 36.0      # nudge used to take the cursor-free background
                          # (clear of the pointer's own diameter)


def hid_move(hx, hy):
    """Firmware move: jump to the top-left corner, then (hx, hy) HID units."""
    return send_serial_message(json.dumps(
        {"type": "move", "x": int(hx), "y": int(hy)}) + "\n")


def hid_click(hx, hy):
    """Firmware click: same path as hid_move(hx, hy), then a left click."""
    return send_serial_message(json.dumps(
        {"type": "click", "x": int(hx), "y": int(hy)}) + "\n")


def hid_nudge(dx, dy, click=False, speed=NUDGE_FINE):
    """Firmware nudge: move (dx, dy) HID units from where the pointer is,
    in steps of speed[0] units every speed[1] s, then optionally click."""
    return send_serial_message(json.dumps(
        {"type": "nudgeclick" if click else "nudge", "x": int(dx),
         "y": int(dy), "step": speed[0], "delay": speed[1]}) + "\n")


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
        # Click point relative to the detected center: (0, 0) for the iOS
        # disc, the tip for an arrow pointer (Android). Set by
        # init_background_nudge().
        self.hotspot = np.zeros(2)
        self._tips = []         # recent arrow-tip measurements
        self._votes = {"arrow": 0, "round": 0}  # pointer shapes seen at init
        self.init_gain = None   # px per unit seen by the last init nudge
        self.init_from = None   # and where the pointer was before it
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
            moving = np.abs(cur - prev).max(axis=2) > 14
            if self.volatile is not None:
                moving &= self.volatile < 0.5  # e.g. a running stopwatch
            # Capture noise flickers single pixels along sharp edges in
            # every frame; real motion changes solid blobs.
            moving = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_OPEN,
                                      np.ones((2, 2), np.uint8))
            changed = int(moving.sum())
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

    def _mask(self, d):
        return d if self.volatile is None else d * (self.volatile < 0.5)

    def _learn_still(self, frame):
        """Grab another frame without moving the pointer: whatever changed
        is animating on its own (a running timer, a video) and is ignored
        from now on."""
        again = self.grab()
        changed = (self._diff(frame, again) > VOLATILE_THRESH).astype(np.uint8)
        changed = cv2.dilate(changed, np.ones((7, 7), np.uint8)).astype(np.float32)
        if changed.any():
            base = np.zeros_like(changed) if self._vol_count is None else self._vol_count
            self._vol_count = np.maximum(base, changed * 3 * VOLATILE_MIN_COUNT)
            self.volatile = (self._vol_count >= VOLATILE_MIN_COUNT).astype(np.float32)
        return again

    def _patch(self, dst, src, x, y, pad=5):
        r = int(math.ceil(self.radius)) + pad
        H, W = dst.shape[:2]
        x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
        dst[y0:y1, x0:x1] = src[y0:y1, x0:x1]

    def init_background(self, a=(100, 100), b=(300, 600), attempts=3):
        """Build a cursor-free background from two well-separated landings."""
        self.volatile = self._vol_count = None  # new scene
        for _ in range(attempts):
            hid_move(*a)
            fa = self._learn_still(self.grab())
            hid_move(*b)
            fb = self.grab()
            pk = self._peaks(self._score_map(self._mask(self._diff(fa, fb))), k=2)
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

    def init_background_nudge(self, toward=None, attempts=2, gain=NUDGE_FAST[2],
                              gain_known=False):
        """Like init_background(), but by nudging the pointer from wherever
        it is instead of sending it to the corner twice. Needs the nudge
        firmware. toward: optional (x, y) to nudge toward (e.g. the next
        target), so the move also makes progress. gain: expected px per unit
        at NUDGE_FAST; the actual gain is measured (self.init_gain). Unless
        gain_known, any plausible gain is accepted (new device), which a
        highlight hopping between menu items under a hidden iOS pointer can
        also pass. Returns the new position, or None if the pointer couldn't
        be seen moving (old firmware, or the pointer is hidden)."""
        W, H = self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]
        here = self.pos if self.pos is not None else (W / 2, H / 2)
        goal = toward if toward is not None else (W / 2, H / 2)
        v = np.array(goal, float) - here
        if np.hypot(*v) < 1:
            v = np.array([W / 2 - here[0] or 1.0, 0.0])
        v = v / np.hypot(*v)
        self.volatile = self._vol_count = None  # new scene
        units = NUDGE_INIT_PX / gain
        for i in range(attempts):
            fa = self._learn_still(self.grab())
            n = np.round(v * units * (-1 if i % 2 else 1)).astype(int)
            hid_nudge(*n, speed=NUDGE_FAST)
            fb = self.grab()
            d = self._mask(self._diff(fa, fb))
            allpk = self._peaks(self._score_map(d), k=4)
            pk = [p for p in allpk if p[2] >= MIN_SCORE_NEAR]
            if not pk:
                continue
            if self._large_change(d, pk[:2]) > SCENE_CHANGE_FRAC:
                continue  # still animating
            # The two spots lie along the nudge, old one behind, which tells
            # old from new on light and dark backgrounds alike and doesn't
            # assume the gain (iOS ~0.4 px/unit, Android ~1). Only the
            # stronger one has to pass MIN_SCORE: the pointer is faint over
            # e.g. a tab bar. Over a dark background an arrow pointer scores
            # below MIN_SCORE; a pair that stands far above everything else
            # is accepted anyway.
            un = n / np.hypot(*n)
            lo, hi = (0.6 * gain, 1.6 * gain) if gain_known else (0.15, 4.0)
            pair = None
            for q in pk[1:]:
                step = np.array(q[:2]) - pk[0][:2]
                dist = float(np.hypot(*step))
                if not lo <= dist / np.hypot(*n) <= hi:
                    continue
                cos = float(step @ un) / dist
                if cos > 0.9:
                    pair = (pk[0], q)
                elif cos < -0.9:
                    pair = (q, pk[0])
                if pair:
                    break
            if pair is None:
                continue
            rest = [p[2] for p in allpk if p not in pair]
            if pk[0][2] < MIN_SCORE and rest and \
                    max(rest) * 3 > min(pair[0][2], pair[1][2]):
                continue
            old, new = pair
            # Where the pointer was before this init (only if the first
            # attempt worked; a failed one may have moved it).
            self.init_from = (old[0], old[1]) if i == 0 else None
            self.init_gain = math.hypot(new[0] - old[0], new[1] - old[1]) / np.hypot(*n)
            self.hotspot = self._find_hotspot(fa, fb, new)
            self._last_d = None
            self.bg = fb.copy()
            self._patch(self.bg, fa, new[0], new[1])
            self.pos = (new[0], new[1])
            return self.pos
        return None

    def _pointer_blob(self, d, p):
        """Mask of the pointer around p in a diff image, as (xs, ys) in d's
        coordinates, or None."""
        r = int(3 * self.radius)
        H, W = d.shape[:2]
        x0, y0 = max(0, int(p[0]) - r), max(0, int(p[1]) - r)
        x1, y1 = min(W, int(p[0]) + r + 1), min(H, int(p[1]) + r + 1)
        m = (d[y0:y1, x0:x1] > 30).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((3, 3), np.uint8)))
        cx, cy = int(p[0]) - x0, int(p[1]) - y0
        if n < 2 or not 0 <= cy < lab.shape[0] or not 0 <= cx < lab.shape[1] \
                or lab[cy, cx] == 0:
            return None
        ys, xs = np.where((lab == lab[cy, cx]) & (m > 0))
        return (xs + x0, ys + y0) if len(xs) else None

    def current_hotspot(self):
        """Hotspot for the pointer as last located. Android swaps the arrow
        for an I-beam over text, whose hotspot is its center."""
        if not self.hotspot.any() or self.pos is None or \
                getattr(self, "_last_d", None) is None:
            return self.hotspot
        blob = self._pointer_blob(self._last_d, self.pos)
        if blob is not None:
            w = blob[0].max() - blob[0].min() + 1
            h = blob[1].max() - blob[1].min() + 1
            if h > 2.5 * w:
                return np.zeros(2)
        return self.hotspot

    def _find_hotspot(self, before, after, p):
        """Click point relative to the detected center p: the tip of an
        arrow pointer, else the center. An arrow's topmost pixel is at the
        left of its bounding box (a disc's is in the middle) and it fills
        about half the box (a disc ~0.79, iOS's button highlight ~0.9).
        Single measurements scatter by a few px with the background, so the
        result is the median of recent ones. Some iOS shapes (a menu
        highlight, the AssistiveTouch button) can pass for an arrow once, so
        the tip is only used after 3+ arrow sightings that outnumber round
        ones."""
        blob = self._pointer_blob(self._diff(before, after), p)
        if blob is None:
            return self.hotspot
        xs, ys = blob
        bx, by = xs.min(), ys.min()
        bw, bh = xs.max() - bx + 1, ys.max() - by + 1
        tip = (xs[ys == by].min(), by)
        fill = len(xs) / float(bw * bh)
        if bh > 2.5 * bw:
            return self.hotspot  # Android's I-beam: says nothing either way
        if tip[0] - bx > 0.25 * bw or fill > 0.65:
            self._votes["round"] += 1
        else:
            self._votes["arrow"] += 1
            self._tips = (self._tips + [(tip[0] - p[0], tip[1] - p[1])])[-9:]
        v = self._votes
        if v["arrow"] >= 3 and v["arrow"] > v["round"]:
            return np.median(np.array(self._tips, float), axis=0)
        return np.zeros(2)

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
        d = self._mask(d_raw)
        self._last_d = d
        pk = self._peaks(self._score_map(d))
        strong = MIN_SCORE
        if pk and pk[0][2] >= MIN_SCORE_NEAR and (len(pk) < 2 or pk[0][2] >= 3 * pk[1][2]):
            # A lone clear peak, e.g. an arrow pointer on a dark background.
            strong = min(strong, pk[0][2])
        if expected is None:
            pk = [p for p in pk if p[2] >= strong]
        else:
            dist = lambda p: math.hypot(p[0] - expected[0], p[1] - expected[1])
            pk = sorted((p for p in pk if dist(p) <= search_radius and (
                p[2] >= strong or (p[2] >= MIN_SCORE_NEAR and dist(p) <= NEAR_RADIUS))),
                key=dist)
        self.last_peaks = pk
        found = pk[0][:2] if pk else None
        self.scene_change = self._large_change(d, [self.pos, found])
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
        pts = [p for p in keep_out if p is not None]

        def hover(s):
            # A control highlighting under the pointer (Android hover
            # states) surrounds the pointer; that's not a new screen.
            return any(s[0] <= p[0] < s[0] + s[2] and s[1] <= p[1] < s[1] + s[3]
                       for p in pts)
        return sum(int(s[4]) for s in stats[1:]
                   if s[4] > big and not hover(s)) / float(H * W)


class CursorPositioner:
    """Steers the cursor onto screenshot coordinates by watching it."""

    def __init__(self, tracker=None):
        self.t = tracker or CursorTracker()
        x0, y0, x1, y1 = self.t.rect
        self._key = f"{x1 - x0}x{y1 - y0}"
        self.obs = self._load()  # [(hx, hy, px, py)]
        # Whether the firmware has "nudge" (None until seen working or not;
        # older firmware silently ignores it) and its px per HID unit.
        self.nudge_ok = None
        self.nudge_gain = {s: np.array([s[2]] * 2) for s in (NUDGE_FINE, NUDGE_FAST)}
        self._gain_seeded = False
        if self._saved_gain is not None:
            self._seed_gain(self._saved_gain)

    def _seed_gain(self, g):
        """Replace the iOS default gains with a measured FAST gain."""
        self._gain_seeded = True
        self.nudge_gain[NUDGE_FAST][:] = g
        # FINE is slower, so iOS accelerates it less. Assuming the same ratio
        # elsewhere at worst undershoots, which the nudge loop corrects.
        self.nudge_gain[NUDGE_FINE][:] = g * NUDGE_FINE[2] / NUDGE_FAST[2]

    def _init_nudge(self, toward):
        """init_background_nudge(), and on the first success replace the
        iOS default gains with the one it measured (Android's is ~2.5x)."""
        pos = self.t.init_background_nudge(
            toward=toward, gain=float(self.nudge_gain[NUDGE_FAST].mean()),
            gain_known=self._gain_seeded)
        g = self.t.init_gain
        if pos is not None and g is not None and not self._gain_seeded:
            self._seed_gain(g)
        return pos

    # ---------------------------------------------------------- persistence
    def _load(self):
        self._saved_gain = None
        if not os.path.exists(_MAP_FILE):
            return []
        with open(_MAP_FILE) as f:
            data = json.load(f)
        self._saved_gain = data.get(self._key + "/gain")
        # The arrow-tip offset is hard to measure on dark screens, so keep
        # the one measured earlier on this device.
        hs = data.get(self._key + "/hotspot")
        if hs is not None:
            self.t.hotspot = np.array(hs, float)
            self.t._tips = [tuple(hs)] * 3
            self.t._votes["arrow"] = 3
        return [tuple(o) for o in data.get(self._key, [])]

    def save(self, max_obs=2000):
        data = {}
        if os.path.exists(_MAP_FILE):
            with open(_MAP_FILE) as f:
                data = json.load(f)
        data[self._key] = [list(o) for o in self.obs[-max_obs:]]
        if self.t.hotspot.any():
            data[self._key + "/hotspot"] = [float(v) for v in self.t.hotspot]
        else:
            data.pop(self._key + "/hotspot", None)
        if self._gain_seeded:
            data[self._key + "/gain"] = float(self.nudge_gain[NUDGE_FAST].mean())
        with open(_MAP_FILE, "w") as f:
            json.dump(data, f)

    # ---------------------------------------------------------------- model
    def _unpinned(self, o):
        """Drop landings pinned against a screen edge.

        iOS stops the pointer at the edges, so every command past the edge
        lands on the same line; those landings carry no slope information
        and would wreck a local fit.
        """
        if len(o) < 10:
            return o
        keep = np.ones(len(o), bool)
        for col in (2, 3):
            lo, hi = o[:, col].min(), o[:, col].max()
            keep &= (o[:, col] > lo + 1.0) & (o[:, col] < hi - 1.0)
        return o[keep]

    def _fit(self, key, by, k=8, pts=None):
        """Locally weighted affine map from the k observations nearest key."""
        o = np.array(self.obs if pts is None else pts, float)
        if pts is None:
            o = self._unpinned(o)
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

        Returns {"cmd", "pos", "err", "iters", "verified", "snapped",
        "nudged"}, or None if the cursor was never found.

        With the nudge firmware, the cursor gets there with at most one
        absolute move (from the corner) and then short relative nudges, and
        is left exactly at pos; nudged is True and cmd is None, since the
        spot isn't reachable by a single absolute command. Otherwise every
        try is an absolute move and cmd is the command that lands on pos.

        verified is False when the pointer was invisible at the final spot
        (e.g. over a black icon) and pos was estimated from the visible
        landings around it. snapped is True when iOS jumps the resting
        pointer across the target (it snaps onto nearby app icons), so the
        exact pixel can't be reached; pos is then the closest reachable
        landing.
        """
        if self.nudge_ok is not False:
            result = self._goto_nudge(np.array([tx, ty], float), tol, max_iter)
            if result is not None:
                return result
        # Absolute moves steer the detected center; aim it so the hotspot
        # (an arrow's tip) lands on the target.
        hs = self.t.hotspot
        result = self._goto_absolute(*self.aim(tx, ty), tol, max_iter)
        if result is not None:
            result["nudged"] = False
            result["pos"] = (result["pos"][0] + hs[0], result["pos"][1] + hs[1])
        return result

    def command_for(self, tx, ty, tol=1.0, max_err=5.0):
        """Absolute HID command that lands on (tx, ty).

        For firmware commands that start from the corner (drags). Steers
        with absolute moves only; near the screen edge (where the pointer
        stops short, e.g. edge swipes) or if the pointer can't be seen, uses
        the learned map instead. The pointer is left on the target.
        """
        W, H = self.t.rect[2] - self.t.rect[0], self.t.rect[3] - self.t.rect[1]
        if min(tx, ty, W - tx, H - ty) > 10:
            r = self._goto_absolute(*self.aim(tx, ty), tol)
            if r is not None and r["err"] <= max_err:
                return r["cmd"]
        return tuple(int(c) for c in np.round(self.predict_cmd(self.aim(tx, ty))))

    def mapped(self, tx, ty):
        """Whether there are landings to predict commands near (tx, ty)."""
        return self._fit((tx, ty), "pos") is not None

    def aim(self, tx, ty):
        """Where the detected center must go for the hotspot to hit (tx, ty);
        absolute commands and the learned map work in centers."""
        return (tx - self.t.hotspot[0], ty - self.t.hotspot[1])

    def _goto_nudge(self, target, tol, max_iter):
        """goto() via nudges; None if nudging isn't possible here."""
        t = self.t
        it = 0
        pos = None
        # Until there are landings to fit, an absolute move is a guess at
        # the scale (iOS's on a new device) and can bury the pointer in a
        # corner; nudge the whole way instead.
        mapped = self.mapped(*target)
        near = t.pos is not None and (
            not mapped or math.hypot(*(target - t.pos)) <= NUDGE_MAX_PX)
        if t.bg is not None and near:
            # Also notices if the screen changed since (then t.bg is None).
            pos = t.locate(expected=t.pos, search_radius=15)
        cmd = None
        if pos is None and not near:
            # One absolute move to a spot a nudge away from the target (on
            # the screen-center side). If the screen changed, the nudge onto
            # the target rebuilds the background; no trips to the corner.
            W, H = t.rect[2] - t.rect[0], t.rect[3] - t.rect[1]
            v = np.array([W / 2, H / 2]) - target
            v = v / np.hypot(*v) if np.hypot(*v) > 1 else np.array([1.0, 0])
            start = target + v * NUDGE_INIT_PX
            if not mapped:
                # New device, scale unknown: aim well short so the pointer
                # stays on screen (short means toward the top-left corner,
                # where an arrow pointer is still visible).
                start = target * 0.6
            it += 1
            cmd = tuple(int(c) for c in np.round(self.predict_cmd(start)))
            hid_move(*cmd)
            t.pos = tuple(start)
            if t.bg is not None:
                pos = self._locate_after(cmd)
                if pos is not None:
                    self.obs.append((*cmd, float(pos[0]), float(pos[1])))
        if pos is None:
            # Lost (the absolute move went somewhere unexpected, e.g. on a
            # new device before the map has landings) or the screen changed:
            # find the pointer by nudging it from wherever it is.
            lost = t.bg is not None
            pos = self._init_nudge(target)
            if pos is not None and cmd is not None and t.init_from is not None:
                # The init nudge saw where the absolute move had landed.
                self.obs.append((*cmd, *map(float, t.init_from)))
            if pos is None and lost:
                # Possibly against an edge or in a corner, where an arrow
                # pointer is mostly off screen: bring it to the middle.
                W, H = t.rect[2] - t.rect[0], t.rect[3] - t.rect[1]
                hid_move(*np.round(self.predict_cmd((W / 2, H / 2))).astype(int))
                t.pos = (W / 2, H / 2)
                it += 1
                pos = self._init_nudge(target)
            if pos is None:
                # Couldn't see the pointer move. That's either old firmware
                # (no nudge) or a hidden pointer; only the first is permanent.
                if self.nudge_ok is None and not lost:
                    self._nudge_fails = getattr(self, "_nudge_fails", 0) + 1
                    if self._nudge_fails >= 2:
                        self.nudge_ok = False
                return None
            self.nudge_ok = True
        best_err, stalls, verified = None, 0, True
        while True:
            hs = t.current_hotspot() if verified else t.hotspot
            err = target - np.array(pos) - hs
            dist = float(np.hypot(*err))
            if best_err is None or dist < best_err - 0.2:
                stalls = 0
            else:
                stalls += 1
            best_err = dist if best_err is None else min(best_err, dist)
            if dist <= tol or stalls >= 2 or it >= max_iter or not verified:
                break
            speed = NUDGE_FAST if dist > NUDGE_FAST_PX else NUDGE_FINE
            gain = self.nudge_gain[speed]
            n = np.round(err / gain).astype(int)
            if not n.any():
                n[np.argmax(np.abs(err))] = int(np.sign(err[np.argmax(np.abs(err))]))
            it += 1
            hid_nudge(*n, speed=speed)
            exp = np.array(pos) + n * gain
            hop = float(np.hypot(*(n * gain)))
            new = t.locate(expected=exp, search_radius=12 + 0.25 * hop)
            if new is None:
                if t.bg is None:  # scene changed: start over from here
                    return None if self._init_nudge(target) is None \
                        else self._goto_nudge(target, tol, max_iter - it)
                # Invisible here (e.g. over a black icon): nudges are
                # near-linear over short hops, so estimate where it went.
                pos, verified = tuple(exp), False
                t.pos = pos
                continue
            moved = np.array(new) - np.array(pos)
            if self.nudge_ok is None:
                if np.abs(moved).max() > 0.5:
                    self.nudge_ok = True
                elif np.abs(n).max() >= 3:
                    self.nudge_ok = False  # old firmware ignored the nudge
                    return None
            for a in (0, 1):
                if abs(n[a]) >= 10 and moved[a] / n[a] > 0.1:
                    gain[a] = 0.7 * gain[a] + 0.3 * moved[a] / n[a]
            pos = new
        self.save()
        # The pointer stays where it last landed, so report that landing.
        pos = np.array(pos) + hs
        return {"cmd": None, "pos": tuple(float(v) for v in pos),
                "err": float(np.hypot(*(target - pos))),
                "verified": verified, "iters": it, "snapped": False,
                "nudged": True}

    def _goto_absolute(self, tx, ty, tol=1.0, max_iter=8):
        target = np.array([tx, ty], float)
        cmd = np.round(self.predict_cmd(target)).astype(int)
        lo, hi = [None, None], [None, None]  # per-axis bracket on the command
        here, tried = [], {}
        best, misses, stalls, it = None, 0, 0, 0
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
                # landings (this target's included) and keep steering --
                # unless those landings show a snap, which no fit can bridge.
                pred = self.predict_pos(key)
                if pred is None or misses > 3 or _has_jump(here):
                    break
                pos = (float(pred[0]), float(pred[1]))
            else:
                here.append((*key, *pos))
            err = target - np.array(pos)
            dist = float(np.hypot(*err))
            if best is None or dist < best["err"] - 0.2:
                stalls = 0
            else:
                stalls += 1
            if best is None or dist < best["err"]:
                best = {"cmd": key, "pos": pos, "err": dist, "verified": verified}
            if dist <= tol or stalls >= 2:
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
        best["snapped"] = best["err"] > tol and _has_jump(here)
        self.save()
        return best


def _has_jump(landings, max_px_per_unit=4.0):
    """True if commands a few units apart landed implausibly far apart.

    Normal landings move at most ~2 px per HID unit; a bigger jump between
    nearby commands means iOS snapped the pointer (e.g. onto an app icon).
    """
    o = np.array(landings, float)
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            dc = np.abs(o[i, :2] - o[j, :2]).max()
            if 0 < dc <= 3 and np.abs(o[i, 2:] - o[j, 2:]).max() / dc > max_px_per_unit:
                return True
    return False


_positioner = None


def get_positioner():
    """Shared CursorPositioner, rebuilt when the target device changes."""
    global _positioner
    check_device_change()
    if _positioner is None:
        _positioner = CursorPositioner()
    return _positioner
