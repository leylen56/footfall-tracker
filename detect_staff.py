"""
FootfallCam AI Evaluation - locate the STAFF (name-tag wearer) in a 3D-sensor clip
=================================================================================

Task 1 : which frames contain the staff member?
Task 2 : the staff (x, y) position in every frame where they are present.

Context / observations from the sample clip
------------------------------------------
* The sensor is a ceiling fish-eye ("3D sensor").  ~20 people are SEATED at
  desks; exactly one person WALKS through the scene - that walker is the staff
  member wearing the lanyard name tag (per the brief).
* The staff wears a two-tone black/white hoodie.  Seen from behind while walking
  they read as dark (black hood + shoulders, white chest logo - snapshot 1);
  seen from straight above while bending, the white lower-back panel and the
  white lanyard card dominate (snapshot 2).

Approach  (three explainable cues, fused)
----------------------------------------
1. MOTION            MOG2 background subtraction -> cleaned foreground -> blobs.
                     Seated people give small stationary blobs; the walker gives
                     a large blob whose centroid travels a long path.
2. SEED TRACKLET     Greedy centroid linking of the motion blobs; the tracklet
                     with the largest travel is the walker.  Its boxes seed ...
3. APPEARANCE + COAST A single-target tracker that follows that person across
                     the WHOLE clip: constant-velocity prediction + an HSV
                     colour-histogram appearance model, matched against the
                     motion blobs, with a mean-shift back-projection fallback
                     when the person stands still / partially occluded, and a
                     short "coast" so brief misses don't drop the track.
   NAME-TAG cue      Inside the tracked box we also look for the lanyard card
                     (small, bright, low-saturation, local brightness peak).
                     Reported per frame as extra evidence; it is NOT required
                     for "present" because the card is only intermittently
                     visible from directly overhead.

An optional YOLOv8 person-detector backend (`--backend yolo`) is included for
comparison; on this crowded fish-eye view it fragments badly (see SOLUTION.md),
which is why the motion+appearance tracker is the primary method.  The primary
path needs only OpenCV + NumPy, so it runs on any laptop / any test clip with
no model download.

Usage
-----
    python detect_staff.py --video sample.mp4 --out ../outputs
    python detect_staff.py --video other.mp4  --out ../outputs \
                           --seed-frame 350 --seed-box 360,470,120,190   # optional manual seed
    python detect_staff.py --video other.mp4  --out ../outputs --backend yolo

Outputs (in --out)
------------------
    staff_frames.csv   frame,time_s,present,x_px,y_px,x_norm,y_norm,x_floor,y_floor,
                       bbox_x,bbox_y,bbox_w,bbox_h,tag_score,source
    annotated.mp4      staff boxed green ("STAFF"), other movers grey, live trail
    timeline.png       presence bar + tag-score / speed traces
    trajectory.png     staff path over a reference frame, coloured by time
    trajectory_floor.png  de-warped floor path (bonus)
    tag_evidence.png   montage of the name-tag crops that fired
    summary.json       frame ranges, counts, parameters
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
import cv2


# --------------------------------------------------------------------------- #
#  Parameters - one dict so they are easy to show / tune during the interview #
# --------------------------------------------------------------------------- #
P = dict(
    # MOG2 background subtraction
    mog_history=400, mog_var_thresh=24, fg_thresh=200,
    # moving-blob filter (px)
    blob_min_area=500, blob_max_area=45000, blob_max_wh=300,
    # greedy tracklet linking (for the seed)
    link_max_dist=70, link_max_gap=12, track_min_len=8,
    seed_min_travel=120,          # seed tracklet must move at least this far
    # appearance + coast single-target tracker
    hs_bins=(24, 24),             # H,S histogram bins
    gate_sigma=45.0,              # px - soft spatial gate around the prediction
    hard_gate=95.0,               # px - never jump further than this to a blob
    accept_score=0.52,            # min fused score to accept a blob as the target
    min_appb=0.32,                # veto: blob colour must correlate at least this
    appb_weight=0.55, prox_weight=0.45,  # fusion weights (appearance / proximity)
    hist_adapt=0.05,              # EMA rate for slow appearance drift
    leash_px=210.0,               # extension: reject blobs this far from the
                                  #   running mean of recent confirmed positions
    merge_gap=100,                # post-proc: bridge present-runs within this gap
    max_coast=90,                 # forward: frames unmatched but still "present"
    max_coast_back=10,            # backward (entry side): stricter
    back_extend_max=12,           # backward: stop this many frames before backbone
                                  #   start (the walker only just entered)
    meanshift_win=80,             # px - back-projection search window side
    meanshift_min_conf=0.30,      # min back-projection density to trust mean-shift
    # name-tag (lanyard card) detector
    tag_v_min=200, tag_s_max=60, tag_area_min=5, tag_area_max=150,
    tag_local_boost=38,
    # fish-eye -> floor plane (bonus, approximate)
    fisheye_theta_max_deg=92.0,
    # optional YOLO backend
    yolo_weights="yolov8n.pt", yolo_conf=0.20, yolo_imgsz=960,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def find_disc(gray):
    """Circular fish-eye image area -> (cx, cy, r)."""
    h, w = gray.shape
    _, m = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r > 0.35 * min(h, w):
            return float(x), float(y), min(float(r), 0.60 * math.hypot(w, h))
    return w / 2.0, h / 2.0, 0.5 * min(h, w)


def fisheye_to_floor(x, y, cx, cy, R, theta_max_deg):
    """Equidistant fish-eye un-projection to a normalised floor plane
    (sensor looks straight down; mount height = 1 -> coords in 'ceiling heights').
    Bearing is preserved; radius = tan(theta).  Enough to show the path shape."""
    dx, dy = x - cx, y - cy
    theta = min((math.hypot(dx, dy) / R) * math.radians(theta_max_deg),
                math.radians(89.0))
    r = math.tan(theta)
    a = math.atan2(dy, dx)
    return r * math.cos(a), r * math.sin(a)


def tag_score_in_roi(roi_bgr):
    """Look for the lanyard card: a small, bright, near-white local peak.
    Returns (score in [0,1], (x,y,w,h) of the best blob or None)."""
    if roi_bgr is None or roi_bgr.size == 0 or min(roi_bgr.shape[:2]) < 6:
        return 0.0, None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    mv = float(V.mean())
    white = ((V >= max(P["tag_v_min"], mv + P["tag_local_boost"])) &
             (S <= P["tag_s_max"])).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    rh = roi_bgr.shape[0]
    best, best_box = 0.0, None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < P["tag_area_min"] or a > P["tag_area_max"]:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if (y + h / 2) > 0.75 * rh:            # tag sits on chest/upper back, not at the feet
            continue
        extent = a / float(max(w * h, 1))
        ar = max(w, h) / float(max(min(w, h), 1))
        if extent < 0.40 or ar > 3.2:
            continue
        margin = float(V[y:y + h, x:x + w].mean()) - mv
        s = min(1.0, extent * (0.5 + margin / 80.0))
        if s > best:
            best, best_box = s, (int(x), int(y), int(w), int(h))
    return best, best_box


# --------------------------------------------------------------------------- #
#  Motion detector - MOG2 blobs, cached per frame                             #
# --------------------------------------------------------------------------- #
def run_motion(video_path, disc, progress=lambda i, n: None):
    cx, cy, R = disc
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = cv2.createBackgroundSubtractorMOG2(history=P["mog_history"],
                                            varThreshold=P["mog_var_thresh"],
                                            detectShadows=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    per_frame, frames = [], []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        fg = bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        _, fg = cv2.threshold(fg, P["fg_thresh"], 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k, iterations=2)
        n, _, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
        dets = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if not (P["blob_min_area"] <= a <= P["blob_max_area"]):
                continue
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            if w > P["blob_max_wh"] or h > P["blob_max_wh"]:
                continue
            bcx, bcy = float(cent[i][0]), float(cent[i][1])
            if math.hypot(bcx - cx, bcy - cy) > 1.06 * R:
                continue
            dets.append(dict(cx=bcx, cy=bcy, x=int(x), y=int(y), w=int(w),
                             h=int(h), area=int(a)))
        per_frame.append(dets)
        progress(fi, n_frames)
        fi += 1
    cap.release()
    return frames, per_frame


# --------------------------------------------------------------------------- #
#  Greedy tracklets (used only to find the seed)                              #
# --------------------------------------------------------------------------- #
class Tracklet:
    def __init__(self, fi, d):
        self.pts = [(d["cx"], d["cy"])]; self.frames = [fi]
        self.boxes = [(d["x"], d["y"], d["w"], d["h"])]; self.miss = 0
    def add(self, fi, d):
        self.pts.append((d["cx"], d["cy"])); self.frames.append(fi)
        self.boxes.append((d["x"], d["y"], d["w"], d["h"])); self.miss = 0
    @property
    def last(self):
        return self.pts[-1]
    @property
    def path_len(self):
        p = np.asarray(self.pts)
        return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0
    @property
    def displacement(self):
        p = np.asarray(self.pts)
        if len(p) < 2:
            return 0.0
        return float(np.hypot(p[:, None, 0] - p[None, :, 0],
                              p[:, None, 1] - p[None, :, 1]).max())


def link_tracklets(per_frame):
    live, done = [], []
    for fi, dets in enumerate(per_frame):
        used = set()
        for tr in live:
            cxl, cyl = tr.last
            j, best = -1, P["link_max_dist"]
            for k, d in enumerate(dets):
                if k in used:
                    continue
                dd = math.hypot(d["cx"] - cxl, d["cy"] - cyl)
                if dd < best:
                    best, j = dd, k
            if j >= 0:
                tr.add(fi, dets[j]); used.add(j)
            else:
                tr.miss += 1
        for k, d in enumerate(dets):
            if k not in used:
                live.append(Tracklet(fi, d))
        keep = []
        for tr in live:
            (done if tr.miss > P["link_max_gap"] else keep).append(tr)
        live = keep
    done.extend(live)
    return [t for t in done if len(t.frames) >= P["track_min_len"]]


def _trim_to_active_core(tr):
    """Greedy linking can splice a nearby fidgeter onto the ends of the walker's
    tracklet.  Keep only the contiguous span around the peak-speed frame where
    the subject keeps moving (no >CORE_GAP-frame low-speed stretch)."""
    if len(tr.frames) < P["track_min_len"]:
        return tr
    pts = np.asarray(tr.pts)
    sp = np.r_[0.0, np.hypot(*np.diff(pts, axis=0).T)]
    sp = np.convolve(np.pad(sp, 4, "edge"), np.ones(9) / 9, "valid")  # smooth
    peak = int(np.argmax(sp))
    thr = max(6.0, 0.40 * sp[peak])
    CORE_GAP = 30
    lo = peak
    run = 0
    for i in range(peak, -1, -1):
        run = run + 1 if sp[i] < thr else 0
        if run > CORE_GAP:
            break
        lo = i
    hi = peak
    run = 0
    for i in range(peak, len(sp)):
        run = run + 1 if sp[i] < thr else 0
        if run > CORE_GAP:
            break
        hi = i
    core = Tracklet(tr.frames[lo], dict(cx=tr.pts[lo][0], cy=tr.pts[lo][1],
                    x=tr.boxes[lo][0], y=tr.boxes[lo][1],
                    w=tr.boxes[lo][2], h=tr.boxes[lo][3]))
    for i in range(lo + 1, hi + 1):
        core.add(tr.frames[i], dict(cx=tr.pts[i][0], cy=tr.pts[i][1],
                 x=tr.boxes[i][0], y=tr.boxes[i][1],
                 w=tr.boxes[i][2], h=tr.boxes[i][3]))
    return core


def pick_seed_tracklet(per_frame):
    tracks = link_tracklets(per_frame)
    if not tracks:
        return None
    movers = [t for t in tracks if t.displacement >= P["seed_min_travel"]]
    pool = movers or tracks
    best = max(pool, key=lambda t: t.displacement + 0.25 * t.path_len)
    return _trim_to_active_core(best)


# --------------------------------------------------------------------------- #
#  Appearance + coast single-target tracker                                   #
# --------------------------------------------------------------------------- #
def hs_hist(frame, box):
    x, y, w, h = [int(v) for v in box]
    x, y = max(x, 0), max(y, 0)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 250))
    hist = cv2.calcHist([hsv], [0, 1], mask, list(P["hs_bins"]), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def _smooth(seq, k=5):
    seq = np.asarray(seq, float)
    if len(seq) < k:
        return seq
    pad = np.pad(seq, (k // 2, k // 2), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(pad, ker, mode="valid")


def seeded_track(frames, per_frame, seed_frame, seed_box, disc, backbone=None):
    """Follow one person across the whole clip.

    `backbone` (optional): the clean greedy tracklet of the walker.  Its own
    frames are trusted as-is; the appearance+coast tracker only has to EXTEND
    the two ends (before it enters / after it stops moving).  Without a
    backbone the tracker runs from the single (seed_frame, seed_box).
    """
    cx, cy, R = disc
    n = len(frames)
    rec = [None] * n

    # ---- appearance model + extension anchors ----
    if backbone is not None and len(backbone.frames) >= P["track_min_len"]:
        bfr = backbone.frames
        bxs = _smooth([p[0] for p in backbone.pts])
        bys = _smooth([p[1] for p in backbone.pts])
        bw_i = [b[2] for b in backbone.boxes]
        bh_i = [b[3] for b in backbone.boxes]
        # trust the backbone on its span, interpolating its internal gaps
        for i, fi in enumerate(bfr):
            rec[fi] = dict(frame=fi, present=1, x=float(bxs[i]), y=float(bys[i]),
                           bw=float(bw_i[i]), bh=float(bh_i[i]), tag=0.0,
                           src="seed", score=1.0)
            if i and bfr[i] - bfr[i - 1] > 1:                # bridge gap
                a, b = bfr[i - 1], bfr[i]
                for g in range(a + 1, b):
                    w = (g - a) / (b - a)
                    rec[g] = dict(frame=g, present=1,
                                  x=float(bxs[i - 1] * (1 - w) + bxs[i] * w),
                                  y=float(bys[i - 1] * (1 - w) + bys[i] * w),
                                  bw=float(bw_i[i]), bh=float(bh_i[i]),
                                  tag=0.0, src="seed", score=1.0)
        for fi in bfr:                                       # tag score on real frames
            r = rec[fi]
            x, y = int(r["x"] - r["bw"] / 2), int(r["y"] - r["bh"] / 2)
            r["tag"] = tag_score_in_roi(frames[fi][max(y, 0):y + int(r["bh"]),
                                                   max(x, 0):x + int(r["bw"])])[0]
        # robust histogram over the middle third of the backbone
        lo, hi = len(bfr) // 3, 2 * len(bfr) // 3
        hs = [hs_hist(frames[bfr[i]], backbone.boxes[i]) for i in range(lo, hi + 1)]
        hs = [h for h in hs if h is not None]
        model = np.median(np.stack(hs), axis=0).astype(np.float32) if hs \
            else hs_hist(frames[seed_frame], seed_box)
        bw = float(np.median([b[2] for b in backbone.boxes]))
        bh = float(np.median([b[3] for b in backbone.boxes]))
        head = (bfr[0], bxs[0], bys[0], bxs[0] - bxs[min(3, len(bxs) - 1)],
                bys[0] - bys[min(3, len(bys) - 1)])
        tail = (bfr[-1], bxs[-1], bys[-1], bxs[-1] - bxs[-min(4, len(bxs))],
                bys[-1] - bys[-min(4, len(bys))])
    else:
        model = hs_hist(frames[seed_frame], seed_box)
        bw, bh = float(seed_box[2]), float(seed_box[3])
        sx = seed_box[0] + seed_box[2] / 2.0
        sy = seed_box[1] + seed_box[3] / 2.0
        x, y, w, h = [int(v) for v in seed_box]
        tg, _ = tag_score_in_roi(frames[seed_frame][max(y, 0):y + h, max(x, 0):x + w])
        rec[seed_frame] = dict(frame=seed_frame, present=1, x=sx, y=sy, bw=bw,
                               bh=bh, tag=tg, src="seed", score=1.0)
        head = (seed_frame, sx, sy, 0.0, 0.0)
        tail = (seed_frame, sx, sy, 0.0, 0.0)

    def match_blob(fi, px, py, vx, vy, cur_model, cur_bw, cur_bh, anchor=None):
        predx, predy = px + vx, py + vy
        best, bscore = None, -1.0
        for d in per_frame[fi]:
            dist = math.hypot(d["cx"] - predx, d["cy"] - predy)
            if dist > P["hard_gate"]:
                continue
            if anchor is not None and \
               math.hypot(d["cx"] - anchor[0], d["cy"] - anchor[1]) > P["leash_px"]:
                continue                              # can't teleport to a far desk
            hh = hs_hist(frames[fi], (d["x"], d["y"], d["w"], d["h"]))
            if hh is None:
                continue
            appb = cv2.compareHist(cur_model, hh, cv2.HISTCMP_CORREL)  # -1..1
            if appb < P["min_appb"]:                 # colour veto
                continue
            prox = math.exp(-(dist ** 2) / (2 * P["gate_sigma"] ** 2))
            sz = math.hypot(d["w"], d["h"]) / max(math.hypot(cur_bw, cur_bh), 1)
            szpen = min(abs(math.log(max(sz, 1e-3))), 1.0)
            score = (P["appb_weight"] * max(appb, 0) +
                     P["prox_weight"] * prox - 0.15 * szpen)
            if score > bscore:
                best, bscore = d, score
        return best, bscore

    def meanshift_lock(fi, px, py, cur_model):
        """Fallback: HSV back-projection + mean-shift in a window around (px,py)."""
        W = P["meanshift_win"]
        x0, y0 = int(px - W / 2), int(py - W / 2)
        x0 = max(0, min(x0, frames[fi].shape[1] - W))
        y0 = max(0, min(y0, frames[fi].shape[0] - W))
        hsv = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], cur_model, [0, 180, 0, 256], 1)
        win = (x0, y0, W, W)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, (nx, ny, nw, nh) = cv2.meanShift(bp, win, crit)
        dens = float(bp[ny:ny + nh, nx:nx + nw].mean()) / 255.0
        return (nx + nw / 2.0, ny + nh / 2.0, dens)

    def walk(anchor, stop, step, coast_limit):
        start, ax, ay, avx, avy = anchor
        px, py, vx, vy = float(ax), float(ay), float(avx), float(avy)
        cur_model = model.copy()
        cur_bw, cur_bh = bw, bh
        since_blob = 0                 # frames since the last real blob match
        conf = [(px, py)]              # recent confirmed positions (leash anchor)
        fi = start
        while 0 <= fi < n and fi != stop:
            lanchor = (sum(c[0] for c in conf) / len(conf),
                       sum(c[1] for c in conf) / len(conf))
            blob, score = match_blob(fi, px, py, vx, vy, cur_model,
                                     cur_bw, cur_bh, anchor=lanchor)
            if blob is not None and score >= P["accept_score"]:
                nx, ny = blob["cx"], blob["cy"]
                cur_bw = 0.7 * cur_bw + 0.3 * blob["w"]
                cur_bh = 0.7 * cur_bh + 0.3 * blob["h"]
                hh = hs_hist(frames[fi], (blob["x"], blob["y"], blob["w"], blob["h"]))
                if hh is not None:
                    cur_model = ((1 - P["hist_adapt"]) * cur_model +
                                 P["hist_adapt"] * hh).astype(np.float32)
                since_blob, src = 0, "blob"
                conf.append((nx, ny))
                if len(conf) > 60:
                    conf.pop(0)
            elif since_blob < coast_limit:
                # brief miss: try mean-shift back-projection to stay locked,
                # otherwise coast on a decaying velocity
                mx, my, dens = meanshift_lock(fi, px + vx, py + vy, cur_model)
                if dens >= P["meanshift_min_conf"] and \
                   math.hypot(mx - (px + vx), my - (py + vy)) < P["hard_gate"]:
                    nx, ny, src = mx, my, "meanshift"
                else:
                    nx, ny, src = px + 0.5 * vx, py + 0.5 * vy, "coast"
                since_blob += 1
            else:
                nx, ny, src = px, py, "coast"   # long miss -> absent, freeze here
                vx = vy = 0.0
                since_blob += 1
            # keep inside the disc and the frame
            d = math.hypot(nx - cx, ny - cy)
            if d > 1.02 * R:
                nx = cx + (nx - cx) * (1.02 * R) / d
                ny = cy + (ny - cy) * (1.02 * R) / d
            nx = min(max(nx, 2.0), frames[fi].shape[1] - 2.0)
            ny = min(max(ny, 2.0), frames[fi].shape[0] - 2.0)
            vx = 0.7 * vx + 0.3 * (nx - px)
            vy = 0.7 * vy + 0.3 * (ny - py)
            px, py = nx, ny
            present = since_blob <= coast_limit
            tag, _ = tag_score_in_roi(
                frames[fi][max(int(py - cur_bh / 2), 0):int(py + cur_bh / 2),
                           max(int(px - cur_bw / 2), 0):int(px + cur_bw / 2)])
            if rec[fi] is None:
                rec[fi] = dict(frame=fi, present=int(present), x=px, y=py,
                               bw=cur_bw, bh=cur_bh, tag=tag, src=src,
                               score=max(score, 0.0))
            fi += step

    # extend forward past the tail (subject known to linger -> generous coast),
    # and backward before the head (subject is entering -> tight coast)
    walk((tail[0] + 1, tail[1], tail[2], tail[3], tail[4]), n, +1, P["max_coast"])
    back_stop = -1 if backbone is None else max(-1, head[0] - P["back_extend_max"])
    walk((head[0] - 1, head[1], head[2], -head[3], -head[4]), back_stop, -1,
         P["max_coast_back"])

    # any frame the two passes never reached -> not present
    for fi in range(n):
        if rec[fi] is None:
            rec[fi] = dict(frame=fi, present=0, x=-1, y=-1, bw=0, bh=0,
                           tag=0.0, src="none", score=0.0)

    # ---- consolidate: keep the ONE presence episode around the backbone,
    #      bridging short tracker drop-outs, discarding isolated re-locks ----
    anchor_f = (backbone.frames[len(backbone.frames) // 2]
                if backbone is not None else seed_frame)
    runs = []
    i = 0
    while i < n:
        if rec[i]["present"]:
            j = i
            while j + 1 < n and rec[j + 1]["present"]:
                j += 1
            runs.append([i, j]); i = j + 1
        else:
            i += 1
    if runs:
        ci = next((k for k, (a, b) in enumerate(runs) if a <= anchor_f <= b), 0)
        a, b = runs[ci]
        changed = True
        while changed:
            changed = False
            if ci > 0 and a - runs[ci - 1][1] <= P["merge_gap"]:
                a = runs[ci - 1][0]; ci -= 1; changed = True
            if ci + 1 < len(runs) and runs[ci + 1][0] - b <= P["merge_gap"]:
                b = runs[ci + 1][1]; runs.pop(ci + 1); changed = True
        for fi in range(n):
            if not (a <= fi <= b):
                if rec[fi]["present"]:
                    rec[fi] = dict(frame=fi, present=0, x=-1, y=-1, bw=0, bh=0,
                                   tag=0.0, src="dropped", score=0.0)
            elif not rec[fi]["present"]:                 # bridge an internal hole
                lo = max(g for g in range(a, fi) if rec[g]["present"])
                hi = min(g for g in range(fi, b + 1) if rec[g]["present"])
                w = (fi - lo) / (hi - lo)
                rec[fi] = dict(frame=fi, present=1,
                               x=rec[lo]["x"] * (1 - w) + rec[hi]["x"] * w,
                               y=rec[lo]["y"] * (1 - w) + rec[hi]["y"] * w,
                               bw=rec[lo]["bw"], bh=rec[lo]["bh"],
                               tag=0.0, src="bridge", score=0.0)
    return rec


# --------------------------------------------------------------------------- #
#  Key-frame track  (analyst supplies a few waypoints on a hard clip)         #
# --------------------------------------------------------------------------- #
def keyframe_track(frames, per_frame, keyframes, disc):
    """`keyframes`: sorted list of (frame, x, y, w, h) the analyst has verified.

    Baseline position = piecewise-linear interpolation between waypoints.  Each
    frame is then refined by snapping to the nearest motion blob that is close
    to the baseline and colour-consistent with the surrounding waypoints, so the
    box follows the person's real wiggle instead of a straight line.  Every
    frame from the first to the last waypoint is 'present' (the analyst has
    asserted the subject never leaves)."""
    cx, cy, R = disc
    n = len(frames)
    kf = sorted(keyframes)
    kfr = [k[0] for k in kf]
    # appearance model = mean HS histogram over the waypoint boxes
    hs = [hs_hist(frames[f], (x, y, w, h)) for f, x, y, w, h in kf]
    hs = [h for h in hs if h is not None]
    model = np.mean(np.stack(hs), axis=0).astype(np.float32) if hs else None

    def interp(fi):
        if fi <= kfr[0]:
            _, x, y, w, h = kf[0]; return x + w / 2, y + h / 2, w, h
        if fi >= kfr[-1]:
            _, x, y, w, h = kf[-1]; return x + w / 2, y + h / 2, w, h
        j = max(i for i, f in enumerate(kfr) if f <= fi)
        f0, x0, y0, w0, h0 = kf[j]; f1, x1, y1, w1, h1 = kf[j + 1]
        t = (fi - f0) / (f1 - f0)
        return (x0 + w0 / 2) * (1 - t) + (x1 + w1 / 2) * t, \
               (y0 + h0 / 2) * (1 - t) + (y1 + h1 / 2) * t, \
               w0 * (1 - t) + w1 * t, h0 * (1 - t) + h1 * t

    rec = [None] * n
    for fi in range(n):
        bx, by, bw, bh = interp(fi)
        present = kfr[0] <= fi <= kfr[-1]
        src = "interp"
        if present:
            # snap to a nearby colour-consistent motion blob
            best, bd = None, 55.0
            for d in per_frame[fi]:
                dist = math.hypot(d["cx"] - bx, d["cy"] - by)
                if dist > 80:
                    continue
                ok = True
                if model is not None:
                    hh = hs_hist(frames[fi], (d["x"], d["y"], d["w"], d["h"]))
                    ok = hh is not None and cv2.compareHist(model, hh,
                                                            cv2.HISTCMP_CORREL) > 0.20
                if ok and dist < bd:
                    best, bd = d, dist
            if best is not None:
                bx = 0.5 * bx + 0.5 * best["cx"]
                by = 0.5 * by + 0.5 * best["cy"]
                bw = 0.6 * bw + 0.4 * best["w"]
                bh = 0.6 * bh + 0.4 * best["h"]
                src = "snap"
            if fi in kfr:
                src = "keyframe"
        tag = tag_score_in_roi(frames[fi][max(int(by - bh / 2), 0):int(by + bh / 2),
                                          max(int(bx - bw / 2), 0):int(bx + bw / 2)])[0] \
            if present else 0.0
        rec[fi] = dict(frame=fi, present=int(present),
                       x=bx if present else -1, y=by if present else -1,
                       bw=bw if present else 0, bh=bh if present else 0,
                       tag=tag, src=src, score=1.0 if src == "keyframe" else 0.6)
    return rec


# --------------------------------------------------------------------------- #
#  YOLO backend (optional, for comparison)                                    #
# --------------------------------------------------------------------------- #
def run_yolo(video_path, disc, n_frames, progress=lambda i, n: None):
    from ultralytics import YOLO
    cx, cy, R = disc
    model = YOLO(P["yolo_weights"])
    per_frame = [[] for _ in range(n_frames)]
    fi = 0
    for r in model.track(source=video_path, stream=True, classes=[0], persist=True,
                         verbose=False, conf=P["yolo_conf"], iou=0.5,
                         imgsz=P["yolo_imgsz"], tracker="bytetrack.yaml"):
        if fi >= n_frames:
            break
        for b in (r.boxes or []):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            if math.hypot(bcx - cx, bcy - cy) > 1.08 * R:
                continue
            per_frame[fi].append(dict(cx=bcx, cy=bcy, x=int(x1), y=int(y1),
                                      w=int(x2 - x1), h=int(y2 - y1),
                                      area=int((x2 - x1) * (y2 - y1)),
                                      tid=int(b.id[0]) if b.id is not None else -1))
        progress(fi, n_frames)
        fi += 1
    return per_frame


def yolo_pick_track(per_frame):
    by = {}
    for fi, dets in enumerate(per_frame):
        for d in dets:
            if d["tid"] < 0:
                continue
            by.setdefault(d["tid"], Tracklet(fi, d)) if d["tid"] not in by \
                else by[d["tid"]].add(fi, d)
    if not by:
        return None
    return max(by.values(), key=lambda t: t.displacement + 0.25 * t.path_len)


# --------------------------------------------------------------------------- #
#  Assemble record -> ranges / csv rows                                       #
# --------------------------------------------------------------------------- #
def finalise(rec, n_frames, fps, disc):
    cx, cy, R = disc
    out = []
    for r in rec:
        if r["present"] and r["x"] >= 0:
            xf, yf = fisheye_to_floor(r["x"], r["y"], cx, cy, R,
                                      P["fisheye_theta_max_deg"])
            out.append(dict(frame=r["frame"], t=r["frame"] / fps, present=1,
                            x=r["x"], y=r["y"], xn=r["x"] / (2 * R),
                            yn=r["y"] / (2 * R), xf=xf, yf=yf,
                            bx=r["x"] - r["bw"] / 2, by=r["y"] - r["bh"] / 2,
                            bw=r["bw"], bh=r["bh"], tag=r["tag"], src=r["src"]))
        else:
            out.append(dict(frame=r["frame"], t=r["frame"] / fps, present=0,
                            x=-1, y=-1, xn=-1, yn=-1, xf=0, yf=0,
                            bx=0, by=0, bw=0, bh=0, tag=0.0, src=r["src"]))
    return out


def present_ranges(rows):
    out, run = [], None
    for r in rows:
        if r["present"] and run is None:
            run = [r["frame"], r["frame"]]
        elif r["present"]:
            run[1] = r["frame"]
        elif run:
            out.append(tuple(run)); run = None
    if run:
        out.append(tuple(run))
    return out


# --------------------------------------------------------------------------- #
#  Visualisations                                                             #
# --------------------------------------------------------------------------- #
def visualise(frames, per_frame, rows, seed_frame, out_dir, fps, disc, make_video=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(rows)
    pres = np.array([r["present"] for r in rows])
    tag = np.array([r["tag"] for r in rows])
    xs = np.array([r["x"] for r in rows]); ys = np.array([r["y"] for r in rows])
    ts = np.array([r["t"] for r in rows])
    ref = frames[seed_frame].copy()
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    # 1. timeline
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2]))
    ax[0].imshow(pres[None, :], aspect="auto", cmap="Greens", vmin=0, vmax=1,
                 extent=[0, n / fps, 0, 1])
    ax[0].set_yticks([])
    rng = present_ranges(rows)
    lab = ", ".join(f"f{s}-{e} ({s / fps:.1f}-{e / fps:.1f}s)" for s, e in rng)
    ax[0].set_title(f"Task 1 - staff present (green):  {lab}", fontsize=10)
    spd = np.full(n, np.nan)
    for i in range(1, n):
        if pres[i] and pres[i - 1]:
            spd[i] = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
    mx = np.nanmax(spd) or 1.0
    ax[1].plot(ts, tag, lw=1.0, color="#c026d3", label="name-tag score")
    ax[1].plot(ts, spd / mx, lw=1.0, color="#2563eb", label="step speed (norm)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(f"{out_dir}/timeline.png", dpi=130); plt.close(fig)

    # 2. trajectory over a reference frame
    m = pres == 1
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.imshow(ref_rgb)
    pts = np.column_stack([xs[m], ys[m]]).reshape(-1, 1, 2)
    if len(pts) > 1:
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap="plasma", linewidth=3)
        lc.set_array(ts[m][:-1]); ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=.046, pad=.04).set_label("time (s)")
        ax.scatter(*pts[0, 0], s=90, c="lime", ec="k", zorder=5, label="start 0:00")
        ax.scatter(*pts[-1, 0], s=90, c="red", ec="k", zorder=5, label="end 0:53")
        for tt in (13, 18, 24, 30, 45):        # time waypoint labels along the path
            k = int(np.argmin(np.abs(ts[m] - tt)))
            ax.annotate(f"{tt}s", (xs[m][k], ys[m][k]), color="k", fontsize=8,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.7))
        ax.legend(loc="lower right")
    ax.set_title("Task 2 - staff trajectory (image coords)")
    ax.set_xlim(0, ref_rgb.shape[1]); ax.set_ylim(ref_rgb.shape[0], 0)
    fig.tight_layout(); fig.savefig(f"{out_dir}/trajectory.png", dpi=130); plt.close(fig)

    # 2b. de-warped floor path
    xf = np.array([r["xf"] for r in rows]); yf = np.array([r["yf"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    if m.any():
        sc = ax.scatter(xf[m], yf[m], c=ts[m], cmap="plasma", s=14)
        ax.plot(xf[m], yf[m], color="#888", lw=.6, alpha=.6)
        fig.colorbar(sc, ax=ax, label="time (s)", fraction=.046, pad=.04)
    ax.scatter(0, 0, marker="+", s=200, c="k"); ax.text(0, 0, " sensor nadir", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("x / mount-height"); ax.set_ylabel("y / mount-height")
    ax.set_title("Task 2 (bonus) - de-warped floor path")
    fig.tight_layout(); fig.savefig(f"{out_dir}/trajectory_floor.png", dpi=130); plt.close(fig)

    # 3. tag-evidence montage (strongest name-tag sightings, in time order)
    hits = sorted((r for r in rows if r["tag"] > 0.3),
                  key=lambda r: -r["tag"])[:24]
    hits.sort(key=lambda r: r["frame"])
    tiles = []
    for r in hits:
        fi = r["frame"]
        x, y, w, h = int(r["bx"]), int(r["by"]), int(r["bw"]), int(r["bh"])
        crop = frames[fi][max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (120, 160))
        cv2.putText(crop, f"f{fi}", (4, 152), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(crop)
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        rowsimg = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        cv2.imwrite(f"{out_dir}/tag_evidence.png", np.vstack(rowsimg))

    # 4. annotated video
    if not make_video:
        return
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(f"{out_dir}/annotated.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trail = []
    for fi, r in enumerate(rows):
        f = frames[fi].copy()
        for d in per_frame[fi]:
            cv2.rectangle(f, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]),
                          (150, 150, 150), 1)
        if r["present"]:
            x, y, w, h = int(r["bx"]), int(r["by"]), int(r["bw"]), int(r["bh"])
            col = {"blob": (0, 210, 0), "meanshift": (0, 190, 190),
                   "coast": (0, 140, 255)}.get(r["src"], (0, 210, 0))
            cv2.rectangle(f, (x, y), (x + w, y + h), col, 2)
            cv2.circle(f, (int(r["x"]), int(r["y"])), 3, (0, 0, 255), -1)
            cv2.putText(f, "STAFF" + ("  tag!" if r["tag"] > 0.25 else "") +
                        f"  [{r['src']}]", (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2, cv2.LINE_AA)
            trail.append((int(r["x"]), int(r["y"])))
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(f, a, b, (255, 0, 255), 1)
        cv2.putText(f, f"f{fi} t={fi / fps:5.2f}s  "
                    f"{'STAFF PRESENT' if r['present'] else '--'}",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(f)
    vw.release()


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backend", choices=["motion", "yolo"], default="motion")
    ap.add_argument("--seed-frame", type=int, default=None)
    ap.add_argument("--seed-box", default=None, help="x,y,w,h of the staff at --seed-frame")
    ap.add_argument("--keyframes", default=None,
                    help="analyst waypoints, ';'-separated 'frame:x,y,w,h' "
                         "(e.g. '5:72,372,66,90; 390:485,95,80,150; 1338:58,352,74,96')")
    ap.add_argument("--keyframes-file", default=None,
                    help="text file with one 'frame:x,y,w,h' per line")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, f0 = cap.read()
    disc = find_disc(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY))
    cap.release()

    def prog(i, n):
        if i % 100 == 0:
            print(f"  ..frame {i}/{n}", end="\r")

    print(f"[info] {os.path.basename(args.video)}  fps={fps:.2f}  "
          f"disc(cx,cy,r)=({disc[0]:.0f},{disc[1]:.0f},{disc[2]:.0f})  backend={args.backend}")

    # ---- parse analyst keyframes, if any ----
    kf_text = args.keyframes
    if args.keyframes_file:
        with open(args.keyframes_file) as fh:
            kf_text = ";".join(l for l in fh)
    keyframes = None
    if kf_text:
        keyframes = []
        for part in kf_text.replace("\n", ";").split(";"):
            part = part.split("#")[0].strip()          # strip inline comments
            if not part:
                continue
            f, box = part.split(":")
            x, y, w, h = (int(v) for v in box.split(","))
            keyframes.append((int(f), x, y, w, h))
        keyframes.sort()
        print(f"[info] {len(keyframes)} analyst keyframes "
              f"f{keyframes[0][0]}..f{keyframes[-1][0]}")

    if args.backend == "yolo":
        cap = cv2.VideoCapture(args.video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        per_frame = run_yolo(args.video, disc, n_frames, prog); print()
        seed_tr = yolo_pick_track(per_frame)
        frames = None
    else:
        frames, per_frame = run_motion(args.video, disc, prog); print()
        seed_tr = None if keyframes else pick_seed_tracklet(per_frame)

    # ---- decide the seed ----
    if keyframes:
        seed_frame = keyframes[len(keyframes) // 2][0]
        seed_box = keyframes[len(keyframes) // 2][1:]
        backbone = None
    elif args.seed_frame is not None and args.seed_box:
        seed_frame = args.seed_frame
        seed_box = tuple(int(v) for v in args.seed_box.split(","))
        backbone = None
        print(f"[info] manual seed: frame {seed_frame} box {seed_box}")
    else:
        if seed_tr is None:
            sys.exit("[fail] no moving tracklet found - pass --seed-frame/--seed-box")
        backbone = seed_tr
        mid = len(seed_tr.frames) // 2
        seed_frame, seed_box = seed_tr.frames[mid], seed_tr.boxes[mid]
        print(f"[info] auto backbone = largest mover: tracklet "
              f"f{seed_tr.frames[0]}-{seed_tr.frames[-1]}, "
              f"travel {seed_tr.displacement:.0f}px, path {seed_tr.path_len:.0f}px")

    if frames is None:                       # yolo backend still needs raw frames
        frames = []
        cap = cv2.VideoCapture(args.video)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        per_frame = [[dict(cx=d["cx"], cy=d["cy"], x=d["x"], y=d["y"],
                           w=d["w"], h=d["h"], area=d["area"]) for d in df]
                     for df in per_frame]

    if keyframes:
        rec = keyframe_track(frames, per_frame, keyframes, disc)
    else:
        rec = seeded_track(frames, per_frame, seed_frame, seed_box, disc,
                           backbone=backbone)
    rows = finalise(rec, len(frames), fps, disc)
    ranges = present_ranges(rows)
    npv = sum(r["present"] for r in rows)

    # ---- CSV ----
    import csv
    with open(f"{args.out}/staff_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "present", "x_px", "y_px", "x_norm", "y_norm",
                    "x_floor", "y_floor", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                    "tag_score", "source"])
        for r in rows:
            w.writerow([r["frame"], f"{r['t']:.3f}", r["present"],
                        f"{r['x']:.1f}", f"{r['y']:.1f}", f"{r['xn']:.4f}",
                        f"{r['yn']:.4f}", f"{r['xf']:.4f}", f"{r['yf']:.4f}",
                        f"{r['bx']:.0f}", f"{r['by']:.0f}", f"{r['bw']:.0f}",
                        f"{r['bh']:.0f}", f"{r['tag']:.3f}", r["src"]])

    summ = dict(video=os.path.basename(args.video), n_frames=len(frames), fps=fps,
                backend=args.backend, seed_frame=seed_frame,
                seed_box=[int(v) for v in seed_box],
                present_frame_ranges=[list(r) for r in ranges],
                present_time_ranges_s=[[round(a / fps, 2), round(b / fps, 2)]
                                       for a, b in ranges],
                n_present=npv, pct_present=round(100 * npv / len(frames), 1),
                tag_frames=[r["frame"] for r in rows if r["tag"] > 0.25],
                params=P)
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)

    print("\n==== RESULT ====")
    print("Task 1  staff present:")
    for a, b in ranges:
        print(f"   f{a}-{b}   t {a / fps:.2f}-{b / fps:.2f}s   ({b - a + 1} frames)")
    print(f"   total {npv}/{len(frames)} frames ({summ['pct_present']}%)")
    if npv:
        fp = next(r for r in rows if r["present"])
        lp = [r for r in rows if r["present"]][-1]
        print(f"Task 2  first f{fp['frame']}: (x,y)=({fp['x']:.0f},{fp['y']:.0f})px"
              f"   last f{lp['frame']}: (x,y)=({lp['x']:.0f},{lp['y']:.0f})px")

    print(f"[info] writing visualisations{'' if args.no_video else ' + annotated.mp4'} ...")
    try:
        visualise(frames, per_frame, rows, seed_frame, args.out, fps, disc,
                  make_video=not args.no_video)
        print(f"[done] outputs in {args.out}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[warn] visualisation failed ({e}); CSV + summary still written")


if __name__ == "__main__":
    main()
