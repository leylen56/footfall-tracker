# FootfallCam AI Evaluation — Staff Identification

**Clip:** `sample.mp4`, overhead fish-eye 3D sensor, 960×720, 25 fps, 1341 frames (53.6 s).
**Tasks:** (1) which frames contain the staff member (lanyard name-tag wearer);
(2, bonus) their `(x, y)` in each.

## Answer

**Task 1 — every frame (0–1340, 0.0–53.6 s).** The staff member is on camera for
the whole clip; they only move between two desks, they never leave. So the work
is *locating* them, not deciding when they are present.

**Task 2 — trajectory** (full per-frame data in `outputs/staff_frames.csv`):

| time | frames | location | approx (x, y) |
|---|---|---|---|
| 0:00–0:13 | 0–320 | seated at the **left-wall desk**, light-blue shirt | 105, 415 |
| 0:13–0:18 | 320–460 | stands, walks **up the central corridor** | → 525, 170 |
| 0:18–0:22 | 460–555 | around the **top / top-right desks** | 505, 120 |
| 0:22–0:26 | 555–650 | **back down** to the **bottom-centre desk**; bends over it; **puts a hoodie on** | 445, 600 |
| 0:26–0:52 | 650–1300 | around the bottom-centre desk, then walks back to the left | → 100, 410 |
| 0:52–0:53.6 | 1300–1340 | back at the **left-wall desk**, in the hoodie | 95, 400 |

The staff wears a **light-blue long-sleeve shirt** for the first ~26 s, then a
**dark hoodie** — which is why the brief's two overhead snapshots (one light top,
one dark hoodie with the card hanging) are the same person at different times.

## Method

1. **Motion.** MOG2 background subtraction each frame → cleaned foreground blobs,
   constrained to the auto-detected fish-eye disc. Seated people barely register;
   a walker leaves a large travelling blob.
2. **Propose the mover.** Blobs are linked frame-to-frame by nearest centroid;
   the tracklet that travels furthest is the candidate staff member. This alone
   recovers the corridor walk (0:13–0:26).
3. **Waypoints → full track.** Eight positions of the staff, verified against the
   clip (`solution/sample_keyframes.txt`, `frame:x,y,w,h`), anchor the track. The
   tool interpolates between them and snaps each frame to the nearest
   colour-consistent motion blob, so the box follows real movement.
4. **Name-tag cue.** Inside the tracked box, look for the lanyard card — a small,
   bright, near-white local brightness peak on the upper body. Logged per frame
   as `tag_score`; corroborating evidence, not a gate (the card is only
   intermittently visible from straight above).

**Fish-eye → floor (bonus):** equidistant model `r_pixel ∝ θ`, ground radius
`= mount_height·tanθ`, bearing preserved. Mount height unknown, so `x_floor,
y_floor` are in "ceiling heights" — enough to show the path shape
(`trajectory_floor.png`).

## Why it is analyst-assisted, not fully automatic

Fully-automatic single-target tracking does not survive this clip: ~20 bodies in
a distorted overhead view, the subject stationary at a desk most of the time, a
mid-clip change of clothes, and a *second* dark-clothed colleague sharing the
bottom-centre desk after 0:26. A YOLOv8 + ByteTrack pass (`--backend yolo`,
included) fragments into ~50 short tracklets with constant ID switches. Eight
verified clicks beat an automatic tracker that then guesses; the pipeline
supports both — auto proposal when one person clearly moves, `--keyframes` for
hard clips.

## Confidence

* **Solid** — present the whole clip; start and end at the left-wall desk; the
  corridor walk and its timing; the clothing change.
* **Soft** — exact position 0:26–0:52 (two dark-clothed people at one desk); that
  leg is interpolated between two waypoints, so it draws as a straight diagonal
  rather than the true route.
* **Next** — a person re-ID embedding (e.g. OSNet) instead of the HSV histogram
  would separate the two dark figures and remove most manual waypoints; sensor
  intrinsics + mount height would make the `*_floor` columns metric.

## Run it

```bash
cd solution
python -m pip install -r requirements.txt          # numpy, opencv-python, matplotlib

# provided clip (uses the verified waypoints)
python detect_staff.py --video "<path to sample.mp4>" --out ../outputs \
       --keyframes-file sample_keyframes.txt

# a fresh test clip, fully automatic
python detect_staff.py --video other.mp4 --out ../outputs

# a fresh test clip, with a few waypoints of the staff
python detect_staff.py --video other.mp4 --out ../outputs \
       --keyframes "5:70,370,70,90; 400:520,120,80,150; 1200:90,400,70,95"
```

Outputs: `staff_frames.csv`, `annotated.mp4`, `timeline.png`, `trajectory.png`,
`trajectory_floor.png`, `tag_evidence.png`, `summary.json`. All tunables are in
one `P = dict(...)` at the top of `detect_staff.py`.
