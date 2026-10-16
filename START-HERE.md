# FootfallCam AI Evaluation — Submission

**Task.** In the overhead fish-eye 3D-sensor clip (`sample.mp4`, 960×720, 25 fps,
1341 frames, 53.6 s): find the frames containing the staff member (lanyard
name-tag wearer) and their `(x, y)` in each.

## Read in this order

1. **`SOLUTION.pdf`** — the 2-page written documentation (method + results + limits).
2. **`staff_findings_report.pdf`** — the same, with the figures (visual walk-through).
3. **`outputs/annotated.mp4`** — the clip with the staff boxed on every frame.
4. **`solution/detect_staff.py`** — the code.

`SOLUTION.md` and `staff_findings_report.html` are the source of the two PDFs, if
you prefer those formats.

## Answer in brief

- **Task 1:** the staff member is on camera in **every one of the 1341 frames**
  (0.0–53.6 s) — they only move between two desks, they never leave the room.
- **Task 2:** they start and end at the left-wall desk; between, they walk a loop
  (up the central corridor → top desks → bottom-centre desk → back). They wear a
  light-blue shirt for the first ~26 s, then a dark hoodie. Per-frame position:
  **`outputs/staff_frames.csv`**.

## Run it

```bash
cd solution
python -m pip install -r requirements.txt        # numpy, opencv-python, matplotlib

# the provided clip (point --video at your copy of sample.mp4)
python detect_staff.py --video "<path to sample.mp4>" \
    --out ../outputs --keyframes-file sample_keyframes.txt

# any fresh clip, fully automatic
python detect_staff.py --video other.mp4 --out ../outputs
```

The optional `--backend yolo` comparison needs `ultralytics`.

## Contents

```
START-HERE.md                 this file
SOLUTION.pdf / SOLUTION.md    2-page documentation
staff_findings_report.pdf/.html   visual one-pager (figures + findings)
solution/
  detect_staff.py             the pipeline (one CLI; params in one dict up top)
  sample_keyframes.txt        8 verified waypoints of the staff for this clip
  requirements.txt
  README.md                   how to run, including on a new clip
outputs/
  staff_frames.csv            per frame: present, x,y (px / normalised / de-warped), bbox, tag score
  annotated.mp4               staff boxed green on every frame + trail
  timeline.png                presence bar + motion / name-tag traces
  trajectory.png              path over a reference frame, coloured by time
  trajectory_floor.png        de-warped floor path (bonus)
  tag_evidence.png            montage of name-tag detections
  summary.json                frame ranges, counts, every parameter used
```

*The source video is not bundled (it is your file). Pass its path with `--video`
to re-run.*
