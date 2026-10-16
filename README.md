# FootfallCam AI Evaluation — staff (name-tag wearer) detection

Find, in an overhead fish-eye "3D sensor" clip:

1. **Task 1** — which frames contain the staff member (the person wearing the
   lanyard name tag).
2. **Task 2 (bonus)** — the staff `(x, y)` position in every frame they appear.

For the provided clip the answer to Task 1 is **every frame** — the staff never
leaves the room — so the substance is Task 2 (localisation). See `../SOLUTION.md`
for the write-up and `../staff_findings_report.html` for the visual summary.

## Quick start

```bash
python -m pip install -r requirements.txt

# provided clip — uses the 8 verified waypoints
python detect_staff.py --video "../WhatsApp Video 2026-09-09 at 8.31.13 AM.mp4" \
       --out ../outputs --keyframes-file sample_keyframes.txt
```

The pipeline needs only **numpy + opencv + matplotlib** — no model download — so
it runs on any laptop against any clip.

## Running on a fresh clip

```bash
# fully automatic: proposes the single most-mobile person and tracks them
python detect_staff.py --video other.mp4 --out ../outputs

# analyst-assisted: give a few 'frame:x,y,w,h' waypoints of the staff
python detect_staff.py --video other.mp4 --out ../outputs \
       --keyframes "5:70,370,70,90; 400:520,120,80,150; 1200:90,400,70,95"

# ...or point at a file of them (one per line, '#' comments allowed)
python detect_staff.py --video other.mp4 --out ../outputs --keyframes-file wp.txt

# person-detector comparison backend
python -m pip install ultralytics
python detect_staff.py --video other.mp4 --out ../outputs --backend yolo
```

Why waypoints exist: on a crowded overhead fish-eye view, with the subject
stationary at a desk most of the time and (here) changing clothes mid-clip,
fully-automatic single-target tracking drifts. The auto mode still works when one
person clearly moves and the others don't; `--keyframes` is the fallback for hard
clips and is a realistic analyst-in-the-loop workflow.

## Outputs (written to `--out`)

| file | contents |
|---|---|
| `staff_frames.csv` | per frame: `present`, `x_px,y_px`, `x_norm,y_norm`, de-warped `x_floor,y_floor`, bbox, `tag_score`, `source` |
| `annotated.mp4` | staff boxed green (`STAFF`), other movers grey, running trail, per-frame status |
| `timeline.png` | presence bar + name-tag / step-speed traces |
| `trajectory.png` | staff path over a reference frame, coloured by time, with time labels |
| `trajectory_floor.png` | de-warped floor path (bonus) |
| `tag_evidence.png` | montage of the name-tag crops that fired |
| `summary.json` | frame ranges, counts, all parameters |

## How it works (one paragraph)

**Motion** — MOG2 background subtraction → cleaned foreground blobs, constrained
to the auto-detected fish-eye disc; seated people barely register, a walker
leaves a large travelling blob. **Proposal** — blobs are linked frame-to-frame by
nearest centroid; the tracklet that travels furthest is the candidate staff
member. **Full track** — from either that proposal (auto) or the analyst
waypoints, positions are interpolated across the clip and each frame is snapped
to the nearest colour-consistent motion blob so the box follows real movement.
**Name-tag cue** — inside the tracked box, a small bright low-saturation local
peak on the upper body is logged as `tag_score` (evidence, not a gate). All
tunables are in one `P = dict(...)` at the top of `detect_staff.py`.
