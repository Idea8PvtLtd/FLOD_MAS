# Runtime flow of one inspection

## From a frame to the result tabs

![One inspection cycle. The loop on the left runs on every new frame; the column on the right runs once per capture.](figures/cycle.svg)

1. **Hands.** Each new frame is reduced to 320 px wide and given to the hand detector.
   The raw yes/no is debounced over `debounce_frames` frames.
2. **Checkpoint.** A capture is triggered when hands were seen and then stayed away for
   `hand_absence_seconds`.
3. **Snapshot.** That exact frame is copied. The hand result, the fabric verdict, the saved
   image and the measurement all refer to this one copy.
4. **Fabric gate.** A classifier labels the snapshot `full_fabric`, `half_fabric` or
   `no_fabric`. Only labels in `accepted_labels` above `fabric_min_confidence` go on;
   `no_fabric` clears the screen.
5. **Undistort**, then **segment** through the provider, then **validate**: polygons must be
   inside the frame; with `fragment_policy = "error"` several pieces for one fabric stop the
   capture; more than `max_fabrics` strips stop it.
6. **Measure** every fabric (chapter 7), **save** the capture set (chapter 9), **show** one
   tab per fabric, cycling every `result_display_seconds` until the fabric is removed.

## The hand-cycle checkpoint

![States of the checkpoint. A capture is taken on entering TRIGGERED.](figures/checkpoint.svg)

A new hand seen while work is in progress raises the controller's *generation* number at
once, before debouncing. Results carry the generation they were started under; a result
from an older generation is saved but not shown. This is how a result for an arrangement the
operator has already changed is kept off the screen.

## Threads and what passes between them

![Threads. Only the Tk thread touches widgets; the others talk to it through the events queue.](figures/threads.svg)

| Thread | Does | Hands over |
|---|---|---|
| Capture | Reads the camera, keeps the newest frame | frame to the Tk thread |
| Tk main | Draws; calls `submit(frame, sequence)`; polls events every 100 ms | newest frame into a one-slot box |
| `automatic-hand-monitor` | Hand, checkpoint, fabric gate | a job to the inspection thread; events to Tk |
| `automatic-inspection` | `InspectionPipeline.run` for one capture | a finished `Future` |

- The one-slot box means the monitor always works on the newest frame; older ones are
  dropped, never queued.
- At most one capture is being processed and one is waiting. A newer capture replaces the
  waiting one, which is recorded as `superseded`.
- Events are `status`, `result`, `clear` and `error`. Errors are always shown, even from an
  outdated capture.

## Statuses a fabric can end with

| Status | Meaning | Counts as |
|---|---|---|
| `PASS` | every segment's average width is inside target +- tolerance | pass |
| `FAIL` | at least one segment is outside | fail |
| `UNMEASURED` | no millimetre calibration: pixels only | warning |
| `NO STRIP` | a fabric was matched but no strip came back (`report_missing_strips`) | warning |
| `FRAGMENTED` | the strip came in pieces; the largest is shown (`fragment_policy = "largest"`) | warning |
| `LOW CONFIDENCE` | below `segmentation.min_confidence` | warning |

The screen shows FAIL if any fabric fails, PASS only if all pass and nothing warned, and
WARNING otherwise. A fabric that could not be graded never reads as a pass.
