# Segmentation providers

## One contract

![The provider contract. Everything a provider does to the image stays inside it; only frame-pixel polygons come out.](figures/providers.svg)

A provider is any class with `__init__(settings)` and `segment(frame_bgr)` that returns a
`SegmentationResult`. Its `Instance` objects carry a polygon in the pixels of the frame it
was given, a confidence, a label, an optional panel box and an optional note. Resizing,
colour conversion, model loading and post-processing are the provider's own business;
`validate_result` refuses anything that is not in frame coordinates.

`segmentation.provider` in `config.json` chooses it: the word `workflow`, or
`package.module:Factory` for any other class.

## The three providers on this branch

| | Cloud workflow (default) | Local detector | Region-guided local detector |
|---|---|---|---|
| Setting | `"workflow"` | `"segmentation.glue_line:GlueLineSegmenter"` | the same, with `roi_model` and `roi_guided` |
| Where it runs | Roboflow (image uploaded) | this PC, CPU | this PC; SAM 3 in the cloud only if `sam_refine` |
| Input | `roi` crop resized to 1024 x 768, optional CLAHE | full-resolution frame | full-resolution frame + panel regions |
| Finds | panels and strips, any fabric | the bead on pale fabric: a bright line between two dark lines | one line along the curved side of each panel |
| Dark fabric | yes | reports nothing | yes (`bead_model = "line"` or `"auto"`) |
| Needs | `ROBOFLOW_API_KEY`, a connection | nothing | the region model file (not in the repository) |
| Measured | by the reference project | 56 pale sample frames: 52 beads, 7-30 s per frame | through the Pipeline Lab: chapters 12-14 |

## How the local detector works, in short

1. Remove the lighting gradient; run a long directional line filter, kept separately for
   dark lines and bright lines.
2. Look only inside pale fabric, away from its outline.
3. At each point of a dark line, look straight across for a second dark line running the
   same way within the width range, with a bright line between them.
4. Join such points into centrelines; bridge short faded stretches (`join_gap_px`,
   `track_gaps`).
5. Keep a strip only where the centre is brighter than just outside its edges.
6. Optionally (`shape_prior`) require the line to have the product's known wave, which
   reports whole lines and rejects seams and folds.

The region-guided mode replaces steps 2-5 by a search inside a corridor along the curved
side of each panel region (`corridor_near_px`, `corridor_far_px`). It is what the research
flow uses for its *rough line*.

## Switches that keep old behaviour

Every key added on this branch is off or neutral as shipped; with the shipped values the
detector's output was checked byte for byte against the version before them.

| Key in `segmentation.glue_line` | Shipped | Turns on |
|---|---|---|
| `shape_prior.enabled` | false | the wave-shape condition |
| `background` | `white` | fabric on any table (`any`) |
| `bead_model` | `edges` | one-band model for coloured fabric (`line`, `auto`) |
| `roi_model` | empty | the panel-region model |
| `roi_guided` | false | one line per panel side |
| `roi_strict` | false | never search outside the region |
| `enhance_bead` | false | search on the bead-enhanced image |
| `sam_refine` | false | outline with SAM 3 (**uploads image tiles**) |

The full tables are in `docs/inspection/AUTOMATIC_INSPECTION.md`.
