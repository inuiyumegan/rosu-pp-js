# tools

Visualisers for the osu!mania accuracy surface. Neither is part of the build; both
read CSVs produced by `#[ignore]`d tests in `src/sunny.rs` and write into
`target/surface/`, which is gitignored.

## Setup

```sh
uv venv .venv && uv pip install --python .venv/bin/python plotly matplotlib numpy
```

Then use `.venv/bin/python` in place of `python3` below.

## `mania_surface.py` — interactive, 3D

Browsable HTML: judgement-band wireframes over an (OD, skill) grid, at one map
difficulty. A dropdown switches between judgement bands, 305-weighted accuracy, and
the EZ gain, for each of classic/lazer scoring and each mod state.

```sh
tools/mania_surface.py                                          # default 13.77* slice
tools/mania_surface.py --map path/to.osu --clock-rate 1.5 --open
tools/mania_surface.py --stars 8.5                              # no beatmap needed
tools/mania_surface.py --no-dump                                # reuse the last CSV
```

Only one mod state draws at a time, and everything is wireframe rather than solid.
Both are deliberate: the bands overlap heavily, EZ's surfaces sit directly above
NM's, and an opaque sheet hides whatever is beneath it.

The OD axis is the reason this view exists. Under classic scoring the 320 ridge is
*flat* in OD — PERFECT is pinned at 16 ms while everything below it shifts by
`3 * (10 - od)` — so OD moves the 300/200 boundary and never the 320 rate. Switch to
a `lazer` entry and that ridge tilts, because lazer interpolates PERFECT over OD too.
EZ scales every window including PERFECT, which is why it is the only thing that
moves classic's 320 count, and why it prices as strongly as it does.

Converts are not swept: their classic windows key off a single `round(od) > 4`
threshold, so an OD axis would be two flat plateaus rather than a surface.

## `mania_surface_2d.py` — static overview, PNG

Four panels over the whole (difficulty, skill) plane at fixed windows. Usually the
better starting point, since the 3D view is one difficulty slice.

```sh
tools/mania_surface_2d.py --fit-skill 10.305 --target-accuracy 0.91672
```

1. Accuracy shortfall `1 - acc`, log-scaled — accuracy itself is flat above ~95% over
   most of the plane and hides the structure.
2. Judgement composition against skill at one difficulty, with implied sigma.
3. The same difficulty under several window sets; `window_scalar` is the horizontal
   gap between the curves.
4. Miss rate, which is just the mass beyond the last window rather than its own term.

## Reading these honestly

The straight, parallel contours in panels 1 and 4 are an *assumption*, not a
measurement. They follow from `sigma(d, skill) = sigma_ref * ((d + floor)/skill)^exp`
with `skill_exponent` fixed at 1.7 and `difficulty_floor` at 0.6 — neither of which
has been fitted, because the calibration set is one player across a narrow
0.96–1.72x star ratio. Straightness claims a 20-star map at skill 20 grades exactly
like a 2-star map at skill 2; that is the thing new data needs to test.

Local difficulty is also still uniform: every note carries the map's whole star
rating, which is why the mod response barely varies between maps.
