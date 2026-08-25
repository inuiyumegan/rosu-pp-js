# tools

Visualisers and measurement harnesses for the osu!mania accuracy surface. None are part
of the build. The visualisers read CSVs produced by `#[ignore]`d tests in `src/sunny.rs`
and write into `target/surface/`, which is gitignored.

## Fixture fetchers

Both write into `local-fixtures/` (gitignored — not redistributable, not needed to
build) and need the bancho.py MySQL container reachable.

- `fetch_batch.sh` — scores ranked by pp across players, with an EZ cohort and a no-mod
  control. The set the mod response was measured on.
- `fetch_ladder.sh` — *difficulty ladders*: many scores from one player inside one
  quarter, stratified across star rating. `fetch_batch.sh` cannot fit sigma's difficulty
  response because it returns roughly one score per player, and the error model has one
  free skill each; a ladder holds skill roughly fixed while difficulty sweeps.

```sh
tools/fetch_ladder.sh 30                            # 4 default cohorts x 30 scores
tools/fetch_ladder.sh 30 4616:2023:3                # userid:year:quarter
tools/parse_replay.py --batch local-fixtures/ladder.tsv --json local-fixtures/ladder-errors.json
```

`fetch_ladder.sh` always writes `local-fixtures/ladder.tsv`, so **fetching a second
cohort overwrites the first** — rename the file after each run if you want more than one
skill band. The TSV is reconstructable from a `--json` dump if it is lost, since that
carries the count vector and the DB columns.

Note that a ladder deliberately excludes saturating scores (`acc between 88 and 99.5`),
which makes it the wrong set for anything about the clean end of the surface: see
`ErrorModel::sigma_floor`, where a replay-measured floor turned out to be contradicted by
the judgement counts of near-perfect scores the ladder never sampled.

`parse_replay.py` turns `.osr` replays into per-note hit errors, which measures a
player's timing sigma directly instead of inferring it from judgement counts. Its
`--verify` mode recomputes judgements and diffs them against the server's stored counts;
that is the correctness check on the whole pipeline. See the module docstring, which
records which pairing rules were tested and rejected.

Fit against sunny's own star ratings, not bancho's stored `maps.diff` — the two are
different calculations (`log`-`log` slope 0.78), and an exponent is only meaningful in
the units its difficulty is expressed in:

```sh
cut -f4 local-fixtures/ladder.tsv | tail -n +2 | sort -u \
  | cargo test --release ladder_stars -- --ignored --nocapture --exact sunny::tests::ladder_stars
```

To see what the surface makes of a ladder — fitted skill, window scalar, fit quality,
grouped per player and summarised by star band:

```sh
cat local-fixtures/ladder-*.tsv \
  | cargo test --release ladder_report -- --ignored --nocapture --exact sunny::tests::ladder_report
```

The TSV's `pp` column is the **live ppy.sb figure, from an older algorithm than sunny**,
so treat a ratio against it as the gap between two algorithms rather than as error in
this one.

To see what a candidate `sigma_floor` would do — to fit quality and to pricing, which
turn out to be different questions:

```sh
cargo test --release sigma_floor_sweep -- --ignored --nocapture --exact sunny::tests::sigma_floor_sweep
```

Read the two halves against each other. `mean_g` is bit-identical across the whole
0-10 ms sweep, because the counts pin sigma and the fit absorbs any small floor into
skill; the window scalar moves anyway, because it is a ratio of skills at two different
sigmas and quadrature is nonlinear. The lower table is the binding constraint: a
1506-note all-320 score allows about 2 ms and rules out 5.

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
