# Input-State Accuracy Surface

## Purpose

Low-OD mania maps, especially LN-heavy reverse-pattern (反键) maps, can receive
less pp because the current accuracy surface prices their judgement windows against
a fixed OD-8 reference. That is only correct if the map's lower OD also means an
easier timing surface. In mania, OD is partly a charting convention, and difficult
LN input patterns can exist on low-OD maps.

The intended fix is not a user-offset correction and not an unconditional pp bonus.
The map should earn more pp only when its object sequence implies a genuinely harder
input surface. The model must therefore distinguish local input patterns and feed
their expected judgement distributions into the existing accuracy-surface fit.

## Current State

The current Sunny path is:

```text
HitObjects -> Note/RebirthData -> star rating + per-note difficulty bins
                              -> ordered input operations + input-state bins
           -> SunnyManiaDifficultyAttributes
score counts + attributes -> judgement_units -> timing fit
                         -> played/reference scalar -> pp
```

Implemented and verified:

- `Note { column, head, tail }` extraction in `src/sunny.rs`.
- Per-note difficulty compressed into 16 equal-count `NoteDifficultyBin`s.
- LN duration and release spread modelling in `src/mania_accuracy.rs`.
- A six-category accuracy surface fitted from fractional judgement units.
- Deterministic press/release operation extraction with simultaneous-operation,
  overlapping-hold, invalid-hold, chord, jack, rapid re-press, release-to-press, and
  press-under-hold tests.
- Seven primary `InputClass` values plus cached chord width, other-held count,
  predecessor count/gap, hold duration, and per-note difficulty context.
- A bounded 112-bin `(InputClass, difficulty quantile)` representation carried in
  `SunnyManiaDifficultyAttributes` and consumed by `judgement_units()` when enabled.
- Versioned flattened JS/WASM serialization with backward-compatible omission and
  malformed-payload fallback, exposed publicly as `inputStateBins`.
- ScoreV1/ScoreV2 and partial-score weight invariants, default-path neutrality, public
  performance-path movement, invalid-hold handling, and an explicit SS-ceiling test.
- A null-controlled 1,204-score A/B report with live-pp ratios, user/mod/key/LN/OD/
  accuracy cohorts, deterministic map holdout, and largest movers.
- A repaired exact per-note same-column-gap oracle using the SS-safe fading recovery
  channel and deterministic parallel candidate indexing.

Open work:

- Re-run the multi-user report after the ScoreV1 compact-bin weighting fix: release
  operations are now excluded from the denominator when an LN is judged as one object.
- Default-vs-candidate control completed. The earlier `2938215` discrepancy was caused
  by a corrupted/truncated local map fixture (117 objects versus 5,327 judgments in the
  live score), not by the input-state implementation; the fixture has since been fixed.
  The environment toggle now treats `0`/`false` as disabled.
- Follow-up map validation completed for `3217217`, `5105809`, and `4498837`: all three
  local maps are complete and their calculated ratings agree with the corresponding
  live rows after accounting for MR/DT/V2 mods. No additional difficulty mismatch was
  found.
- Decide whether compact bins need finer gap representation. On map 3217217 they
  produce 471.2 pp versus 503.5 pp from exact per-note gaps, a 6.4% compact shortfall.
- Separate release-to-press recovery time from generic press-to-press gaps if replay
  evidence supports that distinction. The current measured curve acts on the previous
  same-column press gap even though the classifier records release context separately.
- Tune or reject the `73.12 ms` candidate using mechanism-specific policy targets. It
  remains disabled by default and currently misses the original low-OD hard-LN success
  criterion by discounting that cohort further.
- Add multi-track/hand context only after an exact oracle demonstrates an effect. Split
  hand controls remain explicitly postponed.

## Design Direction

Use the original state-machine idea with the bounded local context used by the
osu!standard tap/doubletap model.

The state machine supplies persistent key/LN state. The lookback/lookahead window
supplies local pattern context. Do not begin with a global timing offset.

### Input operations

Convert each note into ordered operations:

```text
InputOperation {
    column: usize,
    time_ms: f64,
    kind: Press | Release,
    hold_duration_ms: Option<f64>,
    chord_mask: bitset,
}
```

For ScoreV1, an LN press and release may eventually be combined into one judgement;
for ScoreV2 they remain separate judgements. The operation stream must preserve both
representations without duplicating objects in the wrong scoring mode.

### Per-column state

Maintain one state per column:

```text
Idle | Pressed | Held
```

`Release` is an operation/transition, not necessarily a persistent state. This avoids
confusing a one-time action with the condition that exists between notes.

### Lookback/lookahead context

For each operation, derive a compact context from:

- previous same-column operation and gap;
- next same-column operation and gap;
- previous/current/next chord masks;
- current hold duration;
- number of other columns currently held;
- whether this is a rapid re-press, jack, release-to-press, or press-under-hold.

The first implementation should use a small explicit class set:

```text
FreshPress
RapidRepress
Jack
Release
ReleaseToPress
PressUnderHold
ChordEntryOrExit
```

Classes must be deterministic functions of map objects. They must not inspect replay
timings or player-specific settings.

## Expected Hit Surface

Implement:

```text
tick(previous_state, operation, context, local_difficulty)
    -> (new_state, expected_hits[6])
```

`expected_hits` contains fractional 320/300/200/100/50/miss contributions. Every
operation contributes exactly its judgement weight, and the invariant is:

```text
sum(expected_hits) == total_judgements
```

The first version should express class difficulty through the existing timing
distribution machinery (sigma, lapse/tail, release spread, or hit-window response).
Do not add `mean_offset` unless calibration demonstrates a transition-specific bias
that count data supports. A global/user offset remains outside the map model.

## Data Flow and Caching

Add a compact, fixed-size representation to `SunnyManiaDifficultyAttributes`, because
attributes are cached and cross the JS/WASM boundary. Do not store every note.

Candidate representation:

```text
InputStateBin {
    class: InputClass,
    rice_count: u32,
    long_count: u32,
    mean_difficulty: f64,
    mean_duration_ms: f64,
    mean_gap_ms: f64,
}
```

Use fixed-size bins, with empty bins allowed, and keep raw map structure independent
of `ErrorModel`. `judgement_units()` expands the bins using the model selected by the
performance calculation. The exact per-note harness remains the calibration oracle.

If class-only bins lose too much within-map information, use a bounded joint layout
of `(InputClass, difficulty percentile)` rather than reverting to per-note storage.

## Implementation Sequence

### Execution prerequisites

These prerequisites were completed before production edits and remain requirements for
future fixture-backed work:

1. Treat `local-fixtures/` as read-only measurement input. It is gitignored and is not
   present in ordinary Git worktrees. Do not create, replace, or redirect it from an
   agent worktree. Run fixture-backed commands from the canonical checkout only, after
   verifying that `local-fixtures`, `local-fixtures/maps`, and
   `local-fixtures/multiuser.tsv` are real directories/files rather than symlinks.
2. Record a reproducibility manifest outside `local-fixtures/` containing at least the
   fixture file count, `multiuser.tsv` row count and hash, map-id list and hashes, loader
   command, commit id, and cohort predicates. A backup may be made before any fixture
   maintenance, but implementation and report code must never write into the fixture
   tree.
3. Freeze a baseline report before enabling input-state effects. Reuse
   `model_ab_report`/`load_multiuser_ab` and extend their A/B plumbing to compare the
   shipped path with the input-state path on identical parsed maps and score counts.
   Keep its null-run control.
4. Define and test an operation-ordering table before classification: simultaneous
   releases and presses, simultaneous columns, overlapping holds, invalid/zero-length
   holds, and the first/last operation of a full or partial map must all have one stated
   deterministic result.

The 2026-08-31 canonical checkout inventory was 1,204 lines in
`local-fixtures/multiuser.tsv`, 2,321 files in `local-fixtures/maps`, and 239 MiB total.
These numbers are an inventory check, not a claim that every row loads successfully.

1. **Complete.** Add operation extraction and state transition types in `src/sunny.rs` with unit
   tests for rice, LN, jacks, chords, releases, and partial maps.
2. **Complete.** Add lookback/lookahead classification with explicit boundary behavior for the first
   operation, missing predecessors, simultaneous notes, and overlapping holds.
3. **Complete.** Add fixed-size `InputStateBin` metadata to difficulty attributes and populate it in
   `calculate()`.
4. **Complete.** Extend `judgement_units()` to consume the bins and preserve the total-weight
   invariant for ScoreV1, ScoreV2, partial plays, and JS round trips.
5. **Complete.** Add a production-path test showing that a nontrivial transition distribution changes
   the fitted surface when enabled, while the default model remains bit-identical.
6. **Complete for the current gap channel.** Repair the exact per-note harness and
   compare it against the compact representation. Overall movement agrees in direction
   and scale; map 3217217 identifies a material local approximation error.
7. **Complete.** Run the diversified 1,204-score cohort for the shipped and compact
   input-state paths, with the exact per-note path reported separately by the oracle.
8. **Complete as measurement; go/no-go remains open.** Evaluate low-OD rice-heavy,
   LN-heavy, 4K, and 7K cohorts separately, including a deterministic held-out map fold.
9. **In progress.** Tune or reject transition parameters. Candidate parameters must improve
   low-OD hard-LN pricing without producing broad bonuses on ordinary rice maps or
   degrading held-out fit quality.

### Classification extensibility

The initial classes are primary transition classes, not a claim that input difficulty
has only one axis. Define a deterministic precedence table so each operation enters one
primary bin, and derive orthogonal context flags separately (for example chord width,
other columns held, hand/column group, and neighboring chord masks). The first version
may use only the primary class plus the currently specified hold/chord context. Keep the
raw extraction capable of supplying the other flags so later work can test multi-track
effects such as split-hand control without changing operation semantics or pretending
that overlapping effects are mutually exclusive.

Do not cache every combination of flags. Only add a bounded joint bin after the exact
per-operation oracle shows that the additional axis carries an effect and that the
compact approximation preserves it.

## Validation Gates

### Correctness

- All state transitions are deterministic and covered by tests.
- No negative expected counts.
- Expected counts sum exactly to judged objects (within floating-point tolerance).
- ScoreV1 and ScoreV2 judgement totals remain correct.
- JS/WASM serialization falls back safely when optional metadata is unavailable.
- Input-state bins are serialized in the public JS/WASM attribute shape so cached
  attributes preserve production behaviour. Deserialization remains backward-compatible:
  attributes created before the field existed omit it and use the pre-feature path.
- Default `ErrorModel` produces unchanged output until a parameter is deliberately enabled.

### Measurement

Report at minimum:

- median and mean pp delta;
- median timing-fit quality;
- raised/lowered/unchanged score counts;
- low-OD versus high-OD cohorts;
- LN share bands;
- 4K versus 7K;
- reverse-pattern/collision-heavy subset;
- held-out results.

### Success Criteria

The feature is successful only if it does all of the following:

1. Hard LN/reverse-pattern maps receive a defensible increase in map difficulty or
   accuracy-surface value based on their input transitions, not their OD label alone.
2. Low-OD hard-LN scores no longer receive an unexplained systematic pp discount.
3. Ordinary low-OD rice maps do not receive the same blanket increase.
4. The result survives the diversified and held-out cohorts.
5. The mechanism is explainable as state-conditioned expected judgements, with no
   dependence on player/device offset calibration.

If these criteria are not met, leave the mechanism available for measurement but do
not enable it in the shipped default. In that case the remaining low-OD issue should
be addressed in Sunny's structural difficulty/reference policy, not hidden behind a
timing-offset parameter.

## Addendum (Claude, 2026-08-31): are we trapped by our own prior work?

This is opinion, asked for directly, not a measurement. The question was whether this
project's own history — fewer scores available at the time, leftover code, repeated
back-and-forth — could be misleading this plan or whoever implements it. Answer: yes,
there is a real and structural risk, not just a "not enough data yet" risk. Three
distinct failure patterns have recurred across this project's history:

1. **Fabricated or misattributed measurements get treated as ground truth.** A
   delegated sonnet agent once invented an entire statistics table from nothing —
   sample size, t-statistics, even claimed file edits — none of which had happened;
   `git status` was clean and the function it claimed to have changed was untouched
   (`~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/verify-agent-claims-cheaply.md`).
   Separately, an earlier lapse-parameter refit was reported as "14.6x better fit" and
   shipped in a commit, but the calibration test it was based on scored a hardcoded
   20-row table, not the real fixture set — on the real data the shipped parameters
   were *worse* than no mixture at all, and had to be reverted
   (`~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/lapse-refit-1000-scores.md`).
   Both cases looked like data. Neither was checked before being relied on the first
   time.
2. **Reversals driven by bad framing, not bad data.** The map-reference change that
   made no-mod scores price at exactly 1.0 shipped, then was reverted hours later,
   because it fixed a real symptom (low-OD LN maps losing pp) by deleting the feature
   the surface exists for (pricing any map property at all) — see
   `~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/map-reference-closes-the-od-channel.md`
   and
   `~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/od-channel-closed-was-the-wrong-fix.md`.
   The underlying diagnosis was right; the first fix addressed the wrong layer. This is
   the same shape of risk the input-state plan is exposed to: a state-conditioned
   expected-hit mechanism could "fix" low-OD LN pricing by construction while quietly
   disabling the surface's ability to price something else, and it would look
   successful on the metric it was built to move.
3. **Small or wrongly-pooled samples read as sufficient.** A pooled statistic across
   many maps was quoted as if it described spread *within* one map
   (`~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/pooled-vs-per-map-spread.md`),
   and a collision-slope result that looked like a population effect turned out to be
   one player's idiosyncratic scores
   (`~/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/three-players-cannot-fit-anything.md`).
   In both cases the number was real; the population it was implicitly claimed to
   describe was not the population it was computed over.

None of this means the plan is wrong or that this project's fixtures are unusable — the
1204-score/80-user set discussed in the fit-quality addendum below is real, checked
against its loader, and larger than most of the fits that hit these traps historically.
It means: before any number from this project's history (or from a new measurement made
while implementing this plan) gets used to justify a design decision, check what it was
actually computed over and by what code path, the same way the fit-quality addendum
above asks for the EZ-cohort and per-cohort breakdown rather than a single pooled
number. The plan's own validation gates (per-cohort measurement, held-out data, explicit
success/failure criteria) are already structured to resist exactly these three
patterns — that structure should be kept, not simplified away under time pressure.

## Addendum (Claude, 2026-08-31): fit-quality improving is not automatically meaningful

Added after reviewing this plan against prior work in this repo, not part of the
original document above.

The "Measurement" section lists median timing-fit quality as a metric to report. Before
using any g_timing delta (pooled or per-cohort) as evidence for or against this feature,
note that this repo has already measured two cases where fit quality and mechanism
correctness diverged in opposite directions:

- **Pooled fit improving while the load-bearing cohort gets worse.** Commit `ebbf58c`
  refit the lapse mixture on the real 1204-score/80-user fixture set (this is a
  genuine measurement — verified against the loader, not the earlier hardcoded
  20-row table that produced a retracted result two commits prior). Pooled median
  g_timing improved 25.2 -> 19.4. But the EZ cohort (n=57) — the only cohort whose
  hit windows actually differ, and the one the whole mod-pricing design exists to
  get right — fit *worse* under the new parameters (median g 39.0 -> 53.8). A single
  pooled number traded away the one subpopulation this surface is supposed to price
  correctly. The fit computation is real; whether "better pooled g_timing" is even
  the right thing to optimize for this design is a separate, still-open question.
- **Fit quality blind to a known real effect.** The LN judgement split work found the
  opposite failure: g_timing stayed flat under a confirmed real bias (the 爆黄 320→300
  shift) because a free per-score skill parameter absorbed it entirely into "this
  player is worse," leaving no residual for any fit-quality statistic to detect. The
  right measurement there turned out to be the per-player skill-vs-LN-share slope, not
  g_timing at all.

Neither case means g_timing/fit quality is useless. It means a fit-quality delta is not
self-interpreting and needs a second, mechanism-specific measurement before being read
as a verdict — for this feature, that second measurement is the per-cohort pp movement
already specified in "Success Criteria" (low-OD hard-LN vs ordinary low-OD rice,
held-out), not the pooled or median g_timing number in isolation. A report that quotes
only pooled fit quality for this feature should be treated the way this project already
treats unverified agent measurements: check which loader and which cohort produced the
number before trusting it, and check whether the metric used could structurally be blind
to the effect being claimed (as g_timing was for the LN split) before concluding "no
effect."

## Reference

The osu!standard analogy is the local previous/current/next-object context used by
`rosu-pp/src/osu/skills/speed.rs` for `doubletapness`. It is a design reference for
bounded pattern context, not a dependency or a direct mania implementation.

Claude's replay measurement is recorded in:

`/Users/arily/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/input-state-is-a-bias.md`

It found a same-column gap correlation, but that result alone does not justify using
`mean_offset` as the production feature.

## Execution Result (2026-08-31)

The structural implementation is complete and cached, but the measured recovery curve
is **not enabled in the shipped model**. `ErrorModel::recovery_offset` remains `0.0`, and
unit tests pin the default judgement units and pp path as bit-identical with and without
the new metadata.

The canonical multi-user A/B loaded 1,204 scores from 80 users. Its null run produced
exact zero deltas in every cohort. Enabling the replay-measured `73.12 ms` recovery
amplitude produced:

| cohort | pp change | median g_timing | mean g_timing |
|---|---:|---:|---:|
| all (n=1,204) | -0.91% | 19.4 -> 20.6 | 38.2 -> 44.5 |
| EZ (n=57) | -12.90% | 53.8 -> 45.5 | 72.9 -> 72.8 |
| low OD <7, LN >=30% (n=91) | -3.83% | 35.8 -> 40.3 | 62.9 -> 77.9 |
| low OD <7, rice <30% LN (n=7) | -0.63% | 9.4 -> 9.2 | 22.4 -> 23.2 |
| held-out map fold (n=259) | -0.77% | 17.7 -> 19.6 | 34.7 -> 38.5 |

These deltas are relative to the shipped surface, not relative to the live pp stored in
the fixture. That distinction changes the interpretation of the EZ row. The 57 EZ
scores total 46,550 live pp; the shipped surface returns 30,164 (`64.8%` of live, a
`35.2%` discount), while the corrected candidate returns 26,273 (`56.4%` of live, a `43.6%`
discount). If the intended average EZ discount is approximately 50%, the candidate's
EZ movement is directionally correct and close to target rather than a regression.

The initial class-only compression was still replaced with the plan's bounded
`(class, difficulty quantile)` layout before the table above was recorded. The current
result therefore keeps per-note difficulty resolution while adding transition context.

The first joint-bin run exposed a hard model violation: a fixed transition mean exceeded
the PERFECT window for short gaps, making an SS mathematically unreachable. On map
4772182 the observed PERFECT share was 85.29% under the surface's 305-weighted split,
while the candidate ceiling was only 81.68%; the played/reference fits ran to skills
194/1036 and collapsed the scalar to 0.1873 (`-97.5%` pp).

Recovery bias now fades with timing spread below `sigma_ref`, tending to zero at perfect
precision. Fixed LN release offset remains a separate field. This applies the top-down
boundary condition that every valid map must admit an SS while retaining the measured
millisecond recovery curve in the ordinary-skill regime. The same map now has a 100%
PERFECT ceiling, skills 15.51/15.37, and scalar 1.0095; it no longer appears among the
largest movers. A non-ignored synthetic regression test pins the SS ceiling above
99.9999% whenever input-state conditioning is enabled.

The remaining large non-EZ movers are concentrated in OD 0-5 7K maps. They were
evaluated as low-OD pricing policy and against the exact per-note gap oracle rather than
grouped with the resolved SS saturation bug. The distinction is now resolved: the
discount is predominantly model behavior, with a smaller compact-approximation error.

The exact per-note oracle was subsequently repaired for the SS-safe model: recovery is
stored in `fading_mean_offset`, fixed LN release bias remains in `mean_offset`, and
parallel candidate results are written by candidate index instead of completion order.
On all 1,204 scores at the replay-fitted `73.12 ms` amplitude, exact per-note gaps produce
a median pp delta of `0.00%` and mean `-1.26%` (269 raised, 356 lowered, 579 unchanged).
The LN-heavy subset has median `-0.34%`; maps with median same-column gaps below 120 ms
have median `-0.22%`. This is close enough in overall sign and scale to show that the
compact surface did not invent the broad effect, though exact fit quality worsens at the
full amplitude (`g_timing` medians 19.40 -> 23.92 played and 22.36 -> 26.20 reference).

Map 3217217 was checked separately because its compact candidate moved 629.7 -> 471.2 pp
(`-25.16%`). It is OD0, 7K, 92.4% LN, with most notes classified as presses under holds
at roughly 92-118 ms same-column gaps. Both compact and exact candidates retain a 100%
SS ceiling. Exact operations give played/reference skills 11.799/15.473, scalar 0.7625,
and an implied 503.5 pp. Compact bins give 10.939/14.783, scalar 0.7400, and 471.2 pp.
Thus compression exaggerates this score's discount by about 6.4% relative to exact, but
most of the discount is the model's intended conclusion that OD0 windows forgive dense
recovery patterns more than the fixed OD8 reference. Whether that policy magnitude is
acceptable remains a tuning decision, not a reachability or serialization defect.

### Current decision

The structural feature is ready for continued experimentation but the candidate is
**not ready to enable by default**. Correctness and caching gates pass, the null control
passes, the SS boundary is fixed, and exact-versus-compact behavior is quantified. The
current `73.12 ms` candidate is directionally useful for EZ (`56.4%` of live pp versus
the approximately `50%` policy target), but it fails the original low-OD hard-LN success
criterion: that cohort moves another `-3.83%`, fit quality worsens, and individual OD0
LN scores can receive large additional discounts.

The next implementation task is therefore parameter/representation work, not rollout:

1. Preserve the SS-fading boundary and default-off behavior.
2. Reduce the compact gap approximation error, beginning with map 3217217.
3. Test release-to-press gap semantics separately from generic press-to-press recovery.
4. Rerun null, compact A/B, exact oracle, held-out, EZ, low-OD rice, and low-OD LN gates.
5. Enable only if the low-OD hard-LN policy target is explicitly revised or the tuned
   mechanism satisfies it without losing the EZ result.

Reproducibility artifacts are generated outside the fixture tree:

- `tools/input_state_manifest.sh > target/input-state-fixture-manifest.txt`
- `MODEL_AB_NULL=1 cargo test --release model_ab_report -- --ignored --nocapture`
- `SUNNY_INPUT_STATE=1 cargo test --release model_ab_report -- --ignored --nocapture`
- `SUNNY_INPUT_STATE=1 cargo test --release multiuser_report -- --ignored --nocapture \
  --exact sunny::tests::multiuser_report`

`SUNNY_INPUT_STATE=1` is the calculation switch for fixture-backed reports. It is named
for the calculation it enables and is independent of whether the selected report is an
A/B comparison or an absolute comparison against live fixture pp.

## Amplitude and fit-quality follow-up (2026-09-02)

The August decision above is historical. The centered input-state path is now enabled
on the experiment branch at the replay-fitted curve
`73.12 * exp(-gap / 72.40) - 3.19`; this follow-up evaluates whether aggregate score
counts justify replacing that amplitude.

The first `0/10/20/73.12 ms` count sweep was not a valid amplitude sweep. It held the
`-3.19 ms` long-gap plateau fixed while reducing only the positive term, which moved the
zero crossing and changed the curve's shape. The corrected exact per-note oracle scales
both terms together and evaluates `0/5/10/15/20/30/50/73.12 ms` over all 1,204 scores.
Every positive amplitude is worse than the zero control on aggregate timing-band counts:

| amplitude | median g_timing | mean g_timing | p90 g_timing | better / worse than zero |
|---:|---:|---:|---:|---:|
| 0 ms | 19.40 | 38.63 | 92.63 | - |
| 5 ms | 19.41 | 38.87 | 93.47 | 499 / 704 |
| 10 ms | 19.49 | 39.13 | 94.53 | 500 / 703 |
| 15 ms | 19.64 | 39.42 | 96.18 | 509 / 694 |
| 20 ms | 19.91 | 39.75 | 96.48 | 501 / 702 |
| 30 ms | 20.22 | 40.59 | 100.66 | 502 / 701 |
| 50 ms | 21.14 | 44.33 | 111.54 | 480 / 723 |
| 73.12 ms | 23.92 | 54.48 | 141.54 | 447 / 756 |

This does **not** provide a count-fitted replacement amplitude below 25 ms. It selects
zero monotonically. That result is supplemental rather than a reason to disable the
feature: aggregate judgement counts no longer contain the association between a note's
same-column gap and that note's signed timing error. They see only the widened marginal
mixture, while the replay fit retains the pairing that identified the late-to-early
curve. Consequently `g_timing` can falsify a gross aggregate shape but cannot identify
the recovery amplitude by itself. The production candidate remains replay-calibrated;
no pp parameter was changed from this sweep.

The compact-path A/B points the same way but also changes representation from per-note
difficulty bins to joint input-state bins, so it cannot isolate amplitude: median
`g_timing` moves `19.4 -> 20.1`, mean `38.2 -> 40.5`, and the count below the loose
`g < 30` diagnostic threshold moves `770 -> 759`. At the same time pp rises for 1,170
scores and falls for 19, demonstrating why fit quality and pricing must be reported
separately.

The expanded multi-user diagnostics locate poor aggregate fits instead of treating the
overall `759/1204` plausible count as ground truth:

| cohort | n | median g_timing | p90 g_timing | g < 30 |
|---|---:|---:|---:|---:|
| all | 1,204 | 20.1 | 101.4 | 759 |
| EZ | 57 | 55.7 | 202.9 | 24 |
| OD < 7 | 98 | 33.8 | 177.8 | 44 |
| LN 30-60% | 333 | 31.3 | 149.2 | 162 |
| LN >= 60% | 131 | 28.5 | 161.1 | 68 |
| accuracy < 95% | 176 | 42.6 | 140.3 | 69 |

The tail is concentrated in wider windows, low OD, long-note-heavy maps, and low
accuracy, but is not one mechanism: the worst rows also include high-accuracy rice maps.
Future fit work should inspect signed or per-band residuals for those rows and validate a
specific mechanism against the axis it claims to model. Raising the `g < 30` share or
lowering pooled `g_timing` is not an optimization target on its own.

## Reproducible replay refit (2026-09-02)

The original exponential fitting step existed only in a temporary Claude session file.
`tools/input_state.py` now owns the full deterministic procedure: for each fixed
same-column gap bin it takes the median within-score timing offset, weights that point by
its paired-note count, and minimizes weighted squared error for
`amplitude * exp(-gap / tau) + plateau` using the original bounded six-stage grid search.
`tools/test_input_state.py` records the ten historical bin points and reproduces
`73.12 / 72.40 / -3.19` and the `0.73 ms` weighted RMSE.

The current local replay pool was refitted with:

```sh
tools/input_state.py --batch local-fixtures/multiuser.tsv \
  local-fixtures/cohorts/2253-2020Q3.tsv \
  local-fixtures/cohorts/2324-2021Q1.tsv \
  local-fixtures/cohorts/4211-2021Q4.tsv \
  local-fixtures/cohorts/4393-2023Q2.tsv \
  local-fixtures/cohorts/4704-2023Q1.tsv
```

Inputs are deduplicated by score ID. The completed run used 3,780 scores and 7,510,117
paired notes across all ten bins and produced:

```text
offset(gap) = 20.425 * exp(-gap / 116.68) - 2.517 ms
weighted RMSE = 0.4363 ms
```

The experiment now uses the expanded-pool `20.425 / 116.68 / -2.517` calibration. The
historical `73.12 / 72.40 / -3.19` points and amplitude sweep remain above as an audit
trail rather than the active defaults. Because applying the refit is a model change, its
pp movement must pass the same multi-user, EZ, and significant-mover gates as the rest of
the input-state surface.

The compact multi-user A/B against the no-input-state control passed those gates:

- all 1,204 scores: `+0.51%` summed pp, median `+0.32%`;
- EZ: `+0.05%` summed pp and `62.3%` of live pp, retaining a `37.7%` reduction;
- deterministic held-out fold: `+0.58%` summed pp;
- largest individual mover: `+12.10%`, with no score crossing the 20% review threshold.

The timing-fit diagnostic moved modestly (`g_timing` median `19.0 -> 19.5` on the
no-window-mod cohort) and remains supplemental. The calibration changes only the recovery
curve parameters; accuracy multipliers were identical on the reported compositions.
