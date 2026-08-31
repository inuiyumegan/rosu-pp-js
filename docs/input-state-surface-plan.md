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
           -> SunnyManiaDifficultyAttributes
score counts + attributes -> judgement_units -> timing fit
                         -> played/reference scalar -> pp
```

Already present:

- `Note { column, head, tail }` extraction in `src/sunny.rs`.
- Per-note difficulty compressed into 16 equal-count `NoteDifficultyBin`s.
- LN duration and release spread modelling in `src/mania_accuracy.rs`.
- A six-category accuracy surface fitted from fractional judgement units.
- A test-only exact same-column-gap experiment.
- `JudgementUnit::mean_offset` and `ErrorModel::recovery_mean_offset`, which are
  experimental conditional-bias tools, not the state-machine implementation.

Missing:

- Production input operations and persistent per-column state.
- Lookback/lookahead context carried from difficulty calculation into performance.
- State-conditioned fractional expected hit distributions in production.
- Validation that the public `calculate_performance` path, rather than only an
  ignored harness, uses this information.

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

1. Add operation extraction and state transition types in `src/sunny.rs` with unit
   tests for rice, LN, jacks, chords, releases, and partial maps.
2. Add lookback/lookahead classification with explicit boundary behavior for the first
   operation, missing predecessors, simultaneous notes, and overlapping holds.
3. Add fixed-size `InputStateBin` metadata to difficulty attributes and populate it in
   `calculate()`.
4. Extend `judgement_units()` to consume the bins and preserve the total-weight
   invariant for ScoreV1, ScoreV2, partial plays, and JS round trips.
5. Add a production-path test showing that a nontrivial transition distribution changes
   the fitted surface when enabled, while the default model remains bit-identical.
6. Keep the exact per-note harness and compare it against the compact representation;
   quantify approximation error before tuning parameters.
7. Run the diversified 1,204-score cohort in parallel, reporting baseline,
   per-note-only, and input-state-conditioned results.
8. Evaluate low-OD rice-heavy, LN-heavy, 4K, and 7K cohorts separately. Use held-out
   scores/maps for the go/no-go decision.
9. Only then consider fitting transition parameters. Candidate parameters must improve
   low-OD hard-LN pricing without producing broad bonuses on ordinary rice maps or
   degrading held-out fit quality.

## Validation Gates

### Correctness

- All state transitions are deterministic and covered by tests.
- No negative expected counts.
- Expected counts sum exactly to judged objects (within floating-point tolerance).
- ScoreV1 and ScoreV2 judgement totals remain correct.
- JS/WASM serialization falls back safely when optional metadata is unavailable.
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

## Reference

The osu!standard analogy is the local previous/current/next-object context used by
`rosu-pp/src/osu/skills/speed.rs` for `doubletapness`. It is a design reference for
bounded pattern context, not a dependency or a direct mania implementation.

Claude's replay measurement is recorded in:

`/Users/arily/.claude/projects/-Users-Shared-git-ppy-sb-rosu-pp-js/memory/input-state-is-a-bias.md`

It found a same-column gap correlation, but that result alone does not justify using
`mean_offset` as the production feature.
