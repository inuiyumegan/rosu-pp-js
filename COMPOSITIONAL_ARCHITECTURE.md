# Compositional PP Architecture

## Current Formula

The implementation separates the accuracy-neutral pattern value from one merged
accuracy-surface adjustment:

```text
pp = pp_pattern + pp_timing

pp_pattern = sunny_pattern_value * variety * length * fail_multiplier
accuracy_reward = sunny_accuracy_reward * surface_transfer^2.2
pp_timing = pp_pattern * (accuracy_reward - 1)
```

`pp_timing` is signed. It is the combined adjustment for accuracy, played hit
windows, and the experimental input-state model; it is not an independent pool of
timing pp.

This prevents the surface from creating a second full-sized pp contribution while
keeping the scoring composition observable.

## Surface

```text
surface_transfer = played_skill / baseline_skill
```

- `played_skill` is fitted through the actual hit windows and selected per-note
  judgement units.
- `baseline_skill` uses the map's natural hit windows and the same note population.
- EZ widens the played windows and moves the transfer below one.
- HR narrows the played windows and moves the transfer above one.
- There is no fixed OD8 reference, explicit mod multiplier, separate LN scale, or
  hand-authored accuracy penalty.

When input-state recovery is enabled experimentally, both fits use input-state
bins. Recovery is disabled only on the baseline side. This avoids mistaking a
change from input-state units to fallback LN units for timing skill.

## Accuracy Ownership

Sunny's `performance_proportion` and `acc_multiplier` remain as the provisional
absolute-accuracy reward. The surface currently adjusts that reward rather than
adding another reward on top.

This is an explicit migration boundary. Once the per-note surface has a calibrated
absolute reward, it should replace the two Sunny accuracy factors at this point.
They must not remain alongside a complete surface-derived accuracy reward, which
would double-count the same hit results.

## Exposed Components

```rust
pub struct SunnyManiaPerformanceAttributes {
    pub pp: f64,
    pub pp_pattern: f64,
    pub pp_timing: f64,
    pub timing_skill_played: f64,
    pub timing_skill_baseline: f64,
    pub window_scalar: f64,
    // existing fields omitted
}
```

`pp_pattern` and `pp_timing` add exactly to total pp. The fitted skills and legacy
`window_scalar` remain available for diagnostics.

## Validation

Run the invariant tests before cohort measurement:

```bash
cargo test --lib sunny::tests::ez_is_priced_by_the_windows_not_a_multiplier
cargo test --lib sunny::tests::hr_is_rewarded_by_the_same_mechanism
cargo test --lib sunny::tests::input_state_surface_is_effective_by_default_and_can_be_disabled
```

Then run the fixture report with production defaults:

```bash
cargo test --release multiuser_report -- --ignored --nocapture
```

The experiment branch enables the replay-fitted input-state recovery curve by
default. Its offsets are centered over each map's press population because the
replay analysis measured every state relative to the score's own mean error. This
keeps the per-note state distribution in the fit without turning map composition
into a global clock offset. Use the baseline branch for the A/B control.
