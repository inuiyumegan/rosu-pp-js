# Compositional PP Architecture

## Summary

Implemented a compositional architecture that separates pattern difficulty from timing difficulty, replacing the multiplicative scalar approach with additive components.

## What Changed

### Before (Multiplicative)
```
pp = (stars * window_scalar)^2.2 * proportion * variety * acc_mult * length
```

Problems:
- Everything collapsed into scalars
- Low OD maps got inflated pp (window_scalar ~1.0 but skill fitted through wider windows)
- Impossible to see what contributed what
- Double-counting: accuracy affected both proportion and acc_mult

### After (Compositional)
```
pp = pp_pattern + pp_timing

where:
  pp_pattern = 9.8 * stars^2.2 * variety * length
  pp_timing = baseline_skill * skill_ratio * window_difficulty * acc_penalty
```

## Components

### 1. Pattern PP (from sunny)
- Base difficulty from rebirth algorithm (pattern complexity)
- Variety multiplier (jack vs stream balance)
- Length multiplier (note count scaling)
- **Does not include accuracy** - that's handled separately

### 2. Timing PP (from accuracy surface)
- **baseline_skill**: Fitted timing precision through natural windows (no mods, no input-state)
- **skill_ratio**: `played_skill / baseline_skill` - captures input-state effects and mod window changes
- **window_difficulty**: `(40.5 / window_great)^0.4` - penalizes wider windows (low OD)
- **acc_penalty**: `weighted_acc^12` - steep curve that heavily penalizes low accuracy

### 3. Accuracy Multiplier (from sunny)
- Applied to BOTH components
- Based on `acc_scalar = 0.5 * spikiness + 0.5 * switches`
- Ensures spiky/switchy maps penalize low acc more

## How It Addresses Your Requirements

✅ **Input-state machine works**: Recovery offset difference captured in `skill_ratio`

✅ **No mod post-processing**: EZ/HR baked into the windows during fitting, then ratio applied

✅ **Naturally penalizes bad acc**: `acc_penalty` uses steep 12th power curve

✅ **LN maps treated fairly**: Input-state (recovery offset) operates through skill_ratio, independent of OD

✅ **Doesn't conflict with sunny**: Pattern pp from sunny unchanged, timing pp is additive on top

## New Fields Exposed

```rust
pub struct SunnyManiaPerformanceAttributes {
    pub pp: f64,                      // Total
    pub pp_pattern: f64,              // Pattern contribution
    pub pp_timing: f64,               // Timing contribution
    pub timing_skill_played: f64,     // Skill through actual windows
    pub timing_skill_baseline: f64,   // Skill through natural windows
    pub window_scalar: f64,           // Legacy ratio (kept for compatibility)
    // ... other fields unchanged
}
```

## Debugging Low OD Inflation

The new architecture makes the problem **visible**:

```
map      od   pp_pattern  pp_timing  total   issue
1234567  7    450.2       180.3      630.5   timing_pp too high!
1234567  8    450.2       120.5      570.7   reasonable
```

You can now tune `window_difficulty_factor` exponent (currently 0.4) to control how much low OD is penalized.

## Calibration Parameters

### Current Settings
```rust
window_difficulty_factor: ratio^0.4
  - OD 10 (32ms): +10% timing pp
  - OD 8 (40.5ms): baseline (1.0)
  - OD 5 (55.5ms): -15% timing pp

acc_penalty: weighted_acc^12
  - 100% acc: 1.00
  - 99% acc: 0.89
  - 95% acc: 0.54
  - 90% acc: 0.28

base_timing scale: 5.0
  - Controls absolute contribution of timing pp
  - Higher = more timing pp relative to pattern pp
```

### Tuning Recommendations

If **low OD still too high**:
- Increase window_difficulty exponent: `0.4` → `0.6`
- This makes OD penalty stronger

If **timing pp too dominant**:
- Decrease base_timing scale: `5.0` → `3.0`

If **acc not penalized enough**:
- Increase acc_penalty exponent: `12` → `15`

## Migration Notes

### Compatibility
- All existing fields preserved
- `window_scalar` still reported for legacy compatibility
- Tests pass (19/19)

### Deployment
- JS bindings updated with new fields
- TypeScript will see: `ppPattern`, `ppTiming`, `timingSkillPlayed`, `timingSkillBaseline`

## Next Steps

1. Run multiuser report to see pp_pattern vs pp_timing breakdown
2. Identify if low OD timing_pp is still too high
3. Tune window_difficulty exponent if needed
4. Potentially adjust base_timing scale for overall balance

## Testing

```bash
# Run all tests
cargo test --lib sunny::tests --release

# Check specific behavior
cargo test --lib sunny::tests::ez_is_priced_by_the_windows_not_a_multiplier --release
cargo test --lib sunny::tests::hr_is_rewarded_by_the_same_mechanism --release
```

All tests passing ✓
