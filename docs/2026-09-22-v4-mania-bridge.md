# 2026-09-22: public v4 mania bridge

## Why

`rosu-pp-js` main binds `rosu-pp` 3.1, whereas the new mania work lives in
`rosu-pp` 4.0. Updating the old generic binding in place couples this release
to unrelated v4 API changes.

## Release

- Git tag: `v2026.9.22-mania-surface.1`
- Cargo/package version: `2026.9.22-mania-surface.1`
- Upstream public API revision: `rosu-pp` `3530ba72`

## Public bridge contract

The new package is mania-only and exposes a package-owned `Beatmap` plus:

- `RebirthManiaDifficulty` and `RebirthManiaPerformance`
- `SunnyManiaDifficulty` and `SunnyManiaPerformance`

Rebirth intentionally calls the public default `Difficulty` and `Performance`
APIs. Sunny intentionally calls the explicit public
`mania::sunny::{calculate, calculate_performance}` APIs, which include the
new timing surface. No algorithm implementation is copied into this package.

The Node/Wasm package must use its own `Beatmap`; Wasm objects from the legacy
v3 package cannot cross into this module.

## Validation

- `cargo check --manifest-path Cargo.toml`
- `cargo test --manifest-path ../rosu-pp/Cargo.toml --lib mania::sunny -- --nocapture`
  - 93 passed, 0 failed, 37 fixture/report tests ignored

## Server integration after the release asset exists

Register two mania-only algorithms, keeping their score PP rows distinct:

- `sb-mania-rebirth@2026.9.22-mania-surface.1`
- `sunnyxxy-mania@2026.9.22-mania-surface.1`

The existing GitHub workflow creates release assets only on tags. After this
tag is pushed and the NodeJS tarball is available, update `osu-server-ts` to
consume that artifact.
