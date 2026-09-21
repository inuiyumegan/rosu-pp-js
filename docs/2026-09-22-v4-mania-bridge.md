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

- `sunny-od8-deref@2026.9.22.1`
- `sunny-surface@2026.9.22.1`

The existing GitHub workflow creates release assets only on tags. After this
tag is pushed and the NodeJS tarball is available, update `osu-server-ts` to
consume that artifact.

## Release artifact requirement

A Git tag and GitHub source archive contain this Rust source only; neither is a
NodeJS package that `pnpm` can install. The server requires the NodeJS
`wasm-pack` output, published as the release asset
`rosu_pp_js_nodejs.tar.gz`.

To publish the existing immutable tag, dispatch the `CI` workflow with
`v2026.9.22-mania-surface.1` selected as the ref. The `build (nodejs)` job runs
`wasm-pack build --release --target nodejs --out-dir pkg`, archives `pkg` as
`rosu_pp_js_nodejs.tar.gz`, and the `release` job attaches it to the matching
GitHub Release. The first run can spend several minutes installing
`wasm-bindgen`; wait for every job to finish before checking the release page.

Do not move or recreate the existing tag merely to retry publication. A
workflow dispatch against that tag preserves the immutable source/version
mapping while giving the release job `refs/tags/v2026.9.22-mania-surface.1`.

## Server algorithm identifiers

The corresponding immutable test-release algorithm IDs in `osu-server-ts` are:

- `sunny-od8-deref@2026.9.22.1`
- `sunny-surface@2026.9.22.1`

## Packaging retry: .2

The .1 release build reached wasm-opt but its default feature set rejected
bulk-memory and non-trapping float-to-int instructions emitted by the current
Rust toolchain. The .2 package keeps wasm optimization enabled and passes the
corresponding wasm-opt feature flags: --enable-bulk-memory and
--enable-nontrapping-float-to-int.

The pinned rosu-pp revision and all bridge calculation code are unchanged. This
is a packaging retry only, so osu-server-ts continues to use the immutable
algorithm IDs ending in @2026.9.22.1.
