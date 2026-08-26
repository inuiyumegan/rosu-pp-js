//! Star-Rating-Rebirth ("sunny") algorithm port for osu!mania.
//!
//! This module implements the community SR/PP algorithm by [Crz]sunnyxxy
//! (https://github.com/sunnyxxy/Star-Rating-Rebirth) as integrated into
//! osu!lazer on the `author-port` branch of `vernonlim/osu`.
//!
//! Notable differences to the upstream `rosu-pp` rebirth implementation:
//! - The hit leniency `x` is derived from the GREAT hit window which takes
//!   the EZ / HR mods into account (`EZ` widens the window by 1.4, `HR`
//!   shrinks it by 1.4) as well as the convert-specific base windows.
//! - The "switches" measure weights corners by the effective weights
//!   (`density * gap`) instead of the raw difficulty values.

use std::cmp::Ordering;
use std::collections::HashMap;

use rosu_mods::{Acronym, GameMods};
use rosu_pp::model::{
    beatmap::Beatmap,
    hit_object::{HitObject, HitObjectKind},
    mode::GameMode,
};

use crate::mania_accuracy::{
    fit_with_quality, ErrorModel, JudgementUnit, LN_DURATION_BUCKETS,
};
use crate::mania_windows::{hit_windows, ManiaHitWindows};

/// The upper edges, in ms, of the first [`LN_DURATION_BUCKETS`] - 1 duration bins;
/// anything longer falls in the last.
///
/// These are a **quadrature grid, not a taxonomy**. The release-spread model is a
/// continuous function of hold duration
/// ([`crate::mania_accuracy::release_ratio_for_duration`]); the bins exist only because
/// [`SunnyManiaDifficultyAttributes`] is `Copy` and cannot carry a per-note duration
/// list. Each bin contributes one judgement unit evaluated at
/// [`LN_DURATION_REPRESENTATIVES`], so the bins approximate an integral rather than
/// asserting that a 59 ms hold and a 61 ms hold are different kinds of object.
///
/// Log-spaced, because that is how the durations themselves are distributed — over 130k
/// long notes in the fixture set the deciles run 50 ms at p10, 100 ms at p50, 300 ms at
/// p90 and 894 ms at p99. Even spacing would put most notes in one bin and leave the
/// rest nearly empty, which is exactly where a quadrature rule loses accuracy.
///
/// The count and spacing are set by measurement, not taste.
/// `ln_binning_error_stays_small` compares the binned fit against evaluating every long
/// note at its own duration: a coarser five-bin grid let the spread multiplier vary up to
/// 32% *within* one bin and shifted fitted skill by 4.5%, which is the same order as the
/// effect being measured and therefore useless. These edges keep the within-bin variation
/// near 10% and the skill error under 2%.
pub const LN_DURATION_EDGES: [f64; LN_DURATION_BUCKETS - 1] =
    [45.0, 70.0, 100.0, 145.0, 210.0, 320.0, 550.0];

/// The duration, in ms, at which each bin's judgement unit is evaluated.
///
/// Geometric midpoints of the bins rather than arithmetic ones, matching the log spacing
/// of [`LN_DURATION_EDGES`]: for a quantity varying multiplicatively within a bin, the
/// geometric centre is far closer to the typical member than the arithmetic one. The
/// first bin's lower edge is taken as 25 ms rather than zero, since the fixture set's p1
/// is 22 ms, and the open top bin uses a representative near the observed p99 rather than
/// an unbounded midpoint.
pub const LN_DURATION_REPRESENTATIVES: [f64; LN_DURATION_BUCKETS] =
    [34.0, 56.0, 84.0, 120.0, 175.0, 259.0, 419.0, 900.0];

/// Which [`LN_DURATION_EDGES`] bin a long note of `duration` ms belongs to.
fn ln_duration_bucket(duration: f64) -> usize {
    LN_DURATION_EDGES
        .iter()
        .position(|&edge| duration < edge)
        .unwrap_or(LN_DURATION_BUCKETS - 1)
}

/// Every long note in the modal duration bucket, for callers that know how many long
/// notes a map has but not how long they are.
///
/// The fallback for cached attributes round-tripped through JS, where the histogram is
/// not part of the public shape. Approximate by construction: it prices a map of
/// half-second holds as if they were one-beat notes. Prefer passing the beatmap.
pub fn modal_ln_duration_histogram(n_long_notes: usize) -> [usize; LN_DURATION_BUCKETS] {
    let mut buckets = [0; LN_DURATION_BUCKETS];

    // The bin containing the fixture set's median long note (100 ms), which is the
    // least-wrong single choice when the real distribution is unavailable.
    let modal = LN_DURATION_EDGES
        .iter()
        .position(|&edge| 100.0 < edge)
        .unwrap_or(LN_DURATION_BUCKETS - 1);

    buckets[modal] = n_long_notes;

    buckets
}

/// Bucket long notes by how long they are held.
///
/// Durations come from [`Note`], whose times are already divided by the clock rate, so
/// these are the map's own durations rather than what the player experienced. That is
/// the right convention here for the same reason the hit windows are rate-normalised:
/// under `DT` a 100 ms hold arrives as 67 ms of wall-clock but the judgement windows
/// shrink to match, so the *ratio* of hold length to window — which is what decides
/// whether a release is a separate act — is unchanged. Bucketing on wall-clock instead
/// would make `DT` silently reclassify every long note as shorter.
fn ln_duration_histogram(long_notes: &[Note]) -> [usize; LN_DURATION_BUCKETS] {
    let mut buckets = [0; LN_DURATION_BUCKETS];

    for note in long_notes {
        let duration = note.tail_or_head() - note.head;

        if duration > 0.0 {
            buckets[ln_duration_bucket(duration)] += 1;
        }
    }

    buckets
}

/// A single mania note (or hold-note) extracted from a beatmap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Note {
    pub column: usize,
    pub head: f64,
    pub tail: Option<f64>,
}

impl Note {
    fn tail_or_head(self) -> f64 {
        self.tail.unwrap_or(self.head)
    }
}

/// The result of the sunny difficulty calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SunnyManiaDifficultyAttributes {
    /// The star rating.
    pub stars: f64,
    /// The variety measure (Rao quadratic entropy based).
    pub variety: f64,
    /// The accuracy scalar `0.5 * spikiness + 0.5 * switches`.
    pub acc_scalar: f64,
    /// How much the difficulty spikes within the map.
    pub spikiness: f64,
    /// How much the playstyle switches between jack and stream-like patterns.
    pub switches: f64,
    /// The GREAT hit window used for the calculation (incl. mods).
    pub great_hit_window: f64,
    /// The full judgement window set the score will be graded against.
    ///
    /// Mods are already folded in, which is what lets the performance stage price
    /// a mod without knowing it was used: `EZ` widens every window here, so the
    /// same judgement counts imply a lower skill and earn less. See
    /// [`compute_difficulty_value`].
    pub hit_windows: ManiaHitWindows,
    /// The max combo of the map.
    pub max_combo: u32,
    /// The amount of hit objects taken into account.
    pub n_objects: usize,
    /// How many of those hit objects are long notes.
    ///
    /// Read straight off the map, so it is structural input to the judgement model
    /// rather than anything inferred from a score. [`window_scalar`] uses it to split
    /// the map into rice and LN populations, since a ScoreV1 long note is judged on
    /// the sum of two offsets and so carries more timing spread than a press — see
    /// [`crate::mania_accuracy::ln_sigma_scale`].
    pub n_long_notes: usize,
    /// How those long notes are distributed over [`LN_DURATION_EDGES`] duration
    /// buckets, shortest first.
    ///
    /// A histogram rather than a mean, because LN duration spans nearly twenty-fold
    /// *within a single map* — measured over 130k long notes in the fixture set the
    /// deciles run 50 ms at p10, 100 ms at p50 and 894 ms at p99 — and a mean would
    /// put a chordjack's 50 ms taps in the same bucket as a half-second hold. Sums to
    /// [`Self::n_long_notes`].
    ///
    /// A fixed-size array because these attributes are `Copy`.
    pub ln_duration_buckets: [usize; LN_DURATION_BUCKETS],
    /// The map's own judgement windows with the window-affecting mods stripped.
    ///
    /// Identical to [`Self::hit_windows`] for a no-mod score, and narrower or wider than
    /// it under `HR`/`EZ`. Carried so [`window_scalar`] can price a score against the
    /// windows its own map would have given it, which confines the surface to pricing
    /// *mods* rather than also pricing the map's OD. See `REFERENCE_WINDOWS` for what the
    /// alternative costs.
    pub map_windows: ManiaHitWindows,
    /// Whether long notes give a single combined judgement (ScoreV1 / classic)
    /// rather than separate head and release judgements (ScoreV2).
    ///
    /// This is [`is_classic`] carried forward, because the judgement *count* depends
    /// on it: under V1 the map yields `n_objects` judgements, under V2 it yields
    /// `n_objects + n_long_notes`. Verified against 143 live scores — every V2 score
    /// totalled `notes + LN`, and every V1 score bar one totalled `notes`.
    pub ln_judged_as_one: bool,
}

/// The result of the sunny performance calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SunnyManiaPerformanceAttributes {
    /// The total performance points.
    pub pp: f64,
    /// The difficulty portion of the PP.
    pub pp_difficulty: f64,
    /// The variety multiplier applied to the difficulty portion.
    pub variety_multiplier: f64,
    /// The accuracy multiplier applied to the difficulty portion.
    pub acc_multiplier: f64,
    /// The length multiplier applied to the difficulty portion.
    pub length_multiplier: f64,
    /// How much the judgement windows in effect changed the score's value.
    ///
    /// Below 1 when the score was graded through windows wider than the OD 8
    /// reference, which is how `EZ` is priced without a mod-specific factor. Exactly
    /// 1 only when there was nothing to measure. See [`window_scalar`].
    pub window_scalar: f64,
}

/// Score state required for the performance calculation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SunnyScoreState {
    pub n320: u32,
    pub n300: u32,
    pub n200: u32,
    pub n100: u32,
    pub n50: u32,
    pub misses: u32,
}

impl SunnyScoreState {
    pub const fn total_hits(&self) -> u32 {
        self.n320 + self.n300 + self.n200 + self.n100 + self.n50 + self.misses
    }
}

/// Calculate the sunny difficulty attributes for a (converted) mania beatmap.
///
/// `mods` are the rosu-mods, `clock_rate` the custom clock rate, `lazer`
/// whether the play is a lazer (default) or stable play and `passed_objects`
/// the amount of objects to take into account for partial plays.
pub fn calculate(
    map: &Beatmap,
    mods: &GameMods,
    clock_rate: f64,
    lazer: Option<bool>,
    passed_objects: Option<u32>,
) -> Option<SunnyManiaDifficultyAttributes> {
    let total_columns = map.cs.round_ties_even().max(1.0) as usize;

    let has_hr = has_mod(mods, "HR");
    let has_ez = has_mod(mods, "EZ");

    let great_hit_window = get_hit_window_300(map, clock_rate, has_hr, has_ez);
    let hit_leniency = hit_leniency_from_window(great_hit_window);
    let classic = is_classic(lazer, mods);
    let windows = hit_windows(map, mods, clock_rate, classic);

    // The same map judged without the window-affecting mods. Mods reach `hit_windows`
    // only through its difficulty multiplier, so an empty mod set is exactly "this map's
    // own windows". The clock rate stays as passed because the windows are rate-normalised
    // anyway, and `classic` stays because it describes the scoring scheme rather than a
    // mod's effect on leniency.
    let map_windows = hit_windows(map, &GameMods::default(), clock_rate, classic);

    let take = passed_objects.unwrap_or(u32::MAX) as usize;
    let objects = map.hit_objects.iter().take(take);

    let (notes, max_combo) = build_notes(clock_rate, objects, total_columns);

    if notes.len() < 2 || total_columns == 0 {
        return None;
    }

    let data = RebirthData::new(notes, total_columns, hit_leniency);
    let params = calculate_from_data(&data, classic)?;

    Some(SunnyManiaDifficultyAttributes {
        stars: params.sr,
        variety: params.variety,
        acc_scalar: 0.5 * params.spikiness + 0.5 * params.switches,
        spikiness: params.spikiness,
        switches: params.switches,
        great_hit_window,
        hit_windows: windows,
        map_windows,
        max_combo,
        n_objects: data.notes.len(),
        n_long_notes: data.long_notes.len(),
        ln_duration_buckets: ln_duration_histogram(&data.long_notes),
        ln_judged_as_one: classic,
    })
}

/// Calculate the sunny performance attributes.
pub fn calculate_performance(
    attrs: &SunnyManiaDifficultyAttributes,
    mods: &GameMods,
    state: SunnyScoreState,
) -> SunnyManiaPerformanceAttributes {
    // NF still gets a flat factor: failing is a scoring matter that the timing
    // surface says nothing about, so there is nothing for it to price. EZ has no
    // factor here on purpose — see `compute_difficulty_value`.
    let mut multiplier = 1.0;

    if has_mod(mods, "NF") {
        multiplier *= 0.75;
    }

    let score_accuracy = custom_accuracy(state);
    let window_scalar = window_scalar(attrs, state);
    let difficulty_value = compute_difficulty_value(attrs.stars, score_accuracy, window_scalar);
    let variety_multiplier = variety_multiplier(attrs.variety);
    let acc_multiplier = acc_multiplier(score_accuracy, attrs.acc_scalar);
    let length_multiplier = length_multiplier(attrs.n_objects as f64, attrs.stars);

    let pp = difficulty_value
        * multiplier
        * variety_multiplier
        * acc_multiplier
        * length_multiplier;

    SunnyManiaPerformanceAttributes {
        pp,
        pp_difficulty: difficulty_value,
        variety_multiplier,
        acc_multiplier,
        length_multiplier,
        window_scalar,
    }
}

/// OD 8 classic non-convert, the modal mania OD.
///
/// No longer the pricing reference — see [`reference_windows`] for why the map's own
/// windows replaced it. Still the fixed yardstick the calibration harnesses fit against,
/// where a constant is what is wanted so that fit quality across maps is comparable, and
/// still reachable for pricing via `SUNNY_FIXED_REFERENCE`.
///
/// A literal because it must be `const`; `reference_windows_match_od8_no_mod` pins
/// it against [`hit_windows`] so the two cannot drift.
const REFERENCE_WINDOWS: ManiaHitWindows = ManiaHitWindows {
    perfect: 16.5,
    great: 40.5,
    good: 73.5,
    ok: 103.5,
    meh: 127.5,
    miss: 164.5,
};

/// The windows a score is priced *against*, which decides what the surface charges for.
///
/// **The map's own windows**, which confines the surface to pricing *mods*: every no-mod
/// score prices at exactly 1.0 at any OD or keymode, and a mod is charged for how far it
/// moves the windows away from what the map itself asked for.
///
/// Three candidates were measured against the same 143 live scores; the alternatives are
/// kept behind env switches so the comparison can be rerun in one build, the way
/// `SUNNY_NO_LN_SPLIT` is kept.
///
/// - **Fixed [`REFERENCE_WINDOWS`]** (OD 8, `SUNNY_FIXED_REFERENCE`) says a low-OD map is genuinely more lenient,
///   so a score on it demonstrates less precision and should earn less. That claim is
///   very hard to defend in mania, where OD is a charting convention rather than a
///   difficulty setting: 7K charts in the fixture set average OD 4.8 against 4K's 8.2,
///   and 7K LN maps average OD 4.2. Under this reference those maps lose 16.6% of their
///   live pp for their OD alone.
/// - **One-sided** (`SUNNY_ONESIDED_REFERENCE`), the wider of the two per window: a map
///   stricter than OD 8 keeps its bonus, a map more lenient than OD 8 pays no penalty.
///   Asymmetric by construction, and the asymmetry is not merely convenient — the two
///   directions are not equally well evidenced. Above the reference the claim "these 320s
///   came through a 14.5 ms window, so this player was precise to better than 14.5 ms" is
///   directly witnessed by the counts. Below it, "these 320s came through a 20 ms window,
///   so this player was only precise to 20 ms" is *not* witnessed, because a 320 is
///   censored: a player who would have hit inside 16.5 ms anyway produces exactly the same
///   count as one who needed the full 20 ms. The surface can therefore detect precision
///   finer than the window it is given, but not coarser, and a one-sided reference is what
///   that asymmetry looks like when taken seriously. That censoring argument did not
///   survive testing: the low-OD 7K scores the fixed reference penalises average a 63.4%
///   320 share with none above 90%, so they are nowhere near the saturation the argument
///   needs. The asymmetry rests on the endogeneity of mania OD alone.
///
/// Measured against 143 live scores, as a fraction of live pp:
///
/// | group | fixed | one-sided | map |
/// |---|---|---|---|
/// | all (n=143) | −12.49% | −7.12% | −8.27% |
/// | 7K no-mod (n=51) | −13.94% | +0.30% | −0.21% |
/// | 4K no-mod OD≥8.9 (n=19) | +2.80% | +2.80% | +0.10% |
/// | EZ on OD≥8.1 (n=9) | −32.68% | −32.68% | −39.75% |
///
/// The map reference was chosen over the one-sided variant knowing it costs the high-OD
/// 4K bonus (+2.80% to +0.10%), because a symmetric rule is defensible to players in a way
/// "your OD only counts when it helps you" is not. It also repairs `EZ`: under a fixed
/// reference, `EZ`'s widening and a high-OD map's narrowing partly cancelled, so the same
/// mod cost 32.68% on high-OD maps and 39.06% on low-OD ones. Against the map's own
/// windows `EZ` costs the same everywhere (−39.75% / −39.18%), which is what pricing a
/// mod rather than a map means.
fn reference_windows(attrs: &SunnyManiaDifficultyAttributes) -> ManiaHitWindows {
    if std::env::var_os("SUNNY_FIXED_REFERENCE").is_some() {
        return REFERENCE_WINDOWS;
    }

    if std::env::var_os("SUNNY_ONESIDED_REFERENCE").is_some() {
        return REFERENCE_WINDOWS.widest_of(&attrs.map_windows);
    }

    attrs.map_windows
}

/// Whether `SUNNY_NO_LN_SPLIT` is set, which collapses the LN mixture back to a
/// single population.
///
/// An A/B switch for the reporting harnesses, not a feature: the LN split changes no
/// free parameters, so the only way to attribute a change in fit quality to it is to
/// price the same fixtures both ways in one build. Unset in every normal run,
/// including every unit test that pins the split's behaviour.
fn ln_split_disabled() -> bool {
    std::env::var_os("SUNNY_NO_LN_SPLIT").is_some()
}

/// The judgement units a score's counts are fitted against: the map split into a rice
/// population and, under ScoreV1, one long-note population per duration bin.
///
/// Local difficulty is still uniform at the map's star rating — per-note difficulty is
/// the separate, larger change — so the only structure here is the long notes. It
/// matters because an LN chart is a *mixture*: fitting one sigma to a mixture of widths
/// inflates it, which drives estimated skill down and, via the `^2.2` in pp, costs far
/// more than the widening itself. 7K charts in the fixture set average 58% long notes
/// against 4K's 3%, so this is where the two populations actually differ.
///
/// **Why duration bins and not one LN population.** A release is harder to place than a
/// press, and a *short* hold is harder still because the press motion has not finished
/// when the release is already due. Both effects live in
/// [`crate::mania_accuracy::release_ratio_for_duration`], which is continuous in
/// duration; the bins are the quadrature grid that lets a `Copy` attribute struct carry
/// it. Sweeping a single LN width instead wanted two different answers on mixed-LN and
/// LN-saturated maps, which is what forced duration into the model.
///
/// Under ScoreV2 heads and releases are judged separately, so every judgement is a
/// single press and there is no mixture; the units come back uniform and only the count
/// changes. `total` is the score's own judgement total, which the caller has already
/// measured, so the returned weights always sum to exactly what was observed even when
/// the map's structure and the score disagree.
///
/// Everything read here comes from the `.osu` and the mod list. Nothing about how well
/// the player did enters, which is the line that keeps a bad play from being re-read as
/// a hard map.
fn judgement_units(
    attrs: &SunnyManiaDifficultyAttributes,
    total: f64,
    model: &ErrorModel,
) -> Vec<JudgementUnit> {
    let uniform = vec![JudgementUnit::repeated(attrs.stars, total)];

    // Under V2 the head and release are two ordinary single-press judgements, so
    // there is no wide population to separate out.
    if !attrs.ln_judged_as_one || attrs.n_long_notes == 0 || attrs.n_objects == 0 {
        return uniform;
    }

    if ln_split_disabled() {
        return uniform;
    }

    // Work in shares of the score's own judgement total rather than in the map's raw
    // counts. A partial play, or a count vector that disagrees with our object parsing,
    // then still produces weights summing to the observed total, which is what the
    // multinomial fit requires.
    let per_object = total / attrs.n_objects as f64;

    let mut units = Vec::with_capacity(LN_DURATION_BUCKETS + 1);
    let mut ln_total = 0.0;

    for (bin, &count) in attrs.ln_duration_buckets.iter().enumerate() {
        if count == 0 {
            continue;
        }

        let weight = count as f64 * per_object;
        ln_total += weight;

        units.push(JudgementUnit::long_note(
            attrs.stars,
            weight,
            model,
            LN_DURATION_REPRESENTATIVES[bin],
        ));
    }

    // The histogram can undercount long notes relative to `n_long_notes` — a zero-length
    // hold contributes to one and not the other — so derive the rice weight from what
    // the bins actually consumed rather than from the LN count. This keeps the weights
    // summing to `total` regardless.
    let rice_units = (total - ln_total).max(0.0);

    if rice_units > 0.0 {
        units.push(JudgementUnit::repeated(attrs.stars, rice_units));
    }

    if units.is_empty() {
        return uniform;
    }

    units
}

/// How much the windows a score was played under change what it is worth.
///
/// This is where mods get priced, and it is the whole point of widening the windows
/// *before* grading the score. The same judgement counts are fitted twice: once
/// against the windows actually in effect, once against [`reference_windows`]. A
/// player who delivers a given 320 count through wider `EZ` windows demonstrably
/// hit less precisely, so the first fit returns a lower skill and the ratio falls
/// below 1. Nothing here inspects the mod list.
///
/// Deliberately *not* gated on [`ManiaFitQuality::is_plausible`]. The absolute fit is
/// still imperfect on many real scores even after the error model was given a proper
/// tail, but that error is largely common to both fits and so divides out of the
/// ratio. Gating on it made pricing bimodal: whichever scores happened to fit got
/// priced and the rest silently kept their unmodified value, which is a worse failure
/// than a slightly mis-sized adjustment. `is_plausible` stays useful for calibration,
/// where the absolute fit is the thing under test.
///
/// Worth knowing how little the shape calibration moved this: replacing the single
/// normal with the fitted two-component mixture halved mean `g_timing` across the 20
/// real scores (101.6 to 51.6) while the mean `EZ` scalar shifted only from 0.8256 to
/// 0.8273. That is the design working as intended — the scalar is a ratio of two fits
/// that share a shape error, so it is far more robust than the absolute fit is. It
/// also means the mod response is set by `skill_exponent` and the windows, not by the
/// tail, and it is why the shape could be fitted without disturbing pricing.
///
/// Returns 1.0 only when there is nothing to measure: an empty score, or a fit that
/// did not produce a usable positive skill on both sides.
fn window_scalar(attrs: &SunnyManiaDifficultyAttributes, state: SunnyScoreState) -> f64 {
    let total = state.total_hits();

    if total == 0 || attrs.n_objects == 0 || attrs.stars <= 0.0 {
        return 1.0;
    }

    let counts = [
        state.n320,
        state.n300,
        state.n200,
        state.n100,
        state.n50,
        state.misses,
    ];

    let model = ErrorModel::default();
    let units = judgement_units(attrs, f64::from(total), &model);

    let played = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);
    let reference = fit_with_quality(&counts, &units, &reference_windows(attrs), &model);

    if played.skill <= 0.0 || reference.skill <= 0.0 {
        return 1.0;
    }

    played.skill / reference.skill
}

// ---------------------------------------------------------------------------
// Hit window & hit leniency
// ---------------------------------------------------------------------------

/// The GREAT hit window following the C# `ManiaDifficultyCalculator`.
///
/// - non-convert mania maps use `34 + 3 * (10 - od)` clamped to `[34, 64]`
/// - convert maps use `34` if the original OD rounds above 4, else `47`
/// - `HR` divides the window by 1.4, `EZ` multiplies it by 1.4
/// - the clock rate scales the window but is normalized away afterwards
pub(crate) fn get_hit_window_300(
    map: &Beatmap,
    clock_rate: f64,
    has_hr: bool,
    has_ez: bool,
) -> f64 {
    let od = f64::from(map.od);

    let base = if !map.is_convert {
        let anti_od = (10.0 - od).clamp(0.0, 10.0);
        34.0 + 3.0 * anti_od
    } else if od.round() > 4.0 {
        34.0
    } else {
        47.0
    };

    let mut value = base * clock_rate + 1e-6;

    if has_hr {
        value /= 1.4;
    } else if has_ez {
        value *= 1.4;
    }

    ((value as i64) as f64 + 0.5) / clock_rate
}

/// The hit leniency `x` derived from the GREAT hit window as in the C#
/// `SunnySkill`.
fn hit_leniency_from_window(great_hit_window: f64) -> f64 {
    let x = 0.3 * (great_hit_window / 500.0).sqrt();
    x.min(0.6 * (x - 0.09) + 0.09)
}

/// Whether the score is a classic (osu!stable default / lazer with CL mod)
/// style play, i.e. long notes give a single judgement and the difficulty
/// weights use the head-only density.
pub(crate) fn is_classic(lazer: Option<bool>, mods: &GameMods) -> bool {
    let lazer = lazer.unwrap_or(true);
    // `SV2`, not `V2`: that is the acronym `rosu_mods::ScoreV2Mania` reports, and the
    // string is parsed rather than matched, so a wrong one silently never matches. It
    // did exactly that — every score read as ScoreV1, which mattered as soon as long
    // notes started being judged differently under the two.
    let sv2 = has_mod(mods, "SV2");
    let cl = has_mod(mods, "CL");

    (!lazer && !sv2) || cl
}

/// Whether the mods contain the mod with the given acronym.
fn has_mod(mods: &GameMods, acronym: &str) -> bool {
    acronym
        .parse::<Acronym>()
        .map_or(false, |acronym| mods.contains_acronym(acronym))
}

// ---------------------------------------------------------------------------
// Data preparation
// ---------------------------------------------------------------------------

/// Convert the beatmap's hit objects into notes, applying the clock rate.
/// Also computes the max combo.
fn build_notes<'a>(
    clock_rate: f64,
    objects: impl IntoIterator<Item = &'a HitObject>,
    total_columns: usize,
) -> (Vec<Note>, u32) {
    let mut notes = Vec::new();
    let mut max_combo = 0u32;

    for object in objects {
        let column = column_for(object, total_columns);
        let (head, end) = match object.kind {
            HitObjectKind::Circle => {
                max_combo += 1;
                (object.start_time, object.start_time)
            }
            HitObjectKind::Slider(_) | HitObjectKind::Spinner(_) => {
                // Spinners become holds during conversion; mania maps never
                // contain sliders. Treat them as a single note to stay safe.
                max_combo += 1;
                (object.start_time, object.start_time)
            }
            HitObjectKind::Hold(ref hold) => {
                let end = object.start_time + hold.duration;
                max_combo += 1 + (hold.duration / 100.0) as u32;
                (object.start_time, end)
            }
        };

        let head = head / clock_rate;
        let end = end / clock_rate;
        let tail = (end > head + 1e-7).then_some(end);

        notes.push(Note {
            column,
            head,
            tail,
        });
    }

    (notes, max_combo)
}

/// The column of a hit object following `ManiaObject::column`.
fn column_for(object: &HitObject, total_columns: usize) -> usize {
    let x_divisor = 512.0 / total_columns as f64;
    let column = (f64::from(object.pos.x) / x_divisor).floor();

    column.min(total_columns as f64 - 1.0).max(0.0) as usize
}

struct RebirthData {
    total_columns: usize,
    hit_leniency: f64,
    t_end: f64,
    notes: Vec<Note>,
    notes_by_column: Vec<Vec<Note>>,
    long_notes: Vec<Note>,
    tails: Vec<Note>,
    all_corners: Vec<f64>,
    base_corners: Vec<f64>,
    awkwardness_corners: Vec<f64>,
}

impl RebirthData {
    fn new(mut notes: Vec<Note>, total_columns: usize, hit_leniency: f64) -> Self {
        notes.sort_by(compare_notes);

        let mut notes_by_column = vec![Vec::new(); total_columns];
        let mut long_notes = Vec::new();

        for &note in &notes {
            if note.column < total_columns {
                notes_by_column[note.column].push(note);
            }

            if note.tail.is_some() {
                long_notes.push(note);
            }
        }

        let mut tails = long_notes.clone();
        tails.sort_by(|a, b| a.tail_or_head().total_cmp(&b.tail_or_head()));

        let t_end = notes
            .iter()
            .map(|&note| note.tail_or_head().max(note.head))
            .fold(0.0, f64::max)
            + 1.0;
        let (all_corners, base_corners, awkwardness_corners) = get_corners(t_end, &notes);

        Self {
            total_columns,
            hit_leniency,
            t_end,
            notes,
            notes_by_column,
            long_notes,
            tails,
            all_corners,
            base_corners,
            awkwardness_corners,
        }
    }
}

fn compare_notes(a: &Note, b: &Note) -> Ordering {
    a.head
        .total_cmp(&b.head)
        .then_with(|| a.column.cmp(&b.column))
}

// ---------------------------------------------------------------------------
// Corners
// ---------------------------------------------------------------------------

fn get_corners(t_end: f64, notes: &[Note]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut base = Vec::new();
    let mut awkwardness = Vec::new();

    for note in notes {
        let boundaries = [Some(note.head), note.tail];

        for boundary in boundaries.into_iter().flatten() {
            base.extend([
                boundary,
                boundary + 501.0,
                boundary - 499.0,
                boundary + 1.0,
            ]);
            awkwardness.extend([boundary, boundary + 1000.0, boundary - 1000.0]);
        }
    }

    base.extend([0.0, t_end]);
    awkwardness.extend([0.0, t_end]);

    sort_corners(&mut base, t_end);
    sort_corners(&mut awkwardness, t_end);

    let mut all = Vec::with_capacity(base.len() + awkwardness.len());
    all.extend_from_slice(&base);
    all.extend_from_slice(&awkwardness);
    sort_corners(&mut all, t_end);

    (all, base, awkwardness)
}

fn sort_corners(corners: &mut Vec<f64>, t_end: f64) {
    corners.retain(|&corner| (0.0..=t_end).contains(&corner));
    corners.sort_by(f64::total_cmp);
    corners.dedup_by(|a, b| a.total_cmp(b).is_eq());
}

// ---------------------------------------------------------------------------
// Sliding window helpers (exact cumulative-sum technique)
// ---------------------------------------------------------------------------

fn cumulative_sum(x: &[f64], f: &[f64]) -> Vec<f64> {
    let mut cumulative = vec![0.0; x.len()];

    for i in 1..x.len() {
        cumulative[i] = cumulative[i - 1] + f[i - 1] * (x[i] - x[i - 1]);
    }

    cumulative
}

fn query_cumsum(q: f64, x: &[f64], cumulative: &[f64], f: &[f64]) -> f64 {
    let Some((&first, &last)) = x.first().zip(x.last()) else {
        return 0.0;
    };

    if q <= first {
        return 0.0;
    }

    if q >= last {
        return cumulative.last().copied().unwrap_or(0.0);
    }

    let i = x.partition_point(|&value| value < q).saturating_sub(1);

    cumulative[i] + f[i] * (q - x[i])
}

fn smooth_on_corners(x: &[f64], f: &[f64], window: f64, scale: f64, average: bool) -> Vec<f64> {
    let Some((&first, &last)) = x.first().zip(x.last()) else {
        return Vec::new();
    };

    let cumulative = cumulative_sum(x, f);

    x.iter()
        .map(|&s| {
            let a = (s - window).max(first);
            let b = (s + window).min(last);
            let val = query_cumsum(b, x, &cumulative, f) - query_cumsum(a, x, &cumulative, f);

            if average {
                if b > a { val / (b - a) } else { 0.0 }
            } else {
                scale * val
            }
        })
        .collect()
}

fn interp_values(new_x: &[f64], old_x: &[f64], old_vals: &[f64]) -> Vec<f64> {
    if old_x.is_empty() || old_vals.is_empty() {
        return vec![0.0; new_x.len()];
    }

    new_x
        .iter()
        .map(|&x| {
            if x <= old_x[0] {
                return old_vals[0];
            }

            let last = old_x.len() - 1;

            if x >= old_x[last] {
                return old_vals[last];
            }

            let right = old_x.partition_point(|&value| value < x);
            let left = right - 1;
            let width = old_x[right] - old_x[left];

            if width == 0.0 {
                old_vals[left]
            } else {
                let t = (x - old_x[left]) / width;
                old_vals[left] + (old_vals[right] - old_vals[left]) * t
            }
        })
        .collect()
}

fn step_interp(new_x: &[f64], old_x: &[f64], old_vals: &[f64]) -> Vec<f64> {
    if old_x.is_empty() || old_vals.is_empty() {
        return vec![0.0; new_x.len()];
    }

    new_x
        .iter()
        .map(|&x| {
            let idx = old_x
                .partition_point(|&value| value <= x)
                .saturating_sub(1)
                .min(old_vals.len() - 1);

            old_vals[idx]
        })
        .collect()
}

fn lower_bound(values: &[f64], target: f64) -> usize {
    values.partition_point(|&value| value < target)
}

fn upper_bound(values: &[f64], target: f64) -> usize {
    values.partition_point(|&value| value <= target)
}

// ---------------------------------------------------------------------------
// Key usage & anchor
// ---------------------------------------------------------------------------

fn get_key_usage(data: &RebirthData) -> Vec<Vec<bool>> {
    let mut key_usage = vec![vec![false; data.base_corners.len()]; data.total_columns];

    for note in &data.notes {
        if note.column >= data.total_columns {
            continue;
        }

        let start_time = (note.head - 150.0).max(0.0);
        let end_time = note.tail.map_or(note.head + 150.0, |tail| {
            (tail + 150.0).min(data.t_end - 1.0)
        });
        let left = lower_bound(&data.base_corners, start_time);
        let right = lower_bound(&data.base_corners, end_time);

        for used in &mut key_usage[note.column][left..right] {
            *used = true;
        }
    }

    key_usage
}

fn get_key_usage_400(data: &RebirthData) -> Vec<Vec<f64>> {
    let mut key_usage = vec![vec![0.0; data.base_corners.len()]; data.total_columns];

    for note in &data.notes {
        if note.column >= data.total_columns {
            continue;
        }

        let start_time = note.head.max(0.0);
        let end_time = note
            .tail
            .map_or(note.head, |tail| tail.min(data.t_end - 1.0));
        let left400 = lower_bound(&data.base_corners, start_time - 400.0);
        let left = lower_bound(&data.base_corners, start_time);
        let right = lower_bound(&data.base_corners, end_time);
        let right400 = lower_bound(&data.base_corners, end_time + 400.0);

        let body = 3.75 + (end_time - start_time).min(1500.0) / 150.0;

        for value in &mut key_usage[note.column][left..right] {
            *value += body;
        }

        for (idx, value) in key_usage[note.column][left400..left].iter_mut().enumerate() {
            let corner = data.base_corners[left400 + idx];
            *value += 3.75 - 3.75 / 400.0_f64.powi(2) * (corner - start_time).powi(2);
        }

        for (idx, value) in key_usage[note.column][right..right400]
            .iter_mut()
            .enumerate()
        {
            let corner = data.base_corners[right + idx];
            *value += 3.75 - 3.75 / 400.0_f64.powi(2) * (corner - end_time).abs().powi(2);
        }
    }

    key_usage
}

fn compute_anchor(key_usage_400: &[Vec<f64>]) -> Vec<f64> {
    let len = key_usage_400.first().map_or(0, Vec::len);
    let mut anchor = vec![0.0; len];

    for idx in 0..len {
        let mut counts: Vec<_> = key_usage_400.iter().map(|column| column[idx]).collect();
        counts.sort_by(|a, b| b.total_cmp(a));
        counts.retain(|&count| count != 0.0);

        if counts.len() > 1 {
            let mut walk = 0.0;
            let mut max_walk = 0.0;

            for pair in counts.windows(2) {
                walk += pair[0] * (1.0 - 4.0 * (0.5 - pair[1] / pair[0]).powi(2));
                max_walk += pair[0];
            }

            anchor[idx] = walk / max_walk;
        }
    }

    for value in &mut anchor {
        *value = 1.0 + (*value - 0.18).min(5.0 * (*value - 0.22).powi(3));
    }

    anchor
}

// ---------------------------------------------------------------------------
// Jbar
// ---------------------------------------------------------------------------

fn jack_nerfer(delta: f64) -> f64 {
    1.0 - 7e-5 * (0.15 + (delta - 0.08).abs()).powi(-4)
}

fn compute_jbar(data: &RebirthData) -> (Vec<Vec<f64>>, Vec<f64>) {
    let len = data.base_corners.len();
    let mut j_by_column = vec![vec![0.0; len]; data.total_columns];
    let mut delta_by_column = vec![vec![1e9; len]; data.total_columns];

    for (column, notes) in data.notes_by_column.iter().enumerate() {
        for pair in notes.windows(2) {
            let start = pair[0].head;
            let end = pair[1].head;
            let left = lower_bound(&data.base_corners, start);
            let right = lower_bound(&data.base_corners, end);

            if left == right {
                continue;
            }

            let delta = 0.001 * (end - start);
            let val = delta.powi(-1) * (delta + 0.11 * data.hit_leniency.powf(0.25)).powi(-1);
            let j_val = val * jack_nerfer(delta);

            for idx in left..right {
                j_by_column[column][idx] = j_val;
                delta_by_column[column][idx] = delta;
            }
        }
    }

    let jbar_by_column: Vec<_> = j_by_column
        .iter()
        .map(|column| smooth_on_corners(&data.base_corners, column, 500.0, 0.001, false))
        .collect();
    let mut jbar = vec![0.0; len];

    for idx in 0..len {
        let mut num = 0.0;
        let mut den = 0.0;

        for column in 0..data.total_columns {
            let weight = 1.0 / delta_by_column[column][idx];
            num += jbar_by_column[column][idx].max(0.0).powi(5) * weight;
            den += weight;
        }

        jbar[idx] = (num / den.max(1e-9)).powf(0.2);
    }

    (delta_by_column, jbar)
}

// ---------------------------------------------------------------------------
// Xbar
// ---------------------------------------------------------------------------

fn compute_xbar(data: &RebirthData, active_columns: &[Vec<usize>]) -> Vec<f64> {
    const CROSS_MATRIX: [&[f64]; 11] = [
        &[-1.0],
        &[0.075, 0.075],
        &[0.125, 0.05, 0.125],
        &[0.125, 0.125, 0.125, 0.125],
        &[0.175, 0.25, 0.05, 0.25, 0.175],
        &[0.175, 0.25, 0.175, 0.175, 0.25, 0.175],
        &[0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225],
        &[0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225],
        &[0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275],
        &[
            0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275,
        ],
        &[
            0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325,
        ],
    ];

    let k = data.total_columns.min(CROSS_MATRIX.len() - 1);
    let cross_coeff = CROSS_MATRIX[k];
    let len = data.base_corners.len();
    let mut x_by_pair = vec![vec![0.0; len]; data.total_columns + 1];
    let mut fast_cross = vec![vec![0.0; len]; data.total_columns + 1];

    for pair_column in 0..=data.total_columns {
        let notes_in_pair = notes_in_pair(data, pair_column);

        for pair in notes_in_pair.windows(2) {
            let start = pair[0].head;
            let end = pair[1].head;
            let left = lower_bound(&data.base_corners, start);
            let right = lower_bound(&data.base_corners, end);

            if left == right {
                continue;
            }

            let delta = 0.001 * (end - start);
            let mut val = 0.16 * data.hit_leniency.max(delta).powi(-2);

            if (!active_columns_contains(active_columns, left, pair_column as isize - 1)
                && !active_columns_contains(
                    active_columns,
                    right.min(len - 1),
                    pair_column as isize - 1,
                ))
                || (!active_columns_contains(active_columns, left, pair_column as isize)
                    && !active_columns_contains(
                        active_columns,
                        right.min(len - 1),
                        pair_column as isize,
                    ))
            {
                val *= 1.0 - cross_coeff[pair_column.min(cross_coeff.len() - 1)];
            }

            let fast =
                (0.4 * delta.max(0.06).max(0.75 * data.hit_leniency).powi(-2) - 80.0).max(0.0);

            for idx in left..right {
                x_by_pair[pair_column][idx] = val;
                fast_cross[pair_column][idx] = fast;
            }
        }
    }

    let mut x_base = vec![0.0; len];

    for (idx, value) in x_base.iter_mut().enumerate() {
        *value += (0..=data.total_columns)
            .map(|column| x_by_pair[column][idx] * cross_coeff[column.min(cross_coeff.len() - 1)])
            .sum::<f64>();
        *value += (0..data.total_columns)
            .map(|column| {
                (fast_cross[column][idx]
                    * cross_coeff[column.min(cross_coeff.len() - 1)]
                    * fast_cross[column + 1][idx]
                    * cross_coeff[(column + 1).min(cross_coeff.len() - 1)])
                .sqrt()
            })
            .sum::<f64>();
    }

    smooth_on_corners(&data.base_corners, &x_base, 500.0, 0.001, false)
}

fn notes_in_pair(data: &RebirthData, pair_column: usize) -> Vec<Note> {
    match pair_column {
        0 => data.notes_by_column.first().cloned().unwrap_or_default(),
        column if column == data.total_columns => {
            data.notes_by_column.last().cloned().unwrap_or_default()
        }
        column => {
            let mut notes = Vec::with_capacity(
                data.notes_by_column[column - 1].len() + data.notes_by_column[column].len(),
            );
            notes.extend_from_slice(&data.notes_by_column[column - 1]);
            notes.extend_from_slice(&data.notes_by_column[column]);
            notes.sort_by(compare_notes);
            notes
        }
    }
}

fn active_columns_contains(active_columns: &[Vec<usize>], idx: usize, column: isize) -> bool {
    usize::try_from(column).is_ok_and(|column| active_columns[idx].contains(&column))
}

// ---------------------------------------------------------------------------
// Pbar
// ---------------------------------------------------------------------------

struct LongNoteBodyRepresentation {
    points: Vec<f64>,
    cumulative: Vec<f64>,
    values: Vec<f64>,
}

impl LongNoteBodyRepresentation {
    fn new(long_notes: &[Note], t_end: f64) -> Self {
        let mut changes = Vec::with_capacity(3 * long_notes.len());

        for note in long_notes {
            let Some(tail) = note.tail else { continue };
            let t0 = (note.head + 60.0).min(tail);
            let t1 = (note.head + 120.0).min(tail);

            changes.extend([(t0, 1.3), (t1, -0.3), (tail, -1.0)]);
        }

        let mut points = Vec::with_capacity(changes.len() + 2);
        points.extend([0.0, t_end]);
        points.extend(changes.iter().map(|&(time, _)| time));
        sort_corners(&mut points, t_end);

        let mut cumulative = Vec::with_capacity(points.len());
        let mut values = Vec::with_capacity(points.len().saturating_sub(1));
        let mut curr: f64 = 0.0;

        cumulative.push(0.0);

        for pair in points.windows(2) {
            for &(time, change) in &changes {
                if time.total_cmp(&pair[0]).is_eq() {
                    curr += change;
                }
            }

            let value = curr.min(2.5 + 0.5 * curr);
            values.push(value);
            cumulative
                .push(cumulative.last().copied().unwrap_or(0.0) + (pair[1] - pair[0]) * value);
        }

        Self {
            points,
            cumulative,
            values,
        }
    }

    fn sum(&self, a: f64, b: f64) -> f64 {
        if b <= a || self.values.is_empty() {
            return 0.0;
        }

        let a = a.clamp(self.points[0], *self.points.last().unwrap());
        let b = b.clamp(self.points[0], *self.points.last().unwrap());

        if b <= a {
            return 0.0;
        }

        let i = self
            .points
            .partition_point(|&point| point <= a)
            .saturating_sub(1)
            .min(self.values.len() - 1);
        let j = self
            .points
            .partition_point(|&point| point <= b)
            .saturating_sub(1)
            .min(self.values.len() - 1);

        if i == j {
            return (b - a) * self.values[i];
        }

        let first = (self.points[i + 1] - a) * self.values[i];
        let middle = self.cumulative[j] - self.cumulative[i + 1];
        let last = (b - self.points[j]) * self.values[j];

        first + middle + last
    }
}

fn stream_booster(delta: f64) -> f64 {
    let bpm = 7.5 / delta;

    if (160.0..360.0).contains(&bpm) {
        1.0 + 1.7e-7 * (bpm - 160.0) * (bpm - 360.0).powi(2)
    } else {
        1.0
    }
}

fn compute_pbar(data: &RebirthData, ln_rep: &LongNoteBodyRepresentation, anchor: &[f64]) -> Vec<f64> {
    let mut p_step = vec![0.0; data.base_corners.len()];

    for pair in data.notes.windows(2) {
        let h_l = pair[0].head;
        let h_r = pair[1].head;
        let delta_time = h_r - h_l;

        if delta_time < 1e-9 {
            let spike = 1000.0 * (0.02 * (4.0 / data.hit_leniency - 24.0)).powf(0.25);
            let left = lower_bound(&data.base_corners, h_l);
            let right = upper_bound(&data.base_corners, h_l);

            for value in &mut p_step[left..right] {
                *value += spike;
            }

            continue;
        }

        let left = lower_bound(&data.base_corners, h_l);
        let right = lower_bound(&data.base_corners, h_r);

        if left == right {
            continue;
        }

        let delta = 0.001 * delta_time;
        let v = 1.0 + 6.0 * 0.001 * ln_rep.sum(h_l, h_r);
        let booster = stream_booster(delta);
        let base = 0.08 * data.hit_leniency.powi(-1);
        let inc = if delta < 2.0 * data.hit_leniency / 3.0 {
            delta.powi(-1)
                * (base
                    * (1.0
                        - 24.0
                            * data.hit_leniency.powi(-1)
                            * (delta - data.hit_leniency / 2.0).powi(2)))
                .powf(0.25)
                * booster.max(v)
        } else {
            delta.powi(-1)
                * (base
                    * (1.0 - 24.0 * data.hit_leniency.powi(-1) * (data.hit_leniency / 6.0).powi(2)))
                .powf(0.25)
                * booster.max(v)
        };

        for idx in left..right {
            p_step[idx] += (inc * anchor[idx]).min(inc.max(inc * 2.0 - 10.0));
        }
    }

    smooth_on_corners(&data.base_corners, &p_step, 500.0, 0.001, false)
}

// ---------------------------------------------------------------------------
// Abar
// ---------------------------------------------------------------------------

fn compute_abar(
    data: &RebirthData,
    active_columns: &[Vec<usize>],
    delta_by_column: &[Vec<f64>],
) -> Vec<f64> {
    let mut dks = vec![vec![0.0; data.base_corners.len()]; data.total_columns.saturating_sub(1)];

    for idx in 0..data.base_corners.len() {
        for pair in active_columns[idx].windows(2) {
            let k0 = pair[0];
            let k1 = pair[1];

            if k0 < dks.len() && k1 < delta_by_column.len() {
                dks[k0][idx] = (delta_by_column[k0][idx] - delta_by_column[k1][idx]).abs()
                    + 0.4
                        * (delta_by_column[k0][idx].max(delta_by_column[k1][idx]) - 0.11)
                            .max(0.0);
            }
        }
    }

    let mut a_step = vec![1.0; data.awkwardness_corners.len()];

    for (idx, &corner) in data.awkwardness_corners.iter().enumerate() {
        let base_idx = lower_bound(&data.base_corners, corner).min(data.base_corners.len() - 1);

        for pair in active_columns[base_idx].windows(2) {
            let k0 = pair[0];
            let k1 = pair[1];

            if k0 >= dks.len() || k1 >= delta_by_column.len() {
                continue;
            }

            let d_val = dks[k0][base_idx];
            let max_delta = delta_by_column[k0][base_idx].max(delta_by_column[k1][base_idx]);

            if d_val < 0.02 {
                a_step[idx] *= (0.75 + 0.5 * max_delta).min(1.0);
            } else if d_val < 0.07 {
                a_step[idx] *= (0.65 + 5.0 * d_val + 0.5 * max_delta).min(1.0);
            }
        }
    }

    smooth_on_corners(&data.awkwardness_corners, &a_step, 250.0, 1.0, true)
}

// ---------------------------------------------------------------------------
// Rbar
// ---------------------------------------------------------------------------

fn find_next_note_in_column(note: Note, notes: &[Note]) -> Option<Note> {
    let idx = notes.partition_point(|candidate| candidate.head < note.head);

    notes.get(idx + 1).copied()
}

fn compute_rbar(data: &RebirthData) -> Vec<f64> {
    let mut r_step = vec![0.0; data.base_corners.len()];

    if data.tails.len() < 2 {
        return r_step;
    }

    let i_list: Vec<_> = data
        .tails
        .iter()
        .map(|tail| {
            let next_head = find_next_note_in_column(*tail, &data.notes_by_column[tail.column])
                .map_or(1e9, |note| note.head);
            let tail_time = tail.tail_or_head();
            let i_h = 0.001 * (tail_time - tail.head - 80.0).abs() / data.hit_leniency;
            let i_t = 0.001 * (next_head - tail_time - 80.0).abs() / data.hit_leniency;

            2.0 / (2.0 + (-5.0 * (i_h - 0.75)).exp() + (-5.0 * (i_t - 0.75)).exp())
        })
        .collect();

    for idx in 0..data.tails.len() - 1 {
        let t_start = data.tails[idx].tail_or_head();
        let t_end = data.tails[idx + 1].tail_or_head();
        let left = lower_bound(&data.base_corners, t_start);
        let right = lower_bound(&data.base_corners, t_end);

        if left == right {
            continue;
        }

        let delta_r = 0.001 * (t_end - t_start);
        let value = 0.08
            * delta_r.powf(-0.5)
            * data.hit_leniency.powi(-1)
            * (1.0 + 0.8 * (i_list[idx] + i_list[idx + 1]));

        for step in &mut r_step[left..right] {
            *step = value;
        }
    }

    smooth_on_corners(&data.base_corners, &r_step, 500.0, 0.001, false)
}

// ---------------------------------------------------------------------------
// Density & keys
// ---------------------------------------------------------------------------

fn compute_density_and_keys(data: &RebirthData, key_usage: &[Vec<bool>]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let note_hit_times: Vec<_> = data.notes.iter().map(|note| note.head).collect();

    // For the v2 (non-classic) path, long note tails count as additional
    // hits, matching the reference implementation's `noteHitTimesV2`.
    let mut note_hit_times_v2 = note_hit_times.clone();
    note_hit_times_v2.extend(data.long_notes.iter().filter_map(|note| note.tail));
    note_hit_times_v2.sort_by(f64::total_cmp);

    let mut density = vec![0.0; data.base_corners.len()];
    let mut density_v2 = vec![0.0; data.base_corners.len()];
    let mut keys = vec![1.0; data.base_corners.len()];

    for (idx, &corner) in data.base_corners.iter().enumerate() {
        let low = corner - 500.0;
        let high = corner + 500.0;
        density[idx] =
            (lower_bound(&note_hit_times, high) - lower_bound(&note_hit_times, low)) as f64;
        density_v2[idx] =
            (lower_bound(&note_hit_times_v2, high) - lower_bound(&note_hit_times_v2, low)) as f64;
        keys[idx] = key_usage.iter().filter(|column| column[idx]).count().max(1) as f64;
    }

    (density, density_v2, keys)
}

// ---------------------------------------------------------------------------
// Final computation
// ---------------------------------------------------------------------------

struct RebirthParams {
    sr: f64,
    spikiness: f64,
    switches: f64,
    variety: f64,
}

fn calculate_from_data(data: &RebirthData, classic: bool) -> Option<RebirthParams> {
    let key_usage = get_key_usage(data);
    let active_columns: Vec<_> = (0..data.base_corners.len())
        .map(|idx| {
            (0..data.total_columns)
                .filter(|&column| key_usage[column][idx])
                .collect::<Vec<_>>()
        })
        .collect();
    let key_usage_400 = get_key_usage_400(data);
    let anchor = compute_anchor(&key_usage_400);
    let (delta_by_column, jbar_base) = compute_jbar(data);
    let jbar = interp_values(&data.all_corners, &data.base_corners, &jbar_base);
    let xbar_base = compute_xbar(data, &active_columns);
    let xbar = interp_values(&data.all_corners, &data.base_corners, &xbar_base);
    let ln_rep = LongNoteBodyRepresentation::new(&data.long_notes, data.t_end);
    let pbar_base = compute_pbar(data, &ln_rep, &anchor);
    let pbar = interp_values(&data.all_corners, &data.base_corners, &pbar_base);
    let abar_awkwardness = compute_abar(data, &active_columns, &delta_by_column);
    let abar = interp_values(
        &data.all_corners,
        &data.awkwardness_corners,
        &abar_awkwardness,
    );
    let rbar_base = compute_rbar(data);
    let rbar = interp_values(&data.all_corners, &data.base_corners, &rbar_base);
    let (density_base, density_v2_base, keys_base) = compute_density_and_keys(data, &key_usage);
    let density = step_interp(&data.all_corners, &data.base_corners, &density_base);
    let density_v2 = step_interp(&data.all_corners, &data.base_corners, &density_v2_base);
    let keys = step_interp(&data.all_corners, &data.base_corners, &keys_base);

    let d_all: Vec<_> = (0..data.all_corners.len())
        .map(|idx| {
            let s_all = (0.4
                * (abar[idx].powf(3.0 / keys[idx]) * jbar[idx].min(8.0 + 0.85 * jbar[idx]))
                    .powf(1.5)
                + (1.0 - 0.4)
                    * (abar[idx].powf(2.0 / 3.0)
                        * (0.8 * pbar[idx] + rbar[idx] * 35.0 / (density[idx] + 8.0)))
                        .powf(1.5))
            .powf(2.0 / 3.0);
            let t_all = (abar[idx].powf(3.0 / keys[idx]) * xbar[idx]) / (xbar[idx] + s_all + 1.0);

            2.7 * s_all.powf(0.5) * t_all.powf(1.5) + s_all * 0.27
        })
        .collect();

    let mut gaps = vec![0.0; data.all_corners.len()];

    if gaps.len() < 2 {
        return None;
    }

    gaps[0] = (data.all_corners[1] - data.all_corners[0]) / 2.0;
    let last = gaps.len() - 1;
    gaps[last] = (data.all_corners[last] - data.all_corners[last - 1]) / 2.0;

    for idx in 1..last {
        gaps[idx] = (data.all_corners[idx + 1] - data.all_corners[idx - 1]) / 2.0;
    }

    // The D values always use the head-only density, but the effective
    // weights select between the classic (head-only) and v2 (head + LN tail)
    // densities, matching the reference implementation's `ContainsCL` branch.
    let effective_weights: Vec<_> = if classic {
        density.iter().zip(gaps).map(|(&c, gap)| c * gap).collect()
    } else {
        density_v2.iter().zip(gaps).map(|(&c, gap)| c * gap).collect()
    };
    let mut sorted_indices: Vec<_> = (0..d_all.len()).collect();
    sorted_indices.sort_by(|&a, &b| d_all[a].total_cmp(&d_all[b]));
    let d_sorted: Vec<_> = sorted_indices.iter().map(|&idx| d_all[idx]).collect();
    let w_sorted: Vec<_> = sorted_indices
        .iter()
        .map(|&idx| effective_weights[idx])
        .collect();
    let total_weight = w_sorted.iter().sum::<f64>();

    if total_weight <= 0.0 {
        return None;
    }

    let target_percentiles = [0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815];
    let mut cumulative_weight = 0.0;
    let mut norm_cumulative = Vec::with_capacity(w_sorted.len());

    for weight in &w_sorted {
        cumulative_weight += *weight;
        norm_cumulative.push(cumulative_weight / total_weight);
    }

    let percentile_values: Vec<_> = target_percentiles
        .iter()
        .map(|&target| {
            let idx = lower_bound(&norm_cumulative, target).min(d_sorted.len() - 1);
            d_sorted[idx]
        })
        .collect();
    let percentile_93 = percentile_values[..4].iter().sum::<f64>() / 4.0;
    let percentile_83 = percentile_values[4..].iter().sum::<f64>() / 4.0;
    let weighted_mean = (d_sorted
        .iter()
        .zip(&w_sorted)
        .map(|(&d, &w)| d.powi(5) * w)
        .sum::<f64>()
        / total_weight)
        .powf(0.2);
    let mut sr =
        (0.88 * percentile_93) * 0.25 + (0.94 * percentile_83) * 0.2 + weighted_mean * 0.55;
    let total_notes = data.notes.len() as f64
        + 0.5
            * data
                .long_notes
                .iter()
                .map(|note| (note.tail_or_head() - note.head).min(1000.0) / 200.0)
                .sum::<f64>();

    sr *= total_notes / (total_notes + 60.0);
    sr = rescale_high(sr);
    sr *= 0.975;

    let spikiness = compute_spikiness(&d_sorted, &w_sorted, weighted_mean, total_weight);
    let switches = compute_switches(data, &keys, &effective_weights);
    let variety = compute_variety(data);

    Some(RebirthParams {
        sr,
        spikiness,
        switches,
        variety,
    })
}

fn rescale_high(sr: f64) -> f64 {
    if sr <= 9.0 {
        sr
    } else {
        9.0 + (sr - 9.0) / 1.2
    }
}

/// Spikiness measure from the weighted variance of the corner difficulty
/// values, i.e. how much the difficulty spikes within the map.
fn compute_spikiness(d_sorted: &[f64], w_sorted: &[f64], weighted_mean: f64, total_weight: f64) -> f64 {
    // Degenerate cases where the reference implementation would produce NaN
    if weighted_mean == 0.0 || total_weight <= 0.0 {
        return 0.0;
    }

    let variance_sum_top = d_sorted
        .iter()
        .zip(w_sorted)
        .map(|(&d, &w)| (d.powi(8) - weighted_mean.powi(8)).powi(2) * w)
        .sum::<f64>();

    let weighted_variance = (variance_sum_top / total_weight).powf(1.0 / 8.0);

    weighted_variance.sqrt() / weighted_mean
}

/// Switch measure, i.e. how much the playstyle switches between jack and
/// stream-like patterns. Values are in the range `[0.5, 1.5]`.
///
/// Following the C# reference, the corners are weighted by the effective
/// weights (`density * gap`) rather than the raw difficulty values.
fn compute_switches(data: &RebirthData, ks_arr: &[f64], effective_weights: &[f64]) -> f64 {
    let all_corners = &data.all_corners;

    // Heads of all notes, in (head, column) order
    let heads: Vec<f64> = data.notes.iter().map(|note| note.head).collect();

    // For each head, the index of the first corner >= head (last index dropped)
    let idx_list: Vec<usize> = heads.iter().map(|&head| lower_bound(all_corners, head)).collect();
    let n = idx_list.len().saturating_sub(1);

    let ks_at_note: Vec<f64> = idx_list[..n].iter().map(|&i| ks_arr[i]).collect();
    let weights_at_note: Vec<f64> = idx_list[..n].iter().map(|&i| effective_weights[i]).collect();

    let head_gaps: Vec<f64> = heads.windows(2).map(|w| w[1] - w[0]).collect();
    let num_head_gaps = head_gaps.len();

    // Moving averages over a window of 101 gaps
    let avgs: Vec<f64> = (0..num_head_gaps)
        .map(|i| {
            let start = i.saturating_sub(50);
            let end = (i + 50).min(num_head_gaps - 1);

            head_gaps[start..=end].iter().sum::<f64>() / (end - start + 1) as f64
        })
        .collect();

    let mut signature_head = 0.0;
    let mut sum_ref_head = 0.0;

    for i in 0..num_head_gaps {
        let avg = avgs[i];

        // Skip degenerate windows where all gaps are zero
        if avg == 0.0 {
            continue;
        }

        let ratio = head_gaps[i] / avg / num_head_gaps as f64;
        signature_head += (ratio * weights_at_note[i]).sqrt() * ks_at_note[i].powf(0.25);
        sum_ref_head += (head_gaps[i] / avg) * weights_at_note[i];
    }

    let ref_signature_head = sum_ref_head.sqrt();

    // Tails of long notes, sorted by tail time
    let tails: Vec<f64> = data.tails.iter().map(|note| note.tail_or_head()).collect();

    let mut signature_tail = 0.0;
    let mut ref_signature_tail = 0.0;
    let mut num_tail_gaps = 0;

    if tails.len() > 1 && tails[tails.len() - 1] > tails[0] {
        let idx_list_tails: Vec<usize> =
            tails.iter().map(|&tail| lower_bound(all_corners, tail)).collect();
        let n_tails = idx_list_tails.len() - 1;

        let ks_at_tail: Vec<f64> = idx_list_tails[..n_tails].iter().map(|&i| ks_arr[i]).collect();
        let weights_at_tail: Vec<f64> =
            idx_list_tails[..n_tails].iter().map(|&i| effective_weights[i]).collect();

        let tail_gaps: Vec<f64> = tails.windows(2).map(|w| w[1] - w[0]).collect();
        let num_tail_gaps_tmp = tail_gaps.len();

        if num_tail_gaps_tmp > 0 {
            let avgs_tail: Vec<f64> = (0..num_tail_gaps_tmp)
                .map(|i| {
                    let start = i.saturating_sub(50);
                    let end = (i + 50).min(num_tail_gaps_tmp - 1);

                    tail_gaps[start..=end].iter().sum::<f64>() / (end - start + 1) as f64
                })
                .collect();

            for i in 0..num_tail_gaps_tmp {
                let avg = avgs_tail[i];

                // Skip degenerate windows where all gaps are zero
                if avg == 0.0 {
                    continue;
                }

                let ratio = tail_gaps[i] / avg / num_tail_gaps_tmp as f64;
                signature_tail += (ratio * weights_at_tail[i]).sqrt() * ks_at_tail[i].powf(0.25);
                ref_signature_tail += (tail_gaps[i] / avg) * weights_at_tail[i];
            }

            ref_signature_tail = ref_signature_tail.sqrt();
            num_tail_gaps = num_tail_gaps_tmp;
        }
    }

    let numerator = signature_head * num_head_gaps as f64 + signature_tail * num_tail_gaps as f64;
    let denominator =
        ref_signature_head * num_head_gaps as f64 + ref_signature_tail * num_tail_gaps as f64;

    // Degenerate case where the reference implementation would produce NaN
    if denominator == 0.0 {
        return 0.5;
    }

    numerator / denominator / 2.0 + 0.5
}

/// Variety measure based on the Rao quadratic entropy of the head, tail and
/// per-column head gaps.
fn compute_variety(data: &RebirthData) -> f64 {
    let head_gaps: Vec<i64> = data
        .notes
        .windows(2)
        .map(|w| w[1].head as i64 - w[0].head as i64)
        .collect();

    // All notes sorted by their tail time, circles have a tail of -1
    let mut tail_notes: Vec<&Note> = data.notes.iter().collect();
    tail_notes.sort_by_key(|note| tail_value(note));

    let tail_gaps: Vec<i64> = tail_notes
        .windows(2)
        .map(|w| tail_value(w[1]) - tail_value(w[0]))
        .collect();

    let head_variety = rao_quadratic_entropy_log(&head_gaps, 1);
    let tail_variety = rao_quadratic_entropy_log(&tail_gaps, 1);

    let mut head_gaps_new = Vec::new();

    for column in &data.notes_by_column {
        head_gaps_new.extend(column.windows(2).map(|w| w[1].head as i64 - w[0].head as i64));
    }

    let col_variety = 2.5 * rao_quadratic_entropy_log(&head_gaps_new, 2);

    0.5 * head_variety + 0.11 * tail_variety + 0.45 * col_variety
}

fn tail_value(note: &Note) -> i64 {
    note.tail.map_or(-1, |tail| tail as i64)
}

/// Rao's quadratic entropy on the values treated as categories, applying
/// `log_iterations` times the log(1 + |x - y|) distance.
fn rao_quadratic_entropy_log(values: &[i64], log_iterations: u32) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut counts = HashMap::new();

    for &value in values {
        *counts.entry(value).or_insert(0usize) += 1;
    }

    // Iterate in a deterministic order to avoid floating point differences
    // based on the HashMap's randomized iteration order.
    let mut uniques: Vec<i64> = counts.keys().copied().collect();
    uniques.sort_unstable();

    let total = values.len() as f64;
    let mut q = 0.0;

    for &x in &uniques {
        let p_x = counts[&x] as f64 / total;

        for &y in &uniques {
            let mut dist = (x - y).abs() as f64;

            for _ in 0..log_iterations {
                dist = (1.0 + dist).ln();
            }

            q += p_x * (counts[&y] as f64 / total) * dist;
        }
    }

    q
}

// ---------------------------------------------------------------------------
// Performance calculation
// ---------------------------------------------------------------------------

/// Matches the reference implementation's 305-based weighting (perfect hits
/// are weighted with 305 instead of 320).
fn custom_accuracy(state: SunnyScoreState) -> f64 {
    let total_hits = state.total_hits();

    if total_hits == 0 {
        return 0.0;
    }

    let numerator = state.n320 * 305 + state.n300 * 300 + state.n200 * 200 + state.n100 * 100
        + state.n50 * 50;
    let denominator = total_hits * 305;

    f64::from(numerator) / f64::from(denominator)
}

/// The "proportion" of pp that is awarded based on accuracy, i.e. how much
/// of the star rating is rewarded at the given accuracy.
fn performance_proportion(acc: f64) -> f64 {
    if acc > 0.80 {
        4.5 * (acc - 0.8) / f64::powf(100.0 * (1.0 - acc) + f64::powf(0.9, 20.0), 0.05)
    } else {
        0.0
    }
}

/// The difficulty portion of pp.
///
/// `window_scalar` carries the judgement-window effect, which is what removes the
/// need for per-mod factors here: it is derived by grading the score against the
/// windows that were actually in effect, so `EZ` is priced without being named.
/// It enters through the same `^2.2` as the star rating because both describe how
/// hard the score was to produce, so a 1% shift in either should be worth the same.
fn compute_difficulty_value(stars: f64, score_accuracy: f64, window_scalar: f64) -> f64 {
    let proportion = performance_proportion(score_accuracy);
    let effective_stars = f64::max(stars - 0.15, 0.05) * window_scalar.max(0.0);

    9.8 * f64::powf(effective_stars.max(0.05), 2.2) * proportion
}

/// Multiplier based on the map's variety, in the range `[0.945, 1.055]`.
fn variety_multiplier(variety: f64) -> f64 {
    const FLOOR: f64 = 0.945;
    const CAP: f64 = 1.055;
    const V0: f64 = 3.25;
    const K: f64 = 3.0;

    FLOOR + (CAP - FLOOR) / (1.0 + (-K * (variety - V0)).exp())
}

/// Multiplier based on the play's accuracy and the map's accuracy scalar.
fn acc_multiplier(acc: f64, acc_scalar: f64) -> f64 {
    let sigmoid_scaler = 0.87 + 0.26 / (1.0 + (-20.0 * (acc_scalar - 1.0)).exp());

    sigmoid_scaler * (2.0 * acc.powi(20) - 1.0) + 2.0 - 2.0 * acc.powi(20)
}

/// Multiplier based on the amount of notes of the map.
fn length_multiplier(total_notes: f64, stars: f64) -> f64 {
    1.1 / (1.0 + (stars / (2.0 * total_notes)).sqrt())
}

#[allow(dead_code)]
fn _assert_mode(map: &Beatmap) {
    debug_assert_eq!(map.mode, GameMode::Mania);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosu_mods::{GameMod, GameMods as LazerMods};

    const MAP_1638954: &str = r"C:\Users\uuzof\AppData\Local\Temp\opencode\rosu-pp\resources\1638954.osu";
    const MAP_5269878: &str = r"C:\Users\uuzof\AppData\Local\Temp\opencode\rosu-pp\resources\5269878.osu";

    fn single_mod(mods: &mut LazerMods, gamemod: GameMod) {
        mods.insert(gamemod);
    }

    /// Parse a beatmap, skipping the test if the resource file is unavailable.
    fn parse(path: &str) -> Option<Beatmap> {
        let bytes = std::fs::read(path).ok()?;
        Beatmap::from_bytes(&bytes).ok()
    }

    /// A synthetic 4k map: `notes` evenly spaced notes cycling across columns.
    ///
    /// The reference `.osu` files the older tests use live at absolute Windows
    /// paths and are unavailable here, so those tests silently skip. Anything that
    /// must actually run needs a map built in memory.
    fn synthetic_map(od: f32, notes: usize, spacing: f64) -> Beatmap {
        let mut map = Beatmap::default();
        map.mode = GameMode::Mania;
        map.od = od;
        map.cs = 4.0;
        map.is_convert = false;

        map.hit_objects = (0..notes)
            .map(|idx| HitObject {
                pos: rosu_pp::model::hit_object::Pos {
                    // Column from x position: lazer maps x to a column index by
                    // `x * columns / 512`.
                    x: (idx % 4) as f32 * 128.0 + 64.0,
                    y: 192.0,
                },
                start_time: idx as f64 * spacing,
                kind: HitObjectKind::Circle,
            })
            .collect();

        map
    }

    /// The Python reference (Star-Rating-Rebirth) uses the OD-based hit
    /// leniency while this port uses the C# great-hit-window based one, so
    /// the SR values differ by a small margin.
    #[test]
    fn matches_python_reference_1638954() {
        let Some(map) = parse(MAP_1638954) else {
            return;
        };
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

        // Python reference: 3.712606
        let relative = (attrs.stars - 3.712606).abs() / 3.712606;
        assert!(relative < 0.03, "SR {} deviates by {relative}", attrs.stars);
    }

    #[test]
    fn matches_python_reference_5269878() {
        let Some(map) = parse(MAP_5269878) else {
            return;
        };
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

        // Python reference: 9.299379
        let relative = (attrs.stars - 9.299379).abs() / 9.299379;
        assert!(relative < 0.03, "SR {} deviates by {relative}", attrs.stars);
    }

    #[test]
    fn hit_window_300_formula() {
        let Some(map) = parse(MAP_1638954) else {
            return;
        };
        // OD 8, non-convert: (int)(34 + 3 * (10 - 8) + 1e-6) + 0.5 = 40.5
        assert!((get_hit_window_300(&map, 1.0, false, false) - 40.5).abs() < 1e-9);

        // HR: (int)(40.000001 / 1.4) + 0.5 = 28.5
        assert!(
            (get_hit_window_300(&map, 1.0, true, false) - 28.5).abs() < 1e-9
        );

        // EZ: (int)(40.000001 * 1.4) + 0.5 = 56.5
        assert!((get_hit_window_300(&map, 1.0, false, true) - 56.5).abs() < 1e-9);

        // clock rate scales the window but the fractional truncation is kept
        // (matches the reference: (int)(40 * 1.5 + 1e-6) + 0.5 = 60.5 / 1.5)
        assert!(
            (get_hit_window_300(&map, 1.5, false, false) - 60.5 / 1.5).abs() < 1e-9
        );
    }

    #[test]
    fn ez_hr_affect_star_rating() {
        let Some(map) = parse(MAP_1638954) else {
            return;
        };
        let nm = calculate(&map, &GameMods::default(), 1.0, Some(true), None).unwrap();

        let mut hr_mods = LazerMods::new();
        single_mod(&mut hr_mods, GameMod::HardRockMania(Default::default()));
        let hr = calculate(&map, &hr_mods, 1.0, Some(true), None).unwrap();

        let mut ez_mods = LazerMods::new();
        single_mod(&mut ez_mods, GameMod::EasyMania(Default::default()));
        let ez = calculate(&map, &ez_mods, 1.0, Some(true), None).unwrap();

        assert!(
            ez.stars < nm.stars && nm.stars < hr.stars,
            "expected EZ {} < NM {} < HR {}",
            ez.stars,
            nm.stars,
            hr.stars
        );
    }

    #[test]
    fn performance_formula() {
        let Some(map) = parse(MAP_1638954) else {
            return;
        };
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

        // SS play
        let state = SunnyScoreState {
            n320: attrs.n_objects as u32,
            ..Default::default()
        };
        let perf = calculate_performance(&attrs, &mods, state);

        assert!(perf.pp > 0.0);
        assert!((perf.variety_multiplier - 0.945..=1.055).contains(&perf.variety_multiplier));
        assert!(perf.length_multiplier > 0.0 && perf.length_multiplier < 1.1);
        assert!((perf.pp - perf.pp_difficulty * perf.variety_multiplier * perf.acc_multiplier * perf.length_multiplier).abs() < 1e-6);

        // NF keeps its flat factor: failing is a scoring matter the timing surface
        // says nothing about.
        let mut nf_mods = LazerMods::new();
        single_mod(&mut nf_mods, GameMod::NoFailMania(Default::default()));
        let perf_nf = calculate_performance(&attrs, &nf_mods, state);
        assert!((perf_nf.pp - perf.pp * 0.75).abs() < 1e-6);
    }

    /// The core of the design: `EZ` is priced by grading the score against the
    /// windows it was played under, not by a mod-specific factor. The same
    /// judgement counts through wider windows imply less precision, so they are
    /// worth less — and `calculate_performance` never looks up `EZ` to do it.
    #[test]
    fn ez_is_priced_by_the_windows_not_a_multiplier() {
        let map = synthetic_map(8.0, 900, 125.0);
        let nm_mods = GameMods::default();

        let mut ez_mods = LazerMods::new();
        single_mod(&mut ez_mods, GameMod::EasyMania(Default::default()));

        let nm = calculate(&map, &nm_mods, 1.0, Some(true), None).unwrap();
        let ez = calculate(&map, &ez_mods, 1.0, Some(true), None).unwrap();

        // EZ must actually widen the windows, otherwise the rest proves nothing.
        assert!(
            ez.hit_windows.great > nm.hit_windows.great,
            "EZ should widen GREAT: {} vs {}",
            ez.hit_windows.great,
            nm.hit_windows.great
        );
        assert!(
            ez.hit_windows.perfect > nm.hit_windows.perfect,
            "EZ is the only thing that moves PERFECT: {} vs {}",
            ez.hit_windows.perfect,
            nm.hit_windows.perfect
        );

        // One observed score, both window sets. Not an SS: a saturated fit carries
        // no information about precision, so the score has to leave some headroom.
        let notes = nm.n_objects as u32;
        let n320 = notes * 92 / 100;
        let state = SunnyScoreState {
            n320,
            n300: notes - n320,
            ..Default::default()
        };

        let perf_nm = calculate_performance(&nm, &nm_mods, state);
        let perf_ez = calculate_performance(&ez, &ez_mods, state);

        assert!(
            perf_ez.window_scalar < 1.0,
            "wider windows should discount the score, got {}",
            perf_ez.window_scalar
        );

        assert!(
            perf_ez.pp < perf_nm.pp,
            "the same counts through EZ windows should be worth less: {} vs {}",
            perf_ez.pp,
            perf_nm.pp
        );

        // And the discount is the windows, not a hidden factor: passing NM windows
        // with the EZ mod list set gives the NM value back.
        let mislabelled = calculate_performance(&nm, &ez_mods, state);

        assert!(
            (mislabelled.pp - perf_nm.pp).abs() < 1e-9,
            "pp should depend on the windows, not the mod list: {} vs {}",
            mislabelled.pp,
            perf_nm.pp
        );
    }

    /// HR is the mirror image and needs no separate rule: it narrows the windows,
    /// so the same counts imply *more* precision and are worth more.
    #[test]
    fn hr_is_rewarded_by_the_same_mechanism() {
        let map = synthetic_map(8.0, 900, 125.0);
        let nm_mods = GameMods::default();

        let mut hr_mods = LazerMods::new();
        single_mod(&mut hr_mods, GameMod::HardRockMania(Default::default()));

        let nm = calculate(&map, &nm_mods, 1.0, Some(true), None).unwrap();
        let hr = calculate(&map, &hr_mods, 1.0, Some(true), None).unwrap();

        let notes = nm.n_objects as u32;
        let n320 = notes * 92 / 100;
        let state = SunnyScoreState {
            n320,
            n300: notes - n320,
            ..Default::default()
        };

        let perf_hr = calculate_performance(&hr, &hr_mods, state);

        assert!(
            perf_hr.window_scalar > 1.0,
            "narrower windows should reward the score, got {}",
            perf_hr.window_scalar
        );
    }

    /// [`REFERENCE_WINDOWS`] has to be a hand-written literal to stay `const`, so it
    /// can silently disagree with what [`hit_windows`] actually produces. It did:
    /// GOOD/OK were written +36/+66 from GREAT when the classic non-convert scheme
    /// offsets them by +33/+63, which priced an OD 8 no-mod score at 1.0072 instead
    /// of exactly 1.
    #[test]
    fn reference_windows_match_od8_no_mod() {
        let map = synthetic_map(8.0, 100, 200.0);
        let mods = GameMods::default();

        let generated = hit_windows(&map, &mods, 1.0, true);

        assert_eq!(
            generated, REFERENCE_WINDOWS,
            "reference set drifted from the OD 8 classic non-convert windows"
        );
    }

    /// Every no-mod score is priced at 1, at any OD. This is the defining property of
    /// pricing against the map's own windows — the two fits are then literally the same
    /// fit — and it is what confines the surface to charging for mods.
    ///
    /// Swept across OD rather than checked at OD 8, because at OD 8 the map reference and
    /// the retired fixed reference agree and the test cannot tell them apart. OD 0 and 10
    /// are the extremes of the mania range, and 4.2 is the fixture mean for 7K LN charts —
    /// the maps that lost 16.6% of their live pp to the fixed reference.
    #[test]
    fn a_no_mod_score_is_priced_at_one_whatever_the_od() {
        let state = SunnyScoreState {
            n320: 1400,
            n300: 480,
            n200: 90,
            n100: 20,
            n50: 5,
            misses: 5,
        };

        for od in [0.0, 4.2, 8.0, 10.0] {
            let map = synthetic_map(od, 2000, 120.0);
            let mods = GameMods::default();
            let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

            let perf = calculate_performance(&attrs, &mods, state);

            assert!(
                (perf.window_scalar - 1.0).abs() < 1e-6,
                "a no-mod score at OD {od} must price at 1, got {}",
                perf.window_scalar
            );
        }
    }

    /// The one case with nothing to measure. Everything else gets priced, however
    /// badly the model fits — see [`window_scalar`].
    #[test]
    fn an_empty_score_has_no_windows_to_price() {
        let map = synthetic_map(8.0, 900, 125.0);
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

        let empty = calculate_performance(&attrs, &mods, SunnyScoreState::default());

        assert_eq!(empty.window_scalar, 1.0, "an empty score has nothing to fit");
    }

    /// Some real scores still fit poorly even with a calibrated tail, so pricing must
    /// not depend on fit quality: gating on it left most `EZ` scores at their
    /// unmodified value, which is the bug this pins against returning.
    ///
    /// The counts are a real score from the live server (map 4229780), the worst fit
    /// in that set both before and after the error model gained its lapse component —
    /// `g_timing` went from 688 to 100, an enormous improvement that still leaves it
    /// implausible, which is precisely why this test is about pricing rather than fit.
    /// They are graded through `EZ` windows here to check that pricing happens; the
    /// original play was no-mod.
    #[test]
    fn an_implausible_fit_is_still_priced() {
        let map = synthetic_map(8.0, 3635, 90.0);
        let mut ez_mods = LazerMods::new();
        single_mod(&mut ez_mods, GameMod::EasyMania(Default::default()));
        let attrs = calculate(&map, &ez_mods, 1.5, Some(true), None).unwrap();

        let state = SunnyScoreState {
            n320: 2459,
            n300: 963,
            n200: 144,
            n100: 56,
            n50: 13,
            misses: 0,
        };

        let counts = [
            state.n320,
            state.n300,
            state.n200,
            state.n100,
            state.n50,
            state.misses,
        ];
        let units = [JudgementUnit::repeated(
            attrs.stars,
            f64::from(state.total_hits()),
        )];
        let fit = fit_with_quality(&counts, &units, &attrs.hit_windows, &ErrorModel::default());

        assert!(
            !fit.is_plausible(),
            "fixture should be an implausible fit, got g_timing={}",
            fit.g_timing
        );

        let perf = calculate_performance(&attrs, &ez_mods, state);

        assert!(
            perf.window_scalar < 0.95,
            "an implausible fit must still be priced by its windows, got {}",
            perf.window_scalar
        );
    }

    // -----------------------------------------------------------------------
    // Real-score comparison
    // -----------------------------------------------------------------------

    /// One real score from the live server, with the pp it was awarded there.
    struct Row {
        map: &'static str,
        n320: u32,
        n300: u32,
        n200: u32,
        n100: u32,
        n50: u32,
        miss: u32,
        live_pp: f64,
        live_acc: f64,
        mods: &'static str,
    }

    /// The top scores of uid 10107, an `EZ` pp exploiter, fetched from the ppy-sb
    /// tRPC API. Beatmaps live alongside in `local-fixtures/maps/`; both are
    /// gitignored, so this report skips when they are absent.
    const REAL_SCORES: &[Row] = &[
        Row { map: "4633018", n320: 1987, n300: 1710, n200: 593, n100: 20, n50: 8, miss: 138, live_pp: 1379.012, live_acc: 91.241, mods: "EZ+DT" },
        Row { map: "5583718", n320: 1399, n300: 980, n200: 324, n100: 46, n50: 5, miss: 13, live_pp: 1356.142, live_acc: 94.368, mods: "EZ+DT" },
        Row { map: "3663002", n320: 1975, n300: 1863, n200: 591, n100: 34, n50: 0, miss: 210, live_pp: 1313.038, live_acc: 90.01, mods: "EZ+DT" },
        Row { map: "4870605", n320: 1436, n300: 1194, n200: 600, n100: 35, n50: 0, miss: 42, live_pp: 1279.841, live_acc: 91.181, mods: "EZ+DT" },
        Row { map: "4870608", n320: 2590, n300: 2266, n200: 928, n100: 132, n50: 30, miss: 50, live_pp: 1240.625, live_acc: 92.123, mods: "EZ+DT" },
        Row { map: "3583718", n320: 1359, n300: 1458, n200: 783, n100: 52, n50: 3, miss: 65, live_pp: 1199.563, live_acc: 89.357, mods: "EZ+DT" },
        Row { map: "5583724", n320: 1323, n300: 1366, n200: 550, n100: 102, n50: 29, miss: 17, live_pp: 1183.49, live_acc: 91.364, mods: "EZ+DT" },
        Row { map: "4459721", n320: 1306, n300: 1502, n200: 648, n100: 34, n50: 0, miss: 71, live_pp: 1095.582, live_acc: 90.408, mods: "EZ+DT" },
        Row { map: "4459716", n320: 1240, n300: 1120, n200: 486, n100: 93, n50: 1, miss: 18, live_pp: 1094.649, live_acc: 91.791, mods: "EZ+DT" },
        Row { map: "4807505", n320: 2407, n300: 1825, n200: 761, n100: 128, n50: 35, miss: 54, live_pp: 1065.901, live_acc: 91.897, mods: "EZ+DT" },
        Row { map: "5583717", n320: 1095, n300: 1149, n200: 544, n100: 34, n50: 1, miss: 105, live_pp: 1028.791, live_acc: 88.565, mods: "EZ+DT" },
        Row { map: "4870609", n320: 1048, n300: 940, n200: 326, n100: 17, n50: 0, miss: 49, live_pp: 984.332, live_acc: 92.098, mods: "EZ+DT" },
        Row { map: "4459712", n320: 1415, n300: 1016, n200: 393, n100: 47, n50: 7, miss: 13, live_pp: 965.078, live_acc: 93.733, mods: "EZ+DT" },
        Row { map: "4459715", n320: 1203, n300: 1536, n200: 779, n100: 37, n50: 0, miss: 65, live_pp: 945.481, live_acc: 89.414, mods: "EZ+DT" },
        Row { map: "4459717", n320: 1213, n300: 1138, n200: 583, n100: 38, n50: 4, miss: 38, live_pp: 920.466, live_acc: 90.503, mods: "EZ+DT" },
        Row { map: "4706643", n320: 882, n300: 538, n200: 195, n100: 48, n50: 1, miss: 19, live_pp: 895.026, live_acc: 93.058, mods: "EZ+DT" },
        Row { map: "4459723", n320: 940, n300: 953, n200: 410, n100: 30, n50: 0, miss: 47, live_pp: 852.64, live_acc: 90.591, mods: "EZ+DT" },
        Row { map: "4229780", n320: 2459, n300: 963, n200: 144, n100: 56, n50: 13, miss: 82, live_pp: 722.134, live_acc: 95.207, mods: "" },
        Row { map: "3477077", n320: 1482, n300: 637, n200: 84, n100: 12, n50: 3, miss: 42, live_pp: 707.994, live_acc: 96.438, mods: "" },
        Row { map: "3477076", n320: 1587, n300: 598, n200: 66, n100: 8, n50: 1, miss: 16, live_pp: 672.317, live_acc: 98.059, mods: "" },
    ];

    /// One fixture reduced to what the surface needs: the windows it was played
    /// under, its star rating, and its judgement counts.
    struct LoadedScore {
        map: &'static str,
        mods: &'static str,
        stars: f64,
        windows: ManiaHitWindows,
        counts: [u32; 6],
    }

    /// Load every fixture that is present on disk. Returns empty when the gitignored
    /// fixture directory is absent, which is how the reports skip cleanly.
    fn load_real_scores() -> Vec<LoadedScore> {
        let mut loaded = Vec::new();

        for row in REAL_SCORES {
            let path = format!("local-fixtures/maps/{}.osu", row.map);
            let Some(map) = parse(&path) else {
                continue;
            };

            let mut mods = LazerMods::new();
            if row.mods.contains("EZ") {
                single_mod(&mut mods, GameMod::EasyMania(Default::default()));
            }
            let clock_rate = if row.mods.contains("DT") { 1.5 } else { 1.0 };

            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(true), None) else {
                continue;
            };

            loaded.push(LoadedScore {
                map: row.map,
                mods: row.mods,
                stars: attrs.stars,
                windows: attrs.hit_windows,
                counts: [row.n320, row.n300, row.n200, row.n100, row.n50, row.miss],
            });
        }

        loaded
    }

    /// Mean `g_timing` across the loaded scores under a candidate model.
    ///
    /// The mean is the right pooling here precisely because `g_timing` does not grow
    /// with map length — every score contributes on the same scale regardless of note
    /// count, so averaging weights each score equally rather than letting the
    /// six-thousand-note maps dominate.
    fn mean_g_timing(scores: &[LoadedScore], model: &ErrorModel) -> f64 {
        if scores.is_empty() {
            return f64::INFINITY;
        }

        let mut total = 0.0;

        for score in scores {
            let units = [JudgementUnit::repeated(
                score.stars,
                f64::from(score.counts.iter().sum::<u32>()),
            )];
            let fit = fit_with_quality(&score.counts, &units, &score.windows, model);

            if !fit.g_timing.is_finite() {
                return f64::INFINITY;
            }

            total += fit.g_timing;
        }

        total / scores.len() as f64
    }

    /// Not an assertion — the calibration itself. Searches `sigma_ref`,
    /// `lapse_weight` and `lapse_ratio` for the combination that best explains the 20
    /// real scores, holding `skill_exponent` and `difficulty_floor` fixed.
    ///
    /// Those two are held deliberately. The fixture set is one player whose fitted
    /// skill sits at 0.96-1.72x the star rating on every map, and `skill_exponent` is
    /// only identified by *variation* in that ratio — at a ratio of 1, `sigma` equals
    /// `sigma_ref` whatever the exponent is, so the two are nearly jointly
    /// unidentified here. Fitting the exponent on this data would mostly absorb one
    /// player's idiosyncrasy while silently resetting the entire mod response, since
    /// it alone sets how the scalar answers a window change.
    ///
    /// Run with `cargo test calibration_search -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn calibration_search() {
        let scores = load_real_scores();

        if scores.is_empty() {
            println!("no fixtures present; nothing to calibrate");
            return;
        }

        let baseline = ErrorModel::default();

        // The single normal this work replaced, kept as the comparison point so the
        // improvement stays visible now that the mixture *is* the default.
        let single_normal = ErrorModel {
            lapse_weight: 0.0,
            ..baseline
        };

        println!(
            "single normal:   mean g_timing={:.2}",
            mean_g_timing(&scores, &single_normal)
        );
        println!(
            "current default: lapse_weight={:.4} lapse_ratio={:.3} mean g_timing={:.2}",
            baseline.lapse_weight,
            baseline.lapse_ratio,
            mean_g_timing(&scores, &baseline)
        );

        // `sigma_ref` is deliberately not searched. It is structurally
        // unidentifiable, not merely weakly identified: it sets the unit skill is
        // measured in, and skill is refit per score, so any change in `sigma_ref` is
        // absorbed exactly by the fitted skill and no observable moves at all. The
        // sweep further down demonstrates this — `g_timing` is identical to four
        // decimals across a 16x range, with skill scaling as
        // `sigma_ref^(1/skill_exponent)`. It stays at its existing value as a gauge
        // choice, which also keeps fitted skill roughly on the star-rating scale.
        //
        // The real fit is therefore two-dimensional, over the shape parameters only.
        // Search from the single normal rather than from the current default, so the
        // result does not depend on the answer already being baked into the defaults.
        let mut best = single_normal;
        let mut best_score = mean_g_timing(&scores, &single_normal);

        let mut weight = 0.0;
        while weight <= 0.60 {
            let mut ratio = 1.5;
            while ratio <= 20.0 {
                let candidate = ErrorModel {
                    lapse_weight: weight,
                    lapse_ratio: ratio,
                    ..baseline
                };
                let value = mean_g_timing(&scores, &candidate);

                if value < best_score {
                    best_score = value;
                    best = candidate;
                }

                ratio += 0.25;
            }
            weight += 0.005;
        }

        println!(
            "grid best: lapse_weight={:.4} lapse_ratio={:.2} mean g_timing={:.2}",
            best.lapse_weight, best.lapse_ratio, best_score
        );

        // Coordinate descent with a shrinking step, refining the grid winner.
        let mut step = [0.0025, 0.125];

        for _ in 0..60 {
            for (axis, &size) in step.iter().enumerate() {
                for direction in [-1.0, 1.0] {
                    let mut candidate = best;
                    let delta = size * direction;

                    if axis == 0 {
                        candidate.lapse_weight = (best.lapse_weight + delta).clamp(0.0, 0.95);
                    } else {
                        candidate.lapse_ratio = (best.lapse_ratio + delta).max(1.0);
                    }

                    let value = mean_g_timing(&scores, &candidate);

                    if value < best_score {
                        best_score = value;
                        best = candidate;
                    }
                }
            }

            for entry in &mut step {
                *entry *= 0.75;
            }
        }

        println!(
            "refined:   lapse_weight={:.4} lapse_ratio={:.3} mean g_timing={:.2}",
            best.lapse_weight, best.lapse_ratio, best_score
        );

        // Profile `lapse_ratio`: at each fixed ratio, re-optimise the other two and
        // report the best achievable objective. A flat profile means the ratio is not
        // separately identified by this data and the value chosen inside the flat
        // region is arbitrary — which is worth knowing before treating any single
        // triple as "the" calibration.
        println!("\nprofile over lapse_ratio (others re-optimised at each point):");
        println!(
            "{:>7} {:>10} {:>10} {:>10}",
            "ratio", "weight", "g_timing", "ez_scalar"
        );

        for &ratio in &[3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0, 10.0, 20.0] {
            let mut local = ErrorModel {
                lapse_ratio: ratio,
                ..best
            };
            let mut local_score = mean_g_timing(&scores, &local);
            let mut local_step = 0.05;

            for _ in 0..50 {
                for direction in [-1.0, 1.0] {
                    let mut candidate = local;
                    candidate.lapse_weight =
                        (local.lapse_weight + local_step * direction).clamp(0.0, 0.95);

                    let value = mean_g_timing(&scores, &candidate);

                    if value < local_score {
                        local_score = value;
                        local = candidate;
                    }
                }

                local_step *= 0.8;
            }

            // The EZ scalar at this point, so the profile shows whether the flat
            // region is also flat in the quantity that actually reaches pp.
            let mut ez_here = Vec::new();

            for score in &scores {
                if !score.mods.contains("EZ") {
                    continue;
                }

                let units = [JudgementUnit::repeated(
                    score.stars,
                    f64::from(score.counts.iter().sum::<u32>()),
                )];
                let played = fit_with_quality(&score.counts, &units, &score.windows, &local);
                let reference = fit_with_quality(&score.counts, &units, &REFERENCE_WINDOWS, &local);

                if played.skill > 0.0 && reference.skill > 0.0 {
                    ez_here.push(played.skill / reference.skill);
                }
            }

            let ez_mean = ez_here.iter().sum::<f64>() / ez_here.len().max(1) as f64;

            println!(
                "{ratio:>7.1} {:>10.4} {local_score:>10.2} {ez_mean:>10.4}",
                local.lapse_weight
            );
        }

        // Is `sigma_ref` identified at all? The profile above wanders it over 7.5-12.6
        // while the objective moves in the third decimal, which suggests not. Sweep it
        // alone, holding the shape fixed, and print the fitted skill alongside.
        println!("\nsigma_ref sweep at fixed shape (skill of the first score shown):");
        println!("{:>10} {:>10} {:>12}", "sigma_ref", "g_timing", "skill[0]");

        for &sigma_ref in &[4.5, 9.0, 18.0, 36.0, 72.0] {
            let candidate = ErrorModel { sigma_ref, ..best };
            let first = &scores[0];
            let units = [JudgementUnit::repeated(
                first.stars,
                f64::from(first.counts.iter().sum::<u32>()),
            )];
            let fit = fit_with_quality(&first.counts, &units, &first.windows, &candidate);

            println!(
                "{sigma_ref:>10.2} {:>10.4} {:>12.4}",
                mean_g_timing(&scores, &candidate),
                fit.skill
            );
        }

        // What the calibrated shape does to the thing under test: the window scalar,
        // and so the mod response. Reported rather than asserted — there is no pp
        // target for EZ, the figure is an output of the calibration.
        let mut ez = Vec::new();
        let mut nm = Vec::new();

        println!(
            "\n{:>9} {:>7} {:>7} {:>9} {:>9}",
            "map", "mods", "scalar", "g_before", "g_after"
        );

        for score in &scores {
            let units = [JudgementUnit::repeated(
                score.stars,
                f64::from(score.counts.iter().sum::<u32>()),
            )];

            let before = fit_with_quality(&score.counts, &units, &score.windows, &single_normal);
            let after = fit_with_quality(&score.counts, &units, &score.windows, &best);
            let reference = fit_with_quality(&score.counts, &units, &REFERENCE_WINDOWS, &best);

            let scalar = if after.skill > 0.0 && reference.skill > 0.0 {
                after.skill / reference.skill
            } else {
                1.0
            };

            println!(
                "{:>9} {:>7} {:>7.4} {:>9.1} {:>9.1}",
                score.map,
                if score.mods.is_empty() { "NM" } else { score.mods },
                scalar,
                before.g_timing,
                after.g_timing,
            );

            if score.mods.contains("EZ") {
                ez.push(scalar);
            } else {
                nm.push(scalar);
            }
        }

        let summarise = |label: &str, values: &[f64]| {
            if values.is_empty() {
                return;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            // pp moves as the scalar to the ~1.1 power through
            // `compute_difficulty_value`; reported as the ratio itself here since the
            // pp mapping is the report above's job.
            println!("{label}: n={} mean scalar {mean:.4}", values.len());
        };

        println!();
        summarise("EZ", &ez);
        summarise("NM", &nm);
    }

    /// Not an assertion — a report. Prices every real score through the current
    /// pipeline and prints the window scalar next to what the live server paid, so
    /// the mod response can be read off real data rather than synthetics.
    ///
    /// Run with `cargo test real_score_report -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn real_score_report() {
        let mut priced = 0usize;
        let mut ez_scalars = Vec::new();
        let mut nm_scalars = Vec::new();

        println!(
            "{:>9} {:>7} {:>4} {:>4} {:>6} {:>7} {:>8} {:>8} {:>7} {:>7} {:>9} {:>8}",
            "map",
            "mods",
            "od",
            "cvt",
            "stars",
            "acc%",
            "livePP",
            "ourPP",
            "scalar",
            "ppRatio",
            "g_timing",
            "plaus"
        );

        for row in REAL_SCORES {
            let path = format!("local-fixtures/maps/{}.osu", row.map);
            let Some(map) = parse(&path) else {
                println!("{:>9} missing beatmap", row.map);
                continue;
            };

            let has_ez = row.mods.contains("EZ");
            let has_dt = row.mods.contains("DT");

            let mut mods = LazerMods::new();
            if has_ez {
                single_mod(&mut mods, GameMod::EasyMania(Default::default()));
            }
            let clock_rate = if has_dt { 1.5 } else { 1.0 };

            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(true), None) else {
                println!("{:>9} no difficulty attributes", row.map);
                continue;
            };

            let state = SunnyScoreState {
                n320: row.n320,
                n300: row.n300,
                n200: row.n200,
                n100: row.n100,
                n50: row.n50,
                misses: row.miss,
            };

            let perf = calculate_performance(&attrs, &mods, state);

            let counts = [
                state.n320,
                state.n300,
                state.n200,
                state.n100,
                state.n50,
                state.misses,
            ];
            let units = [JudgementUnit::repeated(
                attrs.stars,
                f64::from(state.total_hits()),
            )];
            let fit =
                fit_with_quality(&counts, &units, &attrs.hit_windows, &ErrorModel::default());

            // What the same score would be worth with the scalar switched off, so
            // the window effect can be read directly in pp rather than in skill.
            let unpriced = compute_difficulty_value(attrs.stars, custom_accuracy(state), 1.0);
            let pp_ratio = if unpriced > 0.0 {
                compute_difficulty_value(attrs.stars, custom_accuracy(state), perf.window_scalar)
                    / unpriced
            } else {
                1.0
            };

            println!(
                "{:>9} {:>7} {:>4.1} {:>4} {:>6.2} {:>7.3} {:>8.1} {:>8.1} {:>7.4} {:>7.4} {:>9.1} {:>8}",
                row.map,
                if row.mods.is_empty() { "NM" } else { row.mods },
                map.od,
                map.is_convert,
                attrs.stars,
                row.live_acc,
                row.live_pp,
                perf.pp,
                perf.window_scalar,
                pp_ratio,
                fit.g_timing,
                fit.is_plausible()
            );

            priced += 1;
            if has_ez {
                ez_scalars.push((perf.window_scalar, pp_ratio));
            } else {
                nm_scalars.push((perf.window_scalar, pp_ratio));
            }
        }

        if priced == 0 {
            println!("no fixtures present; nothing to report");
            return;
        }

        let summarise = |label: &str, values: &[(f64, f64)]| {
            if values.is_empty() {
                return;
            }
            let n = values.len() as f64;
            let mean_scalar = values.iter().map(|v| v.0).sum::<f64>() / n;
            let mean_pp = values.iter().map(|v| v.1).sum::<f64>() / n;
            let min = values.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
            let max = values.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{label}: n={} mean scalar {mean_scalar:.4} ({min:.4}..{max:.4})  \
                 mean pp ratio {mean_pp:.4}",
                values.len()
            );
        };

        println!();
        summarise("EZ", &ez_scalars);
        summarise("NM", &nm_scalars);
    }

    /// Not an assertion — the one real external check available on the surface.
    ///
    /// A score screenshot supplied an *Unstable Rate*, which is `10 * sigma` of the
    /// player's hit errors. That is a direct measurement of the exact quantity the
    /// surface otherwise has to infer from judgement counts alone, so it tests the
    /// model against ground truth rather than against its own residuals — something
    /// the tRPC fixtures in [`REAL_SCORES`] cannot do, since the API carries no UR.
    ///
    /// Note this works even though `sigma_ref` is unidentifiable
    /// (`sigma_ref_only_sets_the_scale_of_skill`): the *implied sigma* at the fitted
    /// skill is identified, because sigma is the only channel difficulty and skill
    /// enter through. The gauge cancels.
    ///
    /// Run with `cargo test unstable_rate_check -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn unstable_rate_check() {
        use crate::mania_accuracy::expected_counts;
        use crate::mania_windows::ManiaJudgement;

        // Yooh - Decoy [Rachel's Ruins "Buffed Ver."], played by Reflec with DT.
        // OD7 4K non-convert, 3860 notes, no long notes. Counts read off the result
        // screen; the 92.53% shown is ScoreV1 weighting (rainbow 300 counts as 300),
        // which the counts reproduce as 92.54%.
        let Some(map) = parse("local-fixtures/maps/4055699.osu") else {
            println!("beatmap 4055699 absent; nothing to check");
            return;
        };

        let measured_ur = 498.40;
        let measured_sigma = measured_ur / 10.0;
        let mean_error = 34.25;

        let state = SunnyScoreState {
            n320: 1542,
            n300: 1595,
            n200: 603,
            n100: 91,
            n50: 15,
            misses: 14,
        };
        let counts = [
            state.n320,
            state.n300,
            state.n200,
            state.n100,
            state.n50,
            state.misses,
        ];
        let total = state.total_hits();

        // Classic (stable) scoring: the screenshot is osu!stable ScoreV1.
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.5, Some(false), None).unwrap();

        let units = [JudgementUnit::repeated(attrs.stars, f64::from(total))];
        let model = ErrorModel::default();
        let fit = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);

        let windows = attrs.hit_windows;
        println!(
            "map: OD {} convert {} | {total} notes | stars {:.2} (DT 1.5x, classic)",
            map.od, map.is_convert, attrs.stars
        );
        println!(
            "windows: perfect {:.1} great {:.1} good {:.1} ok {:.1} meh {:.1} miss {:.1}",
            windows.perfect,
            windows.great,
            windows.good,
            windows.ok,
            windows.meh,
            windows.miss
        );

        let implied_sigma = model.sigma(attrs.stars, fit.skill);

        println!(
            "\nfitted skill {:.3} (ratio {:.3} of stars) -> implied sigma {:.2} ms",
            fit.skill,
            fit.skill / attrs.stars,
            implied_sigma
        );
        println!(
            "measured: UR {measured_ur:.2} -> sigma {measured_sigma:.2} ms  \
             (mean error {mean_error:.2} ms)",
        );
        println!(
            "ratio implied/measured = {:.3}",
            implied_sigma / measured_sigma
        );

        // The mixture has two widths; the single number comparable to a measured UR is
        // the mixture's own standard deviation, not the core width. For a zero-mean
        // two-component mixture, variance = (1-w)*s^2 + w*(k*s)^2.
        let weight = model.lapse_weight;
        let ratio = model.lapse_ratio;
        let mixture_sigma =
            implied_sigma * ((1.0 - weight) + weight * ratio * ratio).sqrt();

        println!(
            "mixture sigma (both components) = {mixture_sigma:.2} ms  \
             -> UR {:.1}, ratio to measured {:.3}",
            mixture_sigma * 10.0,
            mixture_sigma / measured_sigma
        );

        // Cross-check that does not involve the model at all: what sigma does the
        // observed PERFECT rate alone imply, for a plain normal? P(|e| < w) = share
        // inverts to sigma = w / z where z is the normal quantile. If this lands near
        // the fit's sigma but far from the measured UR, then the UR and the judgement
        // counts disagree with each other, and the model is siding with the counts.
        let perfect_share = f64::from(state.n320) / f64::from(total - state.misses);
        // z such that P(|Z| < z) = share, by bisection on the standard normal.
        let mut lo = 1e-6;
        let mut hi = 10.0;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            // P(|Z| < mid) = 1 - erfc(mid / sqrt(2))
            let inside = 1.0 - crate::mania_accuracy::erfc(mid / std::f64::consts::SQRT_2);
            if inside < perfect_share {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let z = 0.5 * (lo + hi);
        let sigma_from_320 = windows.perfect / z;

        println!(
            "\nmodel-free check: {:.1}% of hit notes inside the {:.1} ms PERFECT window\n\
             implies sigma {:.2} ms for a plain normal (z = {z:.4})",
            perfect_share * 100.0,
            windows.perfect,
            sigma_from_320
        );

        // The same inversion in the real-time frame, in case the client reports UR in
        // real milliseconds rather than map milliseconds. Under DT the two differ by
        // the clock rate, and it is worth showing both since the conclusion should not
        // rest on a convention.
        println!(
            "if the UR is real-time rather than map-time, the measured sigma is \
             {:.2} ms in map time instead",
            measured_sigma * 1.5
        );

        // Observed vs predicted band shares, conditioned on the note being hit.
        let expected = expected_counts(&units, &windows, &model, fit.skill);
        let observed_timing = f64::from(total - state.misses);
        let expected_timing = expected.total() - expected.get(ManiaJudgement::Miss);

        println!("\n{:>10} {:>10} {:>10}", "judgement", "observed", "predicted");

        for (label, judgement, observed) in [
            ("320", ManiaJudgement::Perfect, state.n320),
            ("300", ManiaJudgement::Great, state.n300),
            ("200", ManiaJudgement::Good, state.n200),
            ("100", ManiaJudgement::Ok, state.n100),
            ("50", ManiaJudgement::Meh, state.n50),
        ] {
            println!(
                "{label:>10} {:>10.4} {:>10.4}",
                f64::from(observed) / observed_timing,
                expected.get(judgement) / expected_timing
            );
        }

        println!(
            "\nmisses: observed {} predicted {:.1}",
            state.misses,
            expected.get(ManiaJudgement::Miss) / expected.total() * f64::from(total)
        );
        println!(
            "g_timing {:.1} plausible {} identifiable {}",
            fit.g_timing,
            fit.is_plausible(),
            fit.is_identifiable()
        );

        // What it prices at.
        let perf = calculate_performance(&attrs, &mods, state);
        println!(
            "\npp {:.1} | window_scalar {:.4} | custom_accuracy {:.3}%",
            perf.pp,
            perf.window_scalar,
            custom_accuracy(state) * 100.0
        );
    }

    /// Not an assertion — prints sunny's own star rating for every map named in a
    /// ladder TSV, so a fit against replay-measured sigma can use the difficulty the
    /// model actually grades on.
    ///
    /// `tools/fetch_ladder.sh` selects on bancho.py's stored `maps.diff`, which is a
    /// *different* difficulty calculation. That is fine for choosing a spread of maps
    /// but wrong to regress against: the exponent in
    /// `sigma = sigma_ref * ((d + floor)/skill)^skill_exponent` is only meaningful in
    /// the units `d` is expressed in. Reads map ids from stdin, one per line.
    ///
    /// Run with
    /// `cut -f4 local-fixtures/ladder.tsv | tail -n +2 | cargo test ladder_stars --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn ladder_stars() {
        use std::io::BufRead as _;

        println!("map_id,stars,od,keys,is_convert");
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let id = line.trim();
            if id.is_empty() || id == "mapid" {
                continue;
            }
            let path = format!("local-fixtures/maps/{id}.osu");
            let Some(map) = parse(&path) else {
                eprintln!("skip {id}: cannot parse");
                continue;
            };
            // No mods and rate 1.0: the ladder is deliberately no-mod/NF only, so the
            // windows and note timings are the map's own.
            let Some(attrs) = calculate(&map, &GameMods::default(), 1.0, Some(false), None) else {
                eprintln!("skip {id}: not a mania map");
                continue;
            };
            println!(
                "{id},{:.4},{},{},{}",
                attrs.stars, map.od, map.cs as u32, map.is_convert
            );
        }
    }

    /// Prices every score in a ladder TSV and reports what the surface says about it.
    ///
    /// The point of difference from [`real_score_report`] is coverage: that set is 20
    /// scores chosen to be EZ-heavy, all from strong players on 8-13 star maps, which
    /// is the right shape for reading the mod response and the wrong shape for
    /// checking whether the fit behaves across the population. This reads the ladder
    /// fixtures instead — 270 no-mod scores, 9 players in two disjoint skill bands,
    /// 2.3 to 10.0 stars — and groups by player so the skill estimate can be seen
    /// tracking difficulty within one person rather than across a mixed field.
    ///
    /// The `pp` column in the TSV comes from the live ppy.sb server, which runs an
    /// older algorithm and not sunny, so it is reported as context rather than as a
    /// target: a ratio against it measures the gap between two algorithms and not the
    /// error in this one.
    ///
    /// Usage:
    /// `cargo test --release ladder_report -- --ignored --nocapture --exact
    /// sunny::tests::ladder_report < local-fixtures/ladder.tsv`
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn ladder_report() {
        use crate::mania_accuracy::skill_for_counts;
        use std::collections::BTreeMap;
        use std::io::BufRead as _;

        struct Row {
            stars: f64,
            od: f32,
            acc: f64,
            live_pp: f64,
            our_pp: f64,
            skill: f64,
            scalar: f64,
            g_timing: f64,
            plausible: bool,
            notes: u32,
        }

        let mut by_player: BTreeMap<String, Vec<Row>> = BTreeMap::new();
        let mut skipped = 0usize;

        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let fields: Vec<&str> = line.trim_end().split('\t').collect();
            if fields.len() < 16 || fields[0] == "cohort" {
                continue;
            }

            let parse_u32 = |s: &str| s.parse::<u32>().unwrap_or(0);
            let state = SunnyScoreState {
                n320: parse_u32(fields[10]),
                n300: parse_u32(fields[11]),
                n200: parse_u32(fields[12]),
                n100: parse_u32(fields[13]),
                n50: parse_u32(fields[14]),
                misses: parse_u32(fields[15]),
            };

            let path = format!("local-fixtures/maps/{}.osu", fields[3]);
            let Some(map) = parse(&path) else {
                skipped += 1;
                continue;
            };
            // The ladder is no-mod/NF only by construction, so no window or rate mods
            // apply and the map's own timings are the right ones.
            let Some(attrs) = calculate(&map, &GameMods::default(), 1.0, Some(false), None) else {
                skipped += 1;
                continue;
            };

            let perf = calculate_performance(&attrs, &GameMods::default(), state);
            let counts = [
                state.n320,
                state.n300,
                state.n200,
                state.n100,
                state.n50,
                state.misses,
            ];
            let units = [JudgementUnit::repeated(
                attrs.stars,
                f64::from(state.total_hits()),
            )];
            let model = ErrorModel::default();
            let fit = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);

            by_player.entry(fields[0].to_owned()).or_default().push(Row {
                stars: attrs.stars,
                od: map.od,
                acc: fields[8].parse().unwrap_or(0.0),
                live_pp: fields[9].parse().unwrap_or(0.0),
                our_pp: perf.pp,
                skill: skill_for_counts(&counts, &units, &attrs.hit_windows, &model),
                scalar: perf.window_scalar,
                g_timing: fit.g_timing,
                plausible: fit.is_plausible(),
                notes: state.total_hits(),
            });
        }

        if by_player.is_empty() {
            println!("no rows read; pipe a ladder TSV on stdin");
            return;
        }

        let mut all: Vec<&Row> = Vec::new();

        for (player, rows) in &by_player {
            let mut rows: Vec<&Row> = rows.iter().collect();
            rows.sort_by(|a, b| a.stars.total_cmp(&b.stars));

            println!("\n=== player {player} ({} scores)", rows.len());
            println!(
                "{:>6} {:>4} {:>6} {:>7} {:>8} {:>8} {:>7} {:>7} {:>9} {:>6}",
                "stars", "od", "notes", "acc%", "livePP", "ourPP", "skill", "sk/st", "g_timing",
                "plaus"
            );

            // Every third row: the shape across difficulty is the point, and 30 lines
            // per player would bury it.
            for row in rows.iter().step_by(3) {
                println!(
                    "{:>6.2} {:>4.1} {:>6} {:>7.3} {:>8.1} {:>8.1} {:>7.2} {:>7.2} {:>9.1} {:>6}",
                    row.stars,
                    row.od,
                    row.notes,
                    row.acc,
                    row.live_pp,
                    row.our_pp,
                    row.skill,
                    row.skill / row.stars,
                    row.g_timing,
                    row.plausible
                );
            }

            let mean = |f: &dyn Fn(&Row) -> f64| -> f64 {
                rows.iter().map(|r| f(r)).sum::<f64>() / rows.len() as f64
            };
            println!(
                "  mean skill {:.2}, mean skill/stars {:.2}, plausible {}/{}",
                mean(&|r| r.skill),
                mean(&|r| r.skill / r.stars),
                rows.iter().filter(|r| r.plausible).count(),
                rows.len()
            );

            all.extend(rows);
        }

        println!("\n=== overall ({} scores, {skipped} skipped)", all.len());

        let scalars: Vec<f64> = all.iter().map(|r| r.scalar).collect();
        let lo = scalars.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = scalars.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "window scalar: {lo:.4}..{hi:.4} (no-mod, so departures from 1 are OD alone)"
        );

        let plausible = all.iter().filter(|r| r.plausible).count();
        println!(
            "plausible: {plausible}/{} ({:.0}%)",
            all.len(),
            100.0 * plausible as f64 / all.len() as f64
        );

        let mut g: Vec<f64> = all.iter().map(|r| r.g_timing).collect();
        g.sort_by(f64::total_cmp);
        println!(
            "g_timing median {:.1}, p90 {:.1}",
            g[g.len() / 2],
            g[g.len() * 9 / 10]
        );

        // Does the fit place players consistently? Within one player, skill should be
        // roughly flat across difficulty; a trend means the exponent is off.
        println!("\nskill/stars by star band (flat = the exponent is right):");
        for (lo, hi) in [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 11.0)] {
            let band: Vec<&&Row> = all
                .iter()
                .filter(|r| r.stars >= lo && r.stars < hi)
                .collect();
            if band.is_empty() {
                continue;
            }
            let ratio =
                band.iter().map(|r| r.skill / r.stars).sum::<f64>() / band.len() as f64;
            println!("  {lo:>4.1}-{hi:<4.1} n={:<4} mean skill/stars {ratio:.3}", band.len());
        }
    }

    /// Not an assertion — dumps the surface to CSV under `target/surface/` so it can
    /// be plotted. Three files, each a different slice of the same object:
    ///
    /// - `grid.csv`: 305-weighted accuracy over (difficulty, skill) at
    ///   [`REFERENCE_WINDOWS`]. This *is* the surface.
    /// - `bands.csv`: the five timing-band shares plus miss rate against skill at one
    ///   fixed difficulty — the mechanism the surface is built from.
    /// - `windows.csv`: accuracy against skill at one difficulty for several window
    ///   sets, which is what [`window_scalar`] reads horizontally.
    ///
    /// Run with `cargo test surface_dump -- --ignored --nocapture`.
    #[test]
    #[ignore = "writes CSV for plotting rather than asserting"]
    fn surface_dump() {
        use crate::mania_accuracy::expected_counts;
        use crate::mania_windows::{windows_from_great, ManiaJudgement};
        use std::fmt::Write as _;

        let model = ErrorModel::default();
        let dir = std::path::Path::new("target/surface");
        std::fs::create_dir_all(dir).unwrap();

        // Log-spaced in both axes: skill spans orders of magnitude and difficulty is
        // multiplicative in `sigma`, so a linear grid would waste most of its rows.
        let geom = |low: f64, high: f64, steps: usize| -> Vec<f64> {
            (0..steps)
                .map(|i| {
                    let t = i as f64 / (steps - 1) as f64;
                    low * (high / low).powf(t)
                })
                .collect()
        };

        let difficulties = geom(2.0, 20.0, 121);
        let skills = geom(0.5, 60.0, 161);

        let mut grid = String::from("difficulty,skill,accuracy,miss_rate\n");

        for &difficulty in &difficulties {
            for &skill in &skills {
                let units = [JudgementUnit::new(difficulty)];
                let expected = expected_counts(&units, &REFERENCE_WINDOWS, &model, skill);
                writeln!(
                    grid,
                    "{difficulty},{skill},{},{}",
                    expected.custom_accuracy(),
                    expected.get(ManiaJudgement::Miss) / expected.total()
                )
                .unwrap();
            }
        }

        std::fs::write(dir.join("grid.csv"), grid).unwrap();

        // One difficulty, chosen to be the Decoy score's so the plots line up with the
        // pricing reports.
        let difficulty = 13.774;
        let mut bands = String::from("skill,sigma,n320,n300,n200,n100,n50,miss,accuracy\n");

        for &skill in &skills {
            let units = [JudgementUnit::new(difficulty)];
            let expected = expected_counts(&units, &REFERENCE_WINDOWS, &model, skill);
            let total = expected.total();
            let share = |judgement| expected.get(judgement) / total;

            writeln!(
                bands,
                "{skill},{},{},{},{},{},{},{},{}",
                model.sigma(difficulty, skill),
                share(ManiaJudgement::Perfect),
                share(ManiaJudgement::Great),
                share(ManiaJudgement::Good),
                share(ManiaJudgement::Ok),
                share(ManiaJudgement::Meh),
                share(ManiaJudgement::Miss),
                expected.custom_accuracy()
            )
            .unwrap();
        }

        std::fs::write(dir.join("bands.csv"), bands).unwrap();

        // The same slice under different windows. Named by GREAT window since that is
        // the single parameter the rest are derived from.
        let window_sets = [
            ("HR OD7 DT", 30.5_f64),
            ("reference OD8", 40.5),
            ("OD7 DT", 43.0),
            ("EZ OD7 DT", 60.3),
        ];

        let mut windows_csv = String::from("label,great,skill,accuracy\n");

        for (label, great) in window_sets {
            let windows = if (great - 40.5).abs() < 1e-9 {
                REFERENCE_WINDOWS
            } else {
                windows_from_great(great)
            };

            for &skill in &skills {
                let units = [JudgementUnit::new(difficulty)];
                let accuracy = expected_counts(&units, &windows, &model, skill).custom_accuracy();
                writeln!(windows_csv, "{label},{great},{skill},{accuracy}").unwrap();
            }
        }

        std::fs::write(dir.join("windows.csv"), windows_csv).unwrap();

        println!("wrote {} (grid {} x {})", dir.display(), difficulties.len(), skills.len());
    }

    /// Not an assertion — dumps `target/surface/od_grid.csv`: every judgement band's
    /// share over (OD, skill), with and without `EZ`, for plotting as 3D surfaces.
    ///
    /// OD is the interesting third axis because mania's classic scheme treats it
    /// unevenly — GREAT and below shift by `3 * (10 - od)` ms while PERFECT is pinned
    /// at a flat 16 ms. So the 320 surface should be *flat* in OD and the others
    /// should tilt, and `EZ` should be the only thing that ever moves 320. Both
    /// scoring schemes are dumped since lazer interpolates PERFECT over OD instead.
    ///
    /// The difficulty the OD/skill grid is taken at defaults to the Decoy score's
    /// 13.77 stars so the dump reproduces without any setup, but `SURFACE_MAP` points
    /// it at a real beatmap instead (its rated difficulty under `SURFACE_CLOCK_RATE`
    /// is used), and `SURFACE_STARS` sets the number directly. `tools/mania_surface.py`
    /// passes these through so any map can be inspected.
    ///
    /// Run with `cargo test od_surface_dump -- --ignored --nocapture`.
    #[test]
    #[ignore = "writes CSV for plotting rather than asserting"]
    fn od_surface_dump() {
        use crate::mania_accuracy::expected_counts;
        use crate::mania_windows::{hit_windows, ManiaJudgement};
        use std::fmt::Write as _;

        let model = ErrorModel::default();
        let dir = std::path::Path::new("target/surface");
        std::fs::create_dir_all(dir).unwrap();

        let env = |key: &str| std::env::var(key).ok().filter(|value| !value.is_empty());
        let clock_rate = env("SURFACE_CLOCK_RATE")
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(1.0);

        // Where the slice is taken. A real map wins over an explicit star value, which
        // wins over the Decoy default.
        let (difficulty, source) = if let Some(path) = env("SURFACE_MAP") {
            let map = parse(&path).unwrap_or_else(|| panic!("cannot parse {path}"));
            let attrs = calculate(&map, &GameMods::default(), clock_rate, Some(true), None)
                .unwrap_or_else(|| panic!("{path} is not a mania map"));

            (attrs.stars, path)
        } else if let Some(stars) = env("SURFACE_STARS").and_then(|v| v.parse::<f64>().ok()) {
            (stars, "SURFACE_STARS".to_owned())
        } else {
            (13.774, "default (Decoy DT)".to_owned())
        };

        println!("slice at {difficulty:.3} stars from {source} (clock rate {clock_rate})");
        std::fs::write(
            dir.join("meta.csv"),
            format!("difficulty,clock_rate,source\n{difficulty},{clock_rate},{source}\n"),
        )
        .unwrap();

        let ods: Vec<f64> = (0..=100).map(|i| f64::from(i) / 10.0).collect();
        let skills: Vec<f64> = (0..161)
            .map(|i| {
                let t = i as f64 / 160.0;
                0.5 * (60.0 / 0.5_f64).powf(t)
            })
            .collect();

        let mut with_ez = LazerMods::new();
        single_mod(&mut with_ez, GameMod::EasyMania(Default::default()));

        let mut out = String::from(
            "scheme,mod,od,skill,great,perfect,sigma,n320,n300,n200,n100,n50,miss,accuracy\n",
        );

        for (scheme, classic) in [("classic", true), ("lazer", false)] {
            for (mod_label, mods) in [("NM", GameMods::default()), ("EZ", with_ez.clone())] {
                for &od in &ods {
                    // A bare non-convert map at this OD; only `od`/`is_convert` reach
                    // the window construction. Converts are deliberately not swept:
                    // their classic scheme keys off a single `round(od) > 4` threshold,
                    // so an OD axis would be two flat plateaus rather than a surface.
                    let mut map = Beatmap::default();
                    map.mode = GameMode::Mania;
                    map.od = od as f32;

                    let windows = hit_windows(&map, &mods, clock_rate, classic);

                    for &skill in &skills {
                        let units = [JudgementUnit::new(difficulty)];
                        let expected = expected_counts(&units, &windows, &model, skill);
                        let total = expected.total();
                        let share = |judgement| expected.get(judgement) / total;

                        writeln!(
                            out,
                            "{scheme},{mod_label},{od},{skill},{},{},{},{},{},{},{},{},{},{}",
                            windows.great,
                            windows.perfect,
                            model.sigma(difficulty, skill),
                            share(ManiaJudgement::Perfect),
                            share(ManiaJudgement::Great),
                            share(ManiaJudgement::Good),
                            share(ManiaJudgement::Ok),
                            share(ManiaJudgement::Meh),
                            share(ManiaJudgement::Miss),
                            expected.custom_accuracy()
                        )
                        .unwrap();
                    }
                }
            }
        }

        std::fs::write(dir.join("od_grid.csv"), out).unwrap();
        println!(
            "wrote od_grid.csv ({} od x {} skill x 2 schemes x 2 mod states)",
            ods.len(),
            skills.len()
        );
    }

    /// Not an assertion — prices one real score with and without EZ, holding the
    /// judgement counts fixed.
    ///
    /// Holding counts fixed is the whole point: it asks "what is this exact
    /// performance worth if it had been produced through wider windows", which is
    /// the question a mod multiplier answers by fiat and the surface answers by
    /// refitting skill. Nothing here inspects the mod list — EZ enters only by
    /// widening [`ManiaHitWindows`], and the pp difference is whatever that
    /// widening does to the fit.
    ///
    /// Run with `cargo test decoy_ez_comparison -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn decoy_ez_comparison() {
        // Yooh - Decoy [Rachel's Ruins "Buffed Ver."], played by Reflec with DT.
        let Some(map) = parse("local-fixtures/maps/4055699.osu") else {
            println!("beatmap 4055699 absent; nothing to compare");
            return;
        };

        let state = SunnyScoreState {
            n320: 1542,
            n300: 1595,
            n200: 603,
            n100: 91,
            n50: 15,
            misses: 14,
        };
        let counts = [
            state.n320,
            state.n300,
            state.n200,
            state.n100,
            state.n50,
            state.misses,
        ];
        let total = state.total_hits();
        let units_for = |stars: f64| [JudgementUnit::repeated(stars, f64::from(total))];
        let model = ErrorModel::default();

        let mut with_ez = LazerMods::new();
        single_mod(&mut with_ez, GameMod::EasyMania(Default::default()));

        let mut rows = Vec::new();

        for (label, mods) in [("DT", GameMods::default()), ("DT+EZ", with_ez)] {
            // Classic (stable) scoring, DT 1.5x, as played.
            let attrs = calculate(&map, &mods, 1.5, Some(false), None).unwrap();
            let fit = fit_with_quality(&counts, &units_for(attrs.stars), &attrs.hit_windows, &model);
            let perf = calculate_performance(&attrs, &mods, state);

            rows.push((label, attrs, fit, perf));
        }

        println!(
            "map: OD {} convert {} | {total} notes | counts 320:{} 300:{} 200:{} 100:{} 50:{} miss:{}",
            map.od, map.is_convert, state.n320, state.n300, state.n200, state.n100, state.n50,
            state.misses
        );
        println!("custom_accuracy {:.3}%\n", custom_accuracy(state) * 100.0);

        println!(
            "{:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "mods", "great", "perfect", "stars", "skill", "sigma", "g_tim"
        );

        for (label, attrs, fit, _) in &rows {
            println!(
                "{label:>7} {:>7.1} {:>7.1} {:>7.3} {:>7.3} {:>7.2} {:>7.1}",
                attrs.hit_windows.great,
                attrs.hit_windows.perfect,
                attrs.stars,
                fit.skill,
                model.sigma(attrs.stars, fit.skill),
                fit.g_timing
            );
        }

        println!(
            "\n{:>7} {:>10} {:>10} {:>10}",
            "mods", "scalar", "pp_diff", "pp"
        );

        for (label, _, _, perf) in &rows {
            println!(
                "{label:>7} {:>10.4} {:>10.1} {:>10.1}",
                perf.window_scalar, perf.pp_difficulty, perf.pp
            );
        }

        let (_, _, _, nm) = &rows[0];
        let (_, _, _, ez) = &rows[1];

        println!(
            "\nEZ prices at {:.4}x the no-mod pp ({:.1} -> {:.1}, {:+.1})",
            ez.pp / nm.pp,
            nm.pp,
            ez.pp,
            ez.pp - nm.pp
        );
        println!(
            "of which the window scalar contributes {:.4}x",
            ez.window_scalar / nm.window_scalar
        );
    }

    /// Not an assertion — a report on *where* the fit misses. Prints each real
    /// score's observed timing-band shares next to what the fitted surface
    /// predicts, so the shape of the residual can be read directly instead of
    /// being inferred from a single `g_timing` number.
    ///
    /// Run with `cargo test residual_shape_report -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn residual_shape_report() {
        use crate::mania_accuracy::expected_counts;
        use crate::mania_windows::ManiaJudgement;

        println!(
            "{:>9} {:>7} {:>6} {:>6} {:>7} {:>39} {:>39}",
            "map",
            "mods",
            "stars",
            "skill",
            "g_tim",
            "observed 320/300/200/100/50",
            "predicted 320/300/200/100/50"
        );

        for row in REAL_SCORES {
            let path = format!("local-fixtures/maps/{}.osu", row.map);
            let Some(map) = parse(&path) else {
                continue;
            };

            let mut mods = LazerMods::new();
            if row.mods.contains("EZ") {
                single_mod(&mut mods, GameMod::EasyMania(Default::default()));
            }
            let clock_rate = if row.mods.contains("DT") { 1.5 } else { 1.0 };

            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(true), None) else {
                continue;
            };

            let counts = [row.n320, row.n300, row.n200, row.n100, row.n50, row.miss];
            let total: u32 = counts.iter().sum();
            let units = [JudgementUnit::repeated(attrs.stars, f64::from(total))];
            let model = ErrorModel::default();
            let fit = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);
            let expected = expected_counts(&units, &attrs.hit_windows, &model, fit.skill);

            // Both sides conditioned on the note having been hit, which is the
            // space the fit actually works in.
            let observed_timing = f64::from(total - row.miss);
            let expected_timing = expected.total() - expected.get(ManiaJudgement::Miss);

            let fmt = |shares: [f64; 5]| {
                shares
                    .iter()
                    .map(|share| format!("{share:>7.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };

            let observed_shares = [
                f64::from(row.n320) / observed_timing,
                f64::from(row.n300) / observed_timing,
                f64::from(row.n200) / observed_timing,
                f64::from(row.n100) / observed_timing,
                f64::from(row.n50) / observed_timing,
            ];
            let predicted_shares = [
                expected.get(ManiaJudgement::Perfect) / expected_timing,
                expected.get(ManiaJudgement::Great) / expected_timing,
                expected.get(ManiaJudgement::Good) / expected_timing,
                expected.get(ManiaJudgement::Ok) / expected_timing,
                expected.get(ManiaJudgement::Meh) / expected_timing,
            ];

            println!(
                "{:>9} {:>7} {:>6.2} {:>6.2} {:>7.1} {} {}",
                row.map,
                if row.mods.is_empty() { "NM" } else { row.mods },
                attrs.stars,
                fit.skill,
                fit.g_timing,
                fmt(observed_shares),
                fmt(predicted_shares),
            );
        }
    }

    /// Not an assertion — a report. Sweeps [`ErrorModel::sigma_floor`] over the
    /// physically motivated 1-5 ms band and prints what each value does to fit
    /// quality *and* to pricing, on the same 20 real scores as
    /// [`real_score_report`].
    ///
    /// The band comes from the client rather than from a fit: osu! judges at 1000
    /// ticks per second, so 1 ms is a hard physical floor on the timing anyone can
    /// resolve, and keyboard scan plus OS scheduling jitter add a few ms on top of
    /// it. That argument is independent of the replay measurement, which is what
    /// makes it worth testing — the replay-derived 10 ms is refuted by judgement
    /// counts (see the `sigma_floor` docs and
    /// `a_sigma_floor_would_forbid_scores_that_exist`) but a value in this band is
    /// not.
    ///
    /// Two things are being separated here. Fit quality asks whether the floor
    /// describes the counts better; pricing asks whether it changes any pp. They
    /// are different questions, and the answers turn out to be "not at all" and
    /// "yes, slightly", which is the least convenient pair.
    ///
    /// The result: `mean_g_timing` is *bit-identical* at 51.552835 across the whole
    /// 0-10 ms sweep, so the judgement counts of these 20 scores cannot see the
    /// floor at any value. It is unfittable here for the same reason `sigma_ref` is
    /// unfittable anywhere — see the comment in the body for the quadrature
    /// arithmetic. Meanwhile the EZ window scalar slides 0.8273 to 0.8123 and total
    /// pp falls 2.7% at 10 ms. Within the 1-5 ms band the pricing effect is
    /// -0.03% to -0.69%, small but not nil.
    ///
    /// Run with `cargo test sigma_floor_sweep -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn sigma_floor_sweep() {
        let scores = load_real_scores();

        if scores.is_empty() {
            println!("no fixtures present; nothing to sweep");
            return;
        }

        // The window scalar with the model's floor overridden. Mirrors
        // `window_scalar`, which takes the default model and so cannot be pointed at
        // a candidate.
        let scalar_with = |score: &LoadedScore, model: &ErrorModel| {
            let total: u32 = score.counts.iter().sum();
            if total == 0 || score.stars <= 0.0 {
                return 1.0;
            }
            let units = [JudgementUnit::repeated(score.stars, f64::from(total))];
            let played = fit_with_quality(&score.counts, &units, &score.windows, model);
            let reference =
                fit_with_quality(&score.counts, &units, &REFERENCE_WINDOWS, model);
            if played.skill <= 0.0 || reference.skill <= 0.0 {
                return 1.0;
            }
            played.skill / reference.skill
        };

        // `mean_g` is printed to six decimals deliberately. The floor is very nearly
        // a gauge parameter on this data — the fit absorbs it into skill almost
        // exactly, the way it absorbs `sigma_ref` perfectly — and only that many
        // digits show the residual movement at all.
        println!(
            "{:>6} {:>13} {:>9} {:>8} {:>8} {:>9} {:>9} {:>8}",
            "floor", "mean_g", "median_g", "plaus", "EZ_scal", "NM_scal", "totalPP", "dPP%"
        );

        let mut baseline_pp = 0.0;

        for floor in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0] {
            let model = ErrorModel {
                sigma_floor: floor,
                ..ErrorModel::default()
            };

            let mut gs = Vec::new();
            let mut ez = Vec::new();
            let mut nm = Vec::new();
            let mut total_pp = 0.0;

            for score in &scores {
                let total: u32 = score.counts.iter().sum();
                let units = [JudgementUnit::repeated(score.stars, f64::from(total))];
                let fit =
                    fit_with_quality(&score.counts, &units, &score.windows, &model);
                gs.push(fit.g_timing);

                let scalar = scalar_with(score, &model);
                if score.mods.contains("EZ") {
                    ez.push(scalar);
                } else {
                    nm.push(scalar);
                }

                let state = SunnyScoreState {
                    n320: score.counts[0],
                    n300: score.counts[1],
                    n200: score.counts[2],
                    n100: score.counts[3],
                    n50: score.counts[4],
                    misses: score.counts[5],
                };

                // Everything except the scalar is floor-independent, so recomposing
                // the difficulty value is enough to see the pp effect.
                total_pp +=
                    compute_difficulty_value(score.stars, custom_accuracy(state), scalar);
            }

            if baseline_pp == 0.0 {
                baseline_pp = total_pp;
            }

            gs.sort_by(f64::total_cmp);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

            println!(
                "{floor:>6.1} {:>13.6} {:>9.1} {:>8} {:>8.4} {:>9.4} {:>9.1} {:>+8.2}",
                mean(&gs),
                gs[gs.len() / 2],
                gs.iter().filter(|g| **g < 30.0).count(),
                mean(&ez),
                mean(&nm),
                total_pp,
                100.0 * (total_pp / baseline_pp - 1.0),
            );
        }

        // Why `mean_g` does not move: the counts pin *sigma*, and the fit is free to
        // move skill, so a floor is absorbed by shrinking the skill term to keep
        // `hypot(floor, skill_term)` where the counts want it. At a 16 ms sigma a 2 ms
        // floor needs the skill term to fall to 15.875 ms — a 0.78% change, which
        // skill^-1.7 supplies exactly. The floor only becomes visible once the skill
        // term is itself small (a 2 ms floor inflates a 2 ms skill term by 41%), which
        // is the saturating regime the ladder's `acc between 88 and 99.5` filter
        // excludes by construction.
        //
        // The scalar moves anyway, and that asymmetry is the whole problem. It is a
        // ratio of skills fitted at two *different* window sets, hence two different
        // sigmas, and quadrature is nonlinear — the two skills do not scale by a
        // common factor, so the ratio shifts even though every `g_timing` is
        // unchanged. A floor is therefore unconstrained by this data while still
        // repricing it, which is a worse position than either fitting it or leaving
        // it out.

        // The ceiling a floor imposes, which is the constraint that killed 10 ms.
        // Independent of skill: it is what the model allows at infinite skill.
        println!("\nmax reachable 320 share at infinite skill (OD8, 16ms PERFECT):");
        let windows = crate::mania_windows::windows_from_great(40.0);
        for floor in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let model = ErrorModel {
                sigma_floor: floor,
                ..ErrorModel::default()
            };
            let units = [JudgementUnit::repeated(2.0, 1506.0)];
            let counts = crate::mania_accuracy::expected_counts(
                &units, &windows, &model, 1.0e4,
            );
            let share =
                counts.get(crate::mania_windows::ManiaJudgement::Perfect) / 1506.0;
            println!(
                "  {floor:>4.1} ms -> {:>7.3}%  ({:>6.2} of 1506 notes forced off 320)",
                share * 100.0,
                1506.0 * (1.0 - share)
            );
        }
    }

    #[test]
    fn classic_flag_uses_head_only_density() {
        let Some(map) = parse(MAP_1638954) else {
            return;
        };
        let mods = GameMods::default();

        let lazer = calculate(&map, &mods, 1.0, Some(true), None).unwrap();
        let stable = calculate(&map, &mods, 1.0, Some(false), None).unwrap();

        // The values may differ slightly between lazer and stable (classic)
        // plays because of the density weighting.
        assert!(stable.stars > 0.0 && lazer.stars > 0.0);
    }

    /// A row of `local-fixtures/multiuser.tsv`: one real score from the prod tRPC
    /// API, carrying enough to name the beatmap in a report as well as price it.
    struct MultiRow {
        uid: String,
        map_id: String,
        mods: String,
        live_stars: f64,
        keys: u32,
        counts: [u32; 6],
        acc: f64,
        live_pp: f64,
        title: String,
        version: String,
    }

    /// One priced score: what the surface makes of a [`MultiRow`].
    struct MultiPriced {
        row: MultiRow,
        stars: f64,
        od: f32,
        is_convert: bool,
        before_pp: f64,
        after_pp: f64,
        scalar: f64,
        skill: f64,
        g_timing: f64,
        plausible: bool,
        notes: u32,
        /// The map's long-note share, and the axis the LN mixture actually acts on.
        /// Key count only stands in for it — 7K charts here average 58% long notes
        /// against 4K's 3% — so grouping by this separates the mechanism from the
        /// convention.
        ln_fraction: f64,
        /// Whether the score's long notes were judged as one unit (V1) or two (V2).
        ln_judged_as_one: bool,
    }

    /// Reproduces the pre-change pp for one score: the flat `EZ` `0.90` that
    /// `calculate_performance` used to apply, and no window scalar.
    ///
    /// Kept here rather than behind a flag in the shipping code because the old
    /// behaviour is not something the calculator should still be able to do — the
    /// report needs it only as a baseline to diff against. Mirrors `2c2e8a1`'s
    /// `calculate_performance` exactly: same multiplier stack, `window_scalar` of 1.
    fn pp_before_change(
        attrs: &SunnyManiaDifficultyAttributes,
        mods: &GameMods,
        state: SunnyScoreState,
    ) -> f64 {
        let mut multiplier = 1.0;
        if has_mod(mods, "NF") {
            multiplier *= 0.75;
        }
        if has_mod(mods, "EZ") {
            multiplier *= 0.90;
        }

        let score_accuracy = custom_accuracy(state);

        compute_difficulty_value(attrs.stars, score_accuracy, 1.0)
            * multiplier
            * variety_multiplier(attrs.variety)
            * acc_multiplier(score_accuracy, attrs.acc_scalar)
            * length_multiplier(attrs.n_objects as f64, attrs.stars)
    }

    /// Builds the mod state for a report row from its mod-name string.
    ///
    /// Only mods that reach the sunny path are translated: `EZ` and `HR` scale the
    /// windows, `NF` carries the flat factor, `V2` decides how long notes are judged,
    /// and `DT`/`NC`/`HT` are a clock rate rather than a `GameMod`. `MR` is ignored,
    /// since mirroring does not change difficulty in this calculator.
    ///
    /// `V2` used to be ignored here too, on the grounds that it "only changes the
    /// score number, not the judgements". That is true for rice and false for long
    /// notes, and the fixture set settles it: all 45 V2 rows have a judgement total of
    /// `notes + LN` while 97 of 98 non-V2 rows total `notes`. So V2 splits an LN into
    /// two judgements and V1 combines them, which changes both the count and the
    /// spread — see [`crate::mania_accuracy::LN_SIGMA_SCALE`].
    fn mods_for(names: &str) -> (LazerMods, f64) {
        let mut mods = LazerMods::new();
        if names.contains("V2") {
            single_mod(&mut mods, GameMod::ScoreV2Mania(Default::default()));
        }
        if names.contains("EZ") {
            single_mod(&mut mods, GameMod::EasyMania(Default::default()));
        }
        if names.contains("HR") {
            single_mod(&mut mods, GameMod::HardRockMania(Default::default()));
        }
        if names.contains("NF") {
            single_mod(&mut mods, GameMod::NoFailMania(Default::default()));
        }

        let clock_rate = if names.contains("DT") || names.contains("NC") {
            1.5
        } else if names.contains("HT") {
            0.75
        } else {
            1.0
        };

        (mods, clock_rate)
    }

    /// Reads `local-fixtures/multiuser.tsv` and prices every row twice.
    fn load_multiuser() -> Vec<MultiPriced> {
        let Ok(text) = std::fs::read_to_string("local-fixtures/multiuser.tsv") else {
            return Vec::new();
        };

        let mut out = Vec::new();

        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 18 || f[0] == "uid" {
                continue;
            }

            let u = |s: &str| s.parse::<u32>().unwrap_or(0);
            let row = MultiRow {
                uid: f[0].to_owned(),
                map_id: f[2].to_owned(),
                mods: f[3].to_owned(),
                live_stars: f[4].parse().unwrap_or(0.0),
                keys: u(f[6]),
                counts: [u(f[7]), u(f[8]), u(f[9]), u(f[10]), u(f[11]), u(f[12])],
                acc: f[13].parse().unwrap_or(0.0),
                live_pp: f[14].parse().unwrap_or(0.0),
                title: f[16].to_owned(),
                version: f[17].to_owned(),
            };

            let Some(map) = parse(&format!("local-fixtures/maps/{}.osu", row.map_id)) else {
                continue;
            };

            let (mods, clock_rate) = mods_for(&row.mods);

            // These are ppy.sb scores, i.e. stable, so `lazer: false`. It used to be
            // `Some(true)` here, which silently made every fixture ScoreV2 and hid the
            // LN judgement regime entirely. The judgement totals settle which is
            // right: a non-V2 row totals `notes`, which is the V1/classic count, and
            // only the V2 rows total `notes + LN`. With `is_classic(Some(false), ..)`
            // the V2 bit in `mods` now selects between the two the same way the server
            // does.
            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(false), None) else {
                continue;
            };

            let state = SunnyScoreState {
                n320: row.counts[0],
                n300: row.counts[1],
                n200: row.counts[2],
                n100: row.counts[3],
                n50: row.counts[4],
                misses: row.counts[5],
            };

            let perf = calculate_performance(&attrs, &mods, state);
            let model = ErrorModel::default();
            let units = judgement_units(&attrs, f64::from(state.total_hits()), &model);
            let fit = fit_with_quality(&row.counts, &units, &attrs.hit_windows, &model);

            out.push(MultiPriced {
                stars: attrs.stars,
                od: map.od,
                is_convert: map.is_convert,
                before_pp: pp_before_change(&attrs, &mods, state),
                after_pp: perf.pp,
                scalar: perf.window_scalar,
                skill: fit.skill,
                g_timing: fit.g_timing,
                plausible: fit.is_plausible(),
                notes: state.total_hits(),
                ln_fraction: if attrs.n_objects > 0 {
                    attrs.n_long_notes as f64 / attrs.n_objects as f64
                } else {
                    0.0
                },
                ln_judged_as_one: attrs.ln_judged_as_one,
                row,
            });
        }

        out
    }

    /// Not an assertion — the cross-user report. Prices every score in
    /// `local-fixtures/multiuser.tsv` under both the pre-change stack
    /// ([`pp_before_change`]: flat `EZ` `0.90`, no window scalar) and the current one
    /// (windows priced, no `EZ` factor), and prints them side by side.
    ///
    /// Why both are computed here rather than read from the API's `pp` column: live
    /// ppy.sb runs sunny, but *a sunny predating this branch*, so its stored figure
    /// differs from our "before" only by version drift in the difficulty calculation
    /// itself. Recomputing the old multiplier stack against today's star ratings
    /// isolates the change under test — the pp delta is then attributable to the
    /// surface alone, with the live column left in as a cross-check on how far the
    /// two sunny versions have otherwise moved.
    ///
    /// Usage:
    /// `cargo test --release multiuser_report -- --ignored --nocapture --exact
    /// sunny::tests::multiuser_report`
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn multiuser_report() {
        use std::collections::BTreeMap;

        let scores = load_multiuser();
        if scores.is_empty() {
            println!("no fixtures present (local-fixtures/multiuser.tsv); nothing to report");
            return;
        }

        let mut by_uid: BTreeMap<&str, Vec<&MultiPriced>> = BTreeMap::new();
        for s in &scores {
            by_uid.entry(s.row.uid.as_str()).or_default().push(s);
        }

        for (uid, rows) in &by_uid {
            let mut rows = rows.clone();
            rows.sort_by(|a, b| b.after_pp.total_cmp(&a.after_pp));

            println!("\n=== uid {uid} ({} scores)", rows.len());
            println!(
                "{:>8} {:>9} {:>4} {:>4} {:>4} {:>6} {:>6} {:>6} {:>26} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7} {:>6} {:>5}",
                "map", "mods", "k", "od", "cvt", "our*", "live*", "notes",
                "320/300/200/100/50/miss", "acc%", "livePP", "beforePP", "afterPP",
                "d%", "scalar", "skill", "plaus"
            );

            for r in &rows {
                let delta = if r.before_pp > 0.0 {
                    (r.after_pp / r.before_pp - 1.0) * 100.0
                } else {
                    0.0
                };
                let composition = format!(
                    "{}/{}/{}/{}/{}/{}",
                    r.row.counts[0],
                    r.row.counts[1],
                    r.row.counts[2],
                    r.row.counts[3],
                    r.row.counts[4],
                    r.row.counts[5]
                );
                println!(
                    "{:>8} {:>9} {:>4} {:>4} {:>4} {:>6.2} {:>6.2} {:>6} {:>26} {:>7.3} {:>8.1} {:>8.1} {:>8.1} {:>+7.2} {:>7.4} {:>6.2} {:>5}",
                    r.row.map_id,
                    r.row.mods,
                    r.row.keys,
                    r.od,
                    r.is_convert,
                    r.stars,
                    r.row.live_stars,
                    r.notes,
                    composition,
                    r.row.acc,
                    r.row.live_pp,
                    r.before_pp,
                    r.after_pp,
                    delta,
                    r.scalar,
                    r.skill,
                    r.plausible
                );
            }

            // Titles are printed separately: they are far too wide for the numeric
            // table but are what makes a row identifiable to a human.
            println!("  beatmaps:");
            for r in rows.iter().take(8) {
                println!(
                    "    {:>8}  {} [{}]",
                    r.row.map_id,
                    truncate(&r.row.title, 52),
                    truncate(&r.row.version, 34)
                );
            }
            if rows.len() > 8 {
                println!("    ... and {} more", rows.len() - 8);
            }

            summarise_group(&format!("uid {uid} total"), &rows);
        }

        let all: Vec<&MultiPriced> = scores.iter().collect();

        println!("\n=== overall ({} scores, {} users)", all.len(), by_uid.len());
        summarise_group("all", &all);

        // Split by whether the mod set touches the hit windows. This is the axis the
        // change acts on: EZ/HR scale the windows and so move the scalar, while
        // DT/MR/V2/NF leave them at the map's own values and can only move through
        // OD's distance from the OD-8 reference.
        println!("\nby window-affecting mod:");
        type Pred = fn(&&MultiPriced) -> bool;
        for (label, pred) in [
            ("EZ (windows widened)", (|r| r.row.mods.contains("EZ")) as Pred),
            ("HR (windows narrowed)", |r| r.row.mods.contains("HR")),
            ("no window mod", |r| {
                !r.row.mods.contains("EZ") && !r.row.mods.contains("HR")
            }),
        ] {
            let group: Vec<&MultiPriced> = all.iter().copied().filter(pred).collect();
            summarise_group(label, &group);
        }

        // OD bands, for the no-window-mod scores only: there the scalar is OD alone,
        // so this is the cleanest read on how much the OD-8 reference choice costs or
        // pays an ordinary score.
        println!("\nno-window-mod scores by OD (scalar is OD alone here):");
        let plain: Vec<&MultiPriced> = all
            .iter()
            .copied()
            .filter(|r| !r.row.mods.contains("EZ") && !r.row.mods.contains("HR"))
            .collect();
        for (lo, hi) in [(0.0, 7.0), (7.0, 7.9), (7.9, 8.1), (8.1, 8.9), (8.9, 11.0)] {
            let band: Vec<&MultiPriced> = plain
                .iter()
                .copied()
                .filter(|r| f64::from(r.od) >= lo && f64::from(r.od) < hi)
                .collect();
            if band.is_empty() {
                continue;
            }
            let n = band.len() as f64;
            let scalars: Vec<f64> = band.iter().map(|r| r.scalar).collect();
            println!(
                "  OD {lo:>4.1}-{hi:<4.1} n={:<4} mean scalar {:.4} ({:.4}..{:.4})  mean dPP {:+.2}%",
                band.len(),
                scalars.iter().sum::<f64>() / n,
                scalars.iter().copied().fold(f64::INFINITY, f64::min),
                scalars.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                band.iter()
                    .map(|r| (r.after_pp / r.before_pp - 1.0) * 100.0)
                    .sum::<f64>()
                    / n
            );
        }

        // Key count, which turns out to matter far more than it looks like it should.
        // It is not that the surface treats 4k and 7k differently — it does not know
        // the key count at all — but that the two populations chart at different OD,
        // so a single OD reference lands very differently on each.
        println!("\nby key count (the surface never reads keys; this is OD convention):");
        for keys in [4u32, 5, 6, 7, 8, 9, 10] {
            let band: Vec<&MultiPriced> =
                all.iter().copied().filter(|r| r.row.keys == keys).collect();
            if band.is_empty() {
                continue;
            }
            let n = band.len() as f64;
            let mean_od = band.iter().map(|r| f64::from(r.od)).sum::<f64>() / n;
            let mean_ln = band.iter().map(|r| r.ln_fraction).sum::<f64>() / n;
            summarise_group(
                &format!("{keys}k (mean OD {mean_od:.1}, LN {:.0}%)", 100.0 * mean_ln),
                &band,
            );

            // 7k's shortfall could be its low OD or its long notes, which the key-count
            // grouping alone conflates. Splitting the band separates them: the rice-heavy
            // rows carry OD only, the LN-heavy rows carry both.
            if band.len() >= 8 {
                for (sub, lo, hi) in [("  rice <30% LN", 0.0, 0.3), ("  LN >=30%", 0.3, 1.01)] {
                    let inner: Vec<&MultiPriced> = band
                        .iter()
                        .copied()
                        .filter(|r| r.ln_fraction >= lo && r.ln_fraction < hi)
                        .collect();
                    if inner.len() >= 3 {
                        let mean_inner_od =
                            inner.iter().map(|r| f64::from(r.od)).sum::<f64>() / inner.len() as f64;
                        summarise_group(
                            &format!("{sub} (mean OD {mean_inner_od:.1})"),
                            &inner,
                        );
                    }
                }
            }
        }

        // Long-note share, which is the axis the LN mixture acts on and the thing key
        // count was standing in for. Under V1 a long note is one judgement over two
        // summed offsets, so an LN-heavy map is a mixture of a narrow and a wide
        // population; fitting a single sigma to that inflates it. Grouping here
        // separates the mechanism from the 4k/7k convention above. Set
        // `SUNNY_NO_LN_SPLIT=1` to price the same rows without the split.
        println!(
            "\nby long-note share (the axis the LN mixture acts on; split {}):",
            if ln_split_disabled() {
                "DISABLED"
            } else {
                "on"
            }
        );
        for (lo, hi) in [(0.0, 0.05), (0.05, 0.3), (0.3, 0.6), (0.6, 1.01)] {
            let band: Vec<&MultiPriced> = all
                .iter()
                .copied()
                .filter(|r| r.ln_fraction >= lo && r.ln_fraction < hi)
                .collect();
            if band.is_empty() {
                continue;
            }
            let v1 = band.iter().filter(|r| r.ln_judged_as_one).count();
            summarise_group(
                &format!(
                    "LN {:>3.0}-{:<3.0}% ({v1}/{} judged V1)",
                    100.0 * lo,
                    100.0 * hi,
                    band.len()
                ),
                &band,
            );
        }

        // Our star rating against the live server's, which is the other half of the
        // gap between `beforePP` and the `livePP` column: the two sunny versions
        // disagree on difficulty as well as on the multiplier stack.
        let drift: Vec<f64> = all
            .iter()
            .filter(|r| r.row.live_stars > 0.0)
            .map(|r| r.stars / r.row.live_stars)
            .collect();
        if !drift.is_empty() {
            let n = drift.len() as f64;
            let mut sorted = drift.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "\nstar rating ours/live: mean {:.4} median {:.4} range {:.3}..{:.3} (n={})",
                drift.iter().sum::<f64>() / n,
                sorted[sorted.len() / 2],
                sorted[0],
                sorted[sorted.len() - 1],
                sorted.len()
            );
            println!(
                "  (live runs a sunny predating this branch, so this is version drift in the \
                 difficulty calc, not the change under test)"
            );
        }

        let mut g: Vec<f64> = all.iter().map(|r| r.g_timing).collect();
        g.sort_by(f64::total_cmp);
        println!(
            "\nfit quality: g_timing median {:.1} p90 {:.1}, plausible {}/{}",
            g[g.len() / 2],
            g[g.len() * 9 / 10],
            all.iter().filter(|r| r.plausible).count(),
            all.len()
        );
    }

    /// Sweeps [`ErrorModel::release_sigma_ratio`] and reports fit quality by long-note
    /// share, testing whether a release is harder to place than a press.
    ///
    /// The question this settles: `sqrt(2)` assumes a release lands as precisely as a
    /// press, and players say it does not. The sweep prices the same fixtures at
    /// ratios from 1.0 (no asymmetry) upward and watches median `g_timing` on the
    /// LN-heavy bands, where the parameter is the only thing moving.
    ///
    /// Read the LN 0-5% row as the control: the split cannot touch those maps, so any
    /// movement there would mean the sweep is leaking into rice scores and the
    /// mechanism is not what it claims to be.
    ///
    /// Unlike `sigma_floor_sweep`, which found its parameter unidentifiable because
    /// skill absorbs it exactly, this one changes the *ratio* between two populations
    /// inside a single map, which skill cannot reproduce. So it should be visible here
    /// or nowhere.
    ///
    /// Run with `cargo test --release ln_release_ratio_sweep -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn ln_release_ratio_sweep() {
        use crate::mania_accuracy::ln_sigma_scale;

        let cases = load_ln_cases();

        if cases.is_empty() {
            println!("no fixtures present (local-fixtures/multiuser.tsv); nothing to sweep");
            return;
        }

        // The first band is exclusive of zero on purpose. A "0-5% LN" band is *not* a
        // control: only 24 of the 88 fixtures under 5% have no long notes at all, and
        // the other 64 have a handful, so the ratio does reach them and the band moves.
        // The true control is `n_long_notes == 0` (plus every V2 score), reported
        // separately below.
        let bands: [(f64, f64); 4] = [(1e-9, 0.05), (0.05, 0.3), (0.3, 0.6), (0.6, 1.01)];

        println!(
            "{} cases, {} with a V1 long-note population",
            cases.len(),
            cases.iter().filter(|c| c.has_ln_effect()).count()
        );
        print!("{:>6} {:>7}  {:>13}", "ratio", "scale", "CONTROL");
        for (lo, hi) in bands {
            print!("  {:>13}", format!("LN{:.0}-{:.0}%", 100.0 * lo, 100.0 * hi));
        }
        println!("  {:>13}  {:>9}", "all V1+LN", "plaus");
        println!("{}", "-".repeat(6 + 7 + 4 * 15 + 15 + 11));

        for ratio in [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0] {
            let model = ErrorModel {
                release_sigma_ratio: ratio,
                short_hold_penalty: 0.0,
                ..Default::default()
            };

            // Median g_timing over a subset, refitting each case under `model`.
            let median_g = |subset: &[&LnCase]| -> Option<f64> {
                let mut gs: Vec<f64> = subset
                    .iter()
                    .map(|c| {
                        let total: u32 = c.counts.iter().sum();
                        let units = ln_units_for(c, f64::from(total), &model);
                        fit_with_quality(&c.counts, &units, &c.windows, &model).g_timing
                    })
                    .collect();

                if gs.is_empty() {
                    return None;
                }

                gs.sort_by(f64::total_cmp);
                Some(gs[gs.len() / 2])
            };

            print!("{ratio:>6.2} {:>7.3}", ln_sigma_scale(ratio));

            // The genuine control: cases the LN split cannot reach at all, either
            // because the map has no long notes or because V2 judged them separately.
            // This column must be constant to the digit, or the parameter is doing
            // something other than what it claims.
            let control: Vec<&LnCase> = cases.iter().filter(|c| !c.has_ln_effect()).collect();
            match median_g(&control) {
                Some(g) => print!("  {:>13}", format!("{g:.3} (n={})", control.len())),
                None => print!("  {:>13}", "-"),
            }

            for (lo, hi) in bands {
                let band: Vec<&LnCase> = cases
                    .iter()
                    .filter(|c| {
                        c.has_ln_effect() && c.ln_fraction() >= lo && c.ln_fraction() < hi
                    })
                    .collect();

                match median_g(&band) {
                    Some(g) => print!("  {:>13}", format!("{g:.1} (n={})", band.len())),
                    None => print!("  {:>13}", "-"),
                }
            }

            // Only the cases the parameter can reach, which is the figure to minimise.
            let affected: Vec<&LnCase> = cases.iter().filter(|c| c.has_ln_effect()).collect();
            let plausible = affected
                .iter()
                .filter(|c| {
                    let total: u32 = c.counts.iter().sum();
                    let units = ln_units_for(c, f64::from(total), &model);
                    fit_with_quality(&c.counts, &units, &c.windows, &model).is_plausible()
                })
                .count();

            match median_g(&affected) {
                Some(g) => print!("  {:>13}", format!("{g:.1} (n={})", affected.len())),
                None => print!("  {:>13}", "-"),
            }
            println!("  {:>9}", format!("{plausible}/{}", affected.len()));
        }

        println!(
            "\nCONTROL is the cases the split cannot reach (no long notes, or V2 judging); \
             it must be constant."
        );
        println!(
            "scale is sqrt(1 + ratio^2), the widening a V1 long note gets; ratio 1.00 is the \
             derived sqrt(2)."
        );

        // ---------------------------------------------------------------
        // Phase two: does making the ratio depend on hold duration help?
        // ---------------------------------------------------------------
        //
        // Phase one wanted two different ratios on two different LN populations, which
        // one number cannot supply. The hypothesis is that duration is the missing axis:
        // a short hold gives the player no time to reset before the release is due, so
        // its release should be wider than a long hold's. If that is right, a nonzero
        // penalty should beat every flat ratio above.
        println!("\n=== short-hold surcharge (penalty x decay scale) ===");
        println!(
            "ratio(t) = release_ratio * (1 + penalty * exp(-t / scale)); penalty 0 is phase one"
        );

        let bands_by_median: [(&str, f64, f64); 3] =
            [("short", 0.0, 90.0), ("mid", 90.0, 160.0), ("long", 160.0, 1e9)];

        print!("{:>7} {:>7} {:>6}", "penalty", "scale", "base");
        for (label, _, _) in bands_by_median {
            print!("  {:>13}", format!("medLN {label}"));
        }
        println!("  {:>13}  {:>9}", "all V1+LN", "plaus");

        for &(penalty, scale) in &[
            (0.0, 120.0),
            (0.4, 120.0),
            (0.8, 120.0),
            (0.8, 250.0),
            (1.5, 120.0),
            (1.5, 250.0),
            (2.5, 150.0),
            (4.0, 150.0),
        ] {
            for base in [1.0, 1.5] {
                let model = ErrorModel {
                    release_sigma_ratio: base,
                    short_hold_penalty: penalty,
                    short_hold_scale: scale,
                    ..Default::default()
                };

                let median_g = |subset: &[&LnCase]| -> Option<f64> {
                    let mut gs: Vec<f64> = subset
                        .iter()
                        .map(|c| {
                            let total: u32 = c.counts.iter().sum();
                            let units = ln_units_for(c, f64::from(total), &model);
                            fit_with_quality(&c.counts, &units, &c.windows, &model).g_timing
                        })
                        .collect();
                    if gs.is_empty() {
                        return None;
                    }
                    gs.sort_by(f64::total_cmp);
                    Some(gs[gs.len() / 2])
                };

                print!("{penalty:>7.2} {scale:>7.0} {base:>6.2}");

                // Grouped by the map's *median* hold length, since that is the quantity
                // the surcharge keys off — unlike LN share, which says nothing about
                // whether the holds are taps or half-second presses.
                for (_, lo, hi) in bands_by_median {
                    let band: Vec<&LnCase> = cases
                        .iter()
                        .filter(|c| {
                            if !c.has_ln_effect() {
                                return false;
                            }
                            let mut d = c.ln_durations.clone();
                            if d.is_empty() {
                                return false;
                            }
                            d.sort_by(f64::total_cmp);
                            let median = d[d.len() / 2];
                            median >= lo && median < hi
                        })
                        .collect();

                    match median_g(&band) {
                        Some(g) => print!("  {:>13}", format!("{g:.1} (n={})", band.len())),
                        None => print!("  {:>13}", "-"),
                    }
                }

                let affected: Vec<&LnCase> =
                    cases.iter().filter(|c| c.has_ln_effect()).collect();
                let plausible = affected
                    .iter()
                    .filter(|c| {
                        let total: u32 = c.counts.iter().sum();
                        let units = ln_units_for(c, f64::from(total), &model);
                        fit_with_quality(&c.counts, &units, &c.windows, &model).is_plausible()
                    })
                    .count();

                match median_g(&affected) {
                    Some(g) => print!("  {:>13}", format!("{g:.1} (n={})", affected.len())),
                    None => print!("  {:>13}", "-"),
                }
                println!("  {:>9}", format!("{plausible}/{}", affected.len()));
            }
        }
    }

    /// `is_classic` must actually see the ScoreV2 mod.
    ///
    /// Regression test for a silent failure: [`has_mod`] *parses* the acronym string, so
    /// a wrong one is not a compile error and not a panic — it simply never matches. The
    /// code asked for `"V2"` where `rosu_mods::ScoreV2Mania` reports `"SV2"`, so every
    /// score was classified as ScoreV1. That was harmless while V2 only changed the score
    /// number, and became a real bug the moment long notes were judged differently under
    /// the two schemes.
    ///
    /// Asserts against the mod's own acronym rather than a literal, so this cannot drift
    /// with the mod crate.
    #[test]
    fn classic_detection_sees_the_score_v2_mod() {
        assert_eq!(
            rosu_mods::generated_mods::ScoreV2Mania::acronym().as_str(),
            "SV2",
            "the acronym this code looks up must be the one the mod reports"
        );

        let mut v2 = LazerMods::new();
        single_mod(&mut v2, GameMod::ScoreV2Mania(Default::default()));

        assert!(
            !is_classic(Some(false), &v2),
            "a stable score with ScoreV2 judges long notes as two units, so it is not classic"
        );
        assert!(
            is_classic(Some(false), &LazerMods::new()),
            "a stable score without ScoreV2 is classic"
        );

        // The lazer default is the other direction, and CL overrides it.
        assert!(!is_classic(Some(true), &LazerMods::new()));

        let mut cl = LazerMods::new();
        single_mod(&mut cl, GameMod::ClassicMania(Default::default()));
        assert!(is_classic(Some(true), &cl));
    }

    /// Whether the surface infers *less skill* from the same player on long-note charts
    /// — the "爆黄" complaint, stated in the only form the model can be wrong about.
    ///
    /// 爆黄 is 320 -> 300: on LN-heavy charts players cannot convert PERFECTs no matter
    /// how well they play, and the surplus lands in the yellow 300. This never appears as
    /// a *residual*, because skill is free per score and the fit simply answers a lower
    /// number — a 320/300 ratio the model finds surprising becomes "this player is worse",
    /// not "this score fits badly". So goodness of fit cannot see the complaint at all,
    /// and every `g_timing` figure in the other harnesses is silent about it.
    ///
    /// What it does do is move pricing. If a player's fitted skill falls as LN share
    /// rises, the surface is charging them for a structural property of the chart, which
    /// is exactly what the LN widening exists to undo. The question is whether it undoes
    /// enough of it, and the per-player slope answers that: within one player, true skill
    /// is roughly constant across their top plays, so any systematic trend against LN
    /// share is the model's, not the player's.
    ///
    /// Confounded in one direction worth stating: a player genuinely weaker at LN will
    /// show a real negative slope too, and this cannot separate that from a modelling
    /// artefact. What it *can* do is show whether the LN split moves the slope toward
    /// zero, which is the thing under our control. Run it with and without
    /// `SUNNY_NO_LN_SPLIT=1` to see that.
    ///
    /// Run with `cargo test --release ln_skill_slope -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn ln_skill_slope() {
        use std::collections::BTreeMap;

        let Ok(text) = std::fs::read_to_string("local-fixtures/multiuser.tsv") else {
            println!("no fixtures present; nothing to report");
            return;
        };

        struct Point {
            uid: String,
            ln_share: f64,
            median_hold: f64,
            /// Skill as a multiple of the map's difficulty, which is the scale-free form.
            /// Comparing raw skill across maps of different star rating would mostly
            /// measure which maps the player chose.
            skill_ratio: f64,
            perfect_share: f64,
        }

        let model = ErrorModel::default();
        let mut points = Vec::new();

        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 18 || f[0] == "uid" {
                continue;
            }

            let u = |s: &str| s.parse::<u32>().unwrap_or(0);
            let counts = [u(f[7]), u(f[8]), u(f[9]), u(f[10]), u(f[11]), u(f[12])];
            let total: u32 = counts.iter().sum();

            let Some(map) = parse(&format!("local-fixtures/maps/{}.osu", f[2])) else {
                continue;
            };

            let (mods, clock_rate) = mods_for(f[3]);
            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(false), None) else {
                continue;
            };

            // Only V1 scores: under V2 the head and release are separate judgements and
            // the mechanism does not apply, so mixing them in would dilute the slope.
            if !attrs.ln_judged_as_one || total == 0 || attrs.stars <= 0.0 {
                continue;
            }

            let units = judgement_units(&attrs, f64::from(total), &model);
            let fit = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);

            let total_columns = map.cs.round_ties_even().max(1.0) as usize;
            let (notes, _) = build_notes(clock_rate, map.hit_objects.iter(), total_columns);
            let mut holds: Vec<f64> = notes
                .iter()
                .filter_map(|n| {
                    let d = n.tail_or_head() - n.head;
                    (d > 0.0).then_some(d)
                })
                .collect();
            holds.sort_by(f64::total_cmp);

            let timing: f64 = counts[..5].iter().map(|&c| f64::from(c)).sum();

            points.push(Point {
                uid: f[0].to_owned(),
                ln_share: if attrs.n_objects > 0 {
                    attrs.n_long_notes as f64 / attrs.n_objects as f64
                } else {
                    0.0
                },
                median_hold: if holds.is_empty() {
                    0.0
                } else {
                    holds[holds.len() / 2]
                },
                skill_ratio: fit.skill / attrs.stars,
                perfect_share: if timing > 0.0 {
                    f64::from(counts[0]) / timing
                } else {
                    0.0
                },
            });
        }

        if points.is_empty() {
            println!("no V1 scores in the fixture set; nothing to report");
            return;
        }

        println!(
            "LN split is {}. {} V1 scores.",
            if ln_split_disabled() { "DISABLED" } else { "on" },
            points.len()
        );
        println!(
            "\n320 share and fitted skill/stars against LN share, per player.\n\
             A negative skill/stars trend means the surface reads an LN chart as the \
             player being worse."
        );

        let mut by_uid: BTreeMap<&str, Vec<&Point>> = BTreeMap::new();
        for point in &points {
            by_uid.entry(point.uid.as_str()).or_default().push(point);
        }

        for (uid, rows) in &by_uid {
            println!("\n=== uid {uid} ({} V1 scores)", rows.len());
            println!(
                "{:>14} {:>5}  {:>13}  {:>13}  {:>13}",
                "LN share", "n", "320 share", "skill/stars", "med hold ms"
            );

            for (lo, hi) in [(0.0, 0.05), (0.05, 0.4), (0.4, 0.75), (0.75, 1.01)] {
                let band: Vec<&&Point> = rows
                    .iter()
                    .filter(|p| p.ln_share >= lo && p.ln_share < hi)
                    .collect();
                if band.is_empty() {
                    continue;
                }
                let n = band.len() as f64;
                let mean = |get: &dyn Fn(&Point) -> f64| -> f64 {
                    band.iter().map(|p| get(p)).sum::<f64>() / n
                };
                println!(
                    "{:>13}% {:>5}  {:>13.4}  {:>13.4}  {:>13.0}",
                    format!("{:.0}-{:.0}", 100.0 * lo, 100.0 * hi),
                    band.len(),
                    mean(&|p| p.perfect_share),
                    mean(&|p| p.skill_ratio),
                    mean(&|p| p.median_hold),
                );
            }

            // Least-squares slope of skill/stars on LN share, within this player. The
            // sign is the whole point; the magnitude says how much pp is at stake.
            let n = rows.len() as f64;
            let mean_x = rows.iter().map(|p| p.ln_share).sum::<f64>() / n;
            let mean_y = rows.iter().map(|p| p.skill_ratio).sum::<f64>() / n;
            let covariance: f64 = rows
                .iter()
                .map(|p| (p.ln_share - mean_x) * (p.skill_ratio - mean_y))
                .sum();
            let variance: f64 = rows.iter().map(|p| (p.ln_share - mean_x).powi(2)).sum();

            if variance > 1e-9 {
                let slope = covariance / variance;
                println!(
                    "  slope d(skill/stars)/d(LN share) = {slope:+.4}  \
                     (mean skill/stars {mean_y:.4}, so {:+.1}% across the full LN range)",
                    100.0 * slope / mean_y
                );
            }
        }
    }

    /// Where the residual misfit on short-hold maps actually lives, judgement by
    /// judgement.
    ///
    /// The surcharge sweep says short-hold maps fit worst (median `g_timing` ~40 against
    /// ~25 for long-hold maps) but that widening their sigma does not help. That is only
    /// consistent with the *shape* being wrong rather than the width, so this prints
    /// observed against predicted shares per judgement to see which band the model misses.
    ///
    /// A width error and a shape error look different here: too narrow a sigma
    /// underpredicts every band below 320 together, while a shape error misses one band
    /// in one direction and another in the other, which no single sigma can fix.
    ///
    /// Run with `cargo test --release ln_shape_residuals -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn ln_shape_residuals() {
        use crate::mania_accuracy::expected_counts;

        let cases = load_ln_cases();

        if cases.is_empty() {
            println!("no fixtures present; nothing to report");
            return;
        }

        let model = ErrorModel::default();
        let bands: [(&str, f64, f64); 3] =
            [("short <90ms", 0.0, 90.0), ("mid 90-160", 90.0, 160.0), ("long >160", 160.0, 1e9)];

        println!(
            "observed / predicted judgement shares, LN maps grouped by median hold length"
        );
        println!(
            "{:>12} {:>5}  {:>15} {:>15} {:>15} {:>15} {:>15}",
            "group", "n", "320", "300", "200", "100", "50"
        );

        for (label, lo, hi) in bands {
            let group: Vec<&LnCase> = cases
                .iter()
                .filter(|c| {
                    if !c.has_ln_effect() || c.ln_durations.is_empty() {
                        return false;
                    }
                    let mut d = c.ln_durations.clone();
                    d.sort_by(f64::total_cmp);
                    let median = d[d.len() / 2];
                    median >= lo && median < hi
                })
                .collect();

            if group.is_empty() {
                continue;
            }

            // Pooled over the group, conditioned on the note having been hit — the same
            // conditioning the fit uses, so the comparison is against what was fitted.
            let mut observed = [0.0; 5];
            let mut predicted = [0.0; 5];

            for case in &group {
                let total: u32 = case.counts.iter().sum();
                let units = ln_units_for(case, f64::from(total), &model);
                let fit = fit_with_quality(&case.counts, &units, &case.windows, &model);
                let expected = expected_counts(&units, &case.windows, &model, fit.skill);

                let obs_timing: f64 =
                    case.counts[..5].iter().map(|&c| f64::from(c)).sum();
                let exp_array = expected.as_array();
                let exp_timing: f64 = exp_array[..5].iter().sum();

                if obs_timing <= 0.0 || exp_timing <= 0.0 {
                    continue;
                }

                for judgement in 0..5 {
                    observed[judgement] += f64::from(case.counts[judgement]) / obs_timing;
                    predicted[judgement] += exp_array[judgement] / exp_timing;
                }
            }

            let n = group.len() as f64;
            print!("{label:>12} {:>5}", group.len());
            for judgement in 0..5 {
                print!(
                    "  {:>15}",
                    format!(
                        "{:.3}/{:.3}",
                        observed[judgement] / n,
                        predicted[judgement] / n
                    )
                );
            }
            println!();
        }

        println!(
            "\nA pure width error misses every sub-320 band the same way; a shape error \
             misses them in opposite directions."
        );
    }

    /// What the duration binning costs against evaluating every long note at its own
    /// duration.
    ///
    /// The bins are a quadrature grid over a continuous function, so the question is not
    /// whether they are "correct" but whether the discretisation error is small next to
    /// the effect being measured. Asserts rather than prints, because a silent drift here
    /// would invalidate every figure the sweep produces.
    ///
    /// Deliberately run at a *large* surcharge, where the function varies most across a
    /// bin and the approximation is at its worst. If it holds there it holds everywhere
    /// milder.
    #[test]
    #[ignore = "reads gitignored fixtures"]
    fn ln_binning_error_stays_small() {
        let cases = load_ln_cases();

        if cases.is_empty() {
            println!("no fixtures present; nothing to check");
            return;
        }

        let model = ErrorModel {
            release_sigma_ratio: 1.5,
            short_hold_penalty: 2.5,
            short_hold_scale: 150.0,
            ..Default::default()
        };

        let mut worst_skill = 0.0_f64;
        let mut worst_g = 0.0_f64;
        let mut errors: Vec<(f64, usize, f64, f64)> = Vec::new();

        for case in cases.iter().filter(|c| c.has_ln_effect()) {
            let total: u32 = case.counts.iter().sum();
            if total == 0 {
                continue;
            }

            let binned = ln_units_for(case, f64::from(total), &model);
            let exact = ln_units_exact(case, f64::from(total), &model);

            let a = fit_with_quality(&case.counts, &binned, &case.windows, &model);
            let b = fit_with_quality(&case.counts, &exact, &case.windows, &model);

            let error = if b.skill > 0.0 {
                (a.skill / b.skill - 1.0).abs()
            } else {
                0.0
            };

            worst_skill = worst_skill.max(error);
            worst_g = worst_g.max((a.g_timing - b.g_timing).abs());
            errors.push((error, case.n_long_notes, case.ln_fraction(), b.skill));
        }

        let checked = errors.len();
        errors.sort_by(|a, b| b.0.total_cmp(&a.0));

        println!(
            "binning vs exact over {checked} cases: worst skill error {:.3}%, worst g_timing \
             difference {worst_g:.3}",
            100.0 * worst_skill
        );

        // Where the error concentrates matters more than its maximum: a few pathological
        // maps are a different problem from a systematically biased grid.
        let median = errors[checked / 2].0;
        let p90 = errors[checked / 10].0;
        println!(
            "  distribution: median {:.3}%, p90 {:.3}%, over-2% {} of {checked}",
            100.0 * median,
            100.0 * p90,
            errors.iter().filter(|e| e.0 > 0.02).count()
        );
        println!("  worst offenders (error%, nLN, LNshare, skill):");
        for (error, n_ln, share, skill) in errors.iter().take(5) {
            println!(
                "    {:.3}%  nLN={n_ln:<6} share={:.2}  skill={skill:.2}",
                100.0 * error,
                share
            );
        }

        // The typical case is what the grid has to get right, and it does: the median
        // error is ~0.04%, three orders of magnitude under the effect being measured.
        assert!(
            median < 0.005,
            "duration binning must not shift the typical fit at all: median error {:.3}%",
            100.0 * median
        );

        // The tail is bounded but not tiny, and it is bounded for a reason worth stating.
        // Every case above 2% is an LN-saturated map fitted at skill 15-21, i.e. near the
        // saturation ceiling where the likelihood is flat and `skill` is already a lower
        // bound rather than a measurement (see `SKILL_SATURATION_RATIO`). A flat
        // likelihood is exactly where a small change in expected counts moves the argmax
        // a long way, so this is the fit being insensitive, not the grid being wrong.
        // Refining the bins does not help — going from 5 to 8 bins cut the per-bin
        // variation from 32% to under 10% and moved this figure only 4.5% to 3.5%.
        assert!(
            worst_skill < 0.05,
            "duration binning must not shift any fit by more than 5%: got {:.3}%",
            100.0 * worst_skill
        );
        assert!(
            p90 < 0.02,
            "at most a tenth of cases may exceed 2%: p90 is {:.3}%",
            100.0 * p90
        );
    }

    /// The judgement units for one [`LnCase`], mirroring [`judgement_units`] but
    /// driven by a case rather than by live attributes.
    fn ln_units_for(case: &LnCase, total: f64, model: &ErrorModel) -> Vec<JudgementUnit> {
        if !case.has_ln_effect() || case.n_objects == 0 {
            return vec![JudgementUnit::repeated(case.stars, total)];
        }

        let per_object = total / case.n_objects as f64;
        let mut units = Vec::with_capacity(LN_DURATION_BUCKETS + 1);
        let mut ln_total = 0.0;

        for (bin, &count) in case.ln_duration_buckets.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let weight = count as f64 * per_object;
            ln_total += weight;
            units.push(JudgementUnit::long_note(
                case.stars,
                weight,
                model,
                LN_DURATION_REPRESENTATIVES[bin],
            ));
        }

        let rice = (total - ln_total).max(0.0);
        if rice > 0.0 {
            units.push(JudgementUnit::repeated(case.stars, rice));
        }
        units
    }

    /// The exact per-note units for one [`LnCase`]: every long note at its own
    /// duration, with no binning.
    ///
    /// The reference the binned approximation is checked against. Too slow to fit with
    /// in production — a 5000-note map becomes 5000 units and every likelihood
    /// evaluation walks all of them — which is why the shipped path bins.
    fn ln_units_exact(case: &LnCase, total: f64, model: &ErrorModel) -> Vec<JudgementUnit> {
        if !case.has_ln_effect() || case.n_objects == 0 {
            return vec![JudgementUnit::repeated(case.stars, total)];
        }

        let per_object = total / case.n_objects as f64;
        let mut units = Vec::with_capacity(case.ln_durations.len() + 1);

        for &duration in &case.ln_durations {
            units.push(JudgementUnit::long_note(
                case.stars,
                per_object,
                model,
                duration,
            ));
        }

        let rice = (total - per_object * case.ln_durations.len() as f64).max(0.0);
        if rice > 0.0 {
            units.push(JudgementUnit::repeated(case.stars, rice));
        }
        units
    }

    /// One fixture row reduced to what a refit needs, so the sweep below can vary the
    /// model without re-parsing beatmaps for every candidate.
    struct LnCase {
        counts: [u32; 6],
        stars: f64,
        windows: ManiaHitWindows,
        n_objects: usize,
        n_long_notes: usize,
        ln_duration_buckets: [usize; LN_DURATION_BUCKETS],
        /// Every long note's duration in ms, kept so the binned approximation can be
        /// checked against the exact per-note sum.
        ln_durations: Vec<f64>,
        ln_judged_as_one: bool,
    }

    impl LnCase {
        fn ln_fraction(&self) -> f64 {
            if self.n_objects == 0 {
                0.0
            } else {
                self.n_long_notes as f64 / self.n_objects as f64
            }
        }

        /// Whether the LN mixture can act on this case at all: V1 judging, and some
        /// long notes to widen.
        fn has_ln_effect(&self) -> bool {
            self.ln_judged_as_one && self.n_long_notes > 0
        }
    }

    /// Loads `local-fixtures/multiuser.tsv` into refittable cases.
    ///
    /// Deliberately separate from [`load_multiuser`]: that one prices scores through
    /// the full pp stack with the default model, while this keeps the raw inputs so a
    /// sweep can refit them under any [`ErrorModel`].
    fn load_ln_cases() -> Vec<LnCase> {
        let Ok(text) = std::fs::read_to_string("local-fixtures/multiuser.tsv") else {
            return Vec::new();
        };

        let mut out = Vec::new();

        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 18 || f[0] == "uid" {
                continue;
            }

            let u = |s: &str| s.parse::<u32>().unwrap_or(0);
            let counts = [u(f[7]), u(f[8]), u(f[9]), u(f[10]), u(f[11]), u(f[12])];

            let Some(map) = parse(&format!("local-fixtures/maps/{}.osu", f[2])) else {
                continue;
            };

            let (mods, clock_rate) = mods_for(f[3]);
            let Some(attrs) = calculate(&map, &mods, clock_rate, Some(false), None) else {
                continue;
            };

            // Re-derive the durations the same way `calculate` does, so the exact
            // per-note reference and the binned model see identical inputs.
            let total_columns = map.cs.round_ties_even().max(1.0) as usize;
            let (notes, _) = build_notes(clock_rate, map.hit_objects.iter(), total_columns);
            let ln_durations: Vec<f64> = notes
                .iter()
                .filter_map(|n| {
                    let d = n.tail_or_head() - n.head;
                    (d > 0.0).then_some(d)
                })
                .collect();

            out.push(LnCase {
                counts,
                stars: attrs.stars,
                windows: attrs.hit_windows,
                n_objects: attrs.n_objects,
                n_long_notes: attrs.n_long_notes,
                ln_duration_buckets: attrs.ln_duration_buckets,
                ln_durations,
                ln_judged_as_one: attrs.ln_judged_as_one,
            });
        }

        out
    }

    /// Clip a title to `n` chars on a char boundary, since beatmap metadata is
    /// routinely CJK and byte slicing would panic.
    fn truncate(s: &str, n: usize) -> String {
        if s.chars().count() <= n {
            return s.to_owned();
        }
        s.chars().take(n.saturating_sub(1)).collect::<String>() + "…"
    }

    /// Mean before/after pp and scalar for a set of priced scores, plus what the same
    /// set would look like as a weighted bonus-free pp total.
    fn summarise_group(label: &str, rows: &[&MultiPriced]) {
        if rows.is_empty() {
            return;
        }

        let n = rows.len() as f64;
        let before: f64 = rows.iter().map(|r| r.before_pp).sum();
        let after: f64 = rows.iter().map(|r| r.after_pp).sum();
        let mean_scalar = rows.iter().map(|r| r.scalar).sum::<f64>() / n;
        let plausible = rows.iter().filter(|r| r.plausible).count();

        // Against the *live* server as well as against our own before-stack. These answer
        // different questions and only the second one players can feel: `before_pp` is
        // this branch's difficulty calc with the old multiplier stack, so it isolates the
        // surface change, while `live_pp` is what a player actually sees today and
        // therefore what any complaint about pp being too low is about. The two diverge
        // because our star ratings have drifted from live's independently of this branch.
        let with_live: Vec<&&MultiPriced> =
            rows.iter().filter(|r| r.row.live_pp > 0.0).collect();
        let live_note = if with_live.is_empty() {
            String::new()
        } else {
            let live: f64 = with_live.iter().map(|r| r.row.live_pp).sum();
            let after_live: f64 = with_live.iter().map(|r| r.after_pp).sum();
            let mean_ratio = with_live
                .iter()
                .map(|r| r.after_pp / r.row.live_pp)
                .sum::<f64>()
                / with_live.len() as f64;
            format!(
                "  vs live: sum {:+.1}% mean {:+.1}%",
                (after_live / live - 1.0) * 100.0,
                (mean_ratio - 1.0) * 100.0
            )
        };

        // The per-score mean delta and the aggregate delta answer different
        // questions: the first weights every score equally, the second weights by pp
        // and so is what a player's top-play total actually moves by.
        let mean_delta = rows
            .iter()
            .filter(|r| r.before_pp > 0.0)
            .map(|r| (r.after_pp / r.before_pp - 1.0) * 100.0)
            .sum::<f64>()
            / n;

        // Median rather than mean g_timing: the statistic has a long right tail on
        // real scores, so a handful of unexplainable plays would otherwise set the
        // figure for the whole group.
        let mut gs: Vec<f64> = rows.iter().map(|r| r.g_timing).collect();
        gs.sort_by(f64::total_cmp);
        let median_g = gs[gs.len() / 2];

        let n_rows = rows.len();
        let sum_delta = (after / before - 1.0) * 100.0;
        println!(
            "  {label}: n={n_rows} mean scalar {mean_scalar:.4}  mean dPP {mean_delta:+.2}%  \
             sum {before:.0} -> {after:.0} ({sum_delta:+.2}%)  plausible {plausible}/{n_rows}  \
             med g {median_g:.1}{live_note}"
        );
    }

    /// The release-to-next-press *gap*, which is the physical quantity 反键 charting
    /// varies and the one the accuracy surface currently cannot see.
    ///
    /// The surface bins long notes by how long they are *held*
    /// ([`LN_DURATION_EDGES`]) and charges short holds more, on the reasoning that the
    /// press motion has not finished when the release comes due. 反键 inverts that: the
    /// key is held for most of the map and the *release* is the brief event, so the hold
    /// is long — the cheapest bin — while the thing being timed is a gap of a few tens
    /// of milliseconds. If gap and hold length are close to independent across real
    /// maps, then duration binning is not a proxy for gap and the model is blind to it.
    ///
    /// Prints, per map, the median gap alongside the median hold and the share of map
    /// time spent holding, then correlates the two.
    ///
    /// Run with `cargo test --release inverse_gap_structure -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn inverse_gap_structure() {
        use std::fs;

        struct MapShape {
            id: String,
            keys: usize,
            od: f32,
            median_hold: f64,
            median_gap: f64,
            hold_share: f64,
            ln_share: f64,
            /// Long notes whose gap to the next press in the same column is under
            /// 45 ms — the shortest hold bin's own upper edge, so "shorter than the
            /// shortest thing the model treats as short".
            tight_gap_share: f64,
        }

        let Ok(entries) = fs::read_dir("local-fixtures/maps") else {
            println!("no fixture maps present; nothing to report");
            return;
        };

        let mut shapes = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("osu") {
                continue;
            }

            let Some(path_str) = path.to_str() else {
                continue;
            };
            let Some(map) = parse(path_str) else {
                continue;
            };

            let total_columns = map.cs.round_ties_even().max(1.0) as usize;
            let (notes, _) = build_notes(1.0, map.hit_objects.iter(), total_columns);

            if notes.len() < 2 {
                continue;
            }

            // Group by column so "the next press" means the next press the same finger
            // has to make, which is what a release can collide with.
            let mut by_column: Vec<Vec<Note>> = vec![Vec::new(); total_columns];

            for note in &notes {
                if note.column < total_columns {
                    by_column[note.column].push(*note);
                }
            }

            for column in &mut by_column {
                column.sort_by(|a, b| a.head.total_cmp(&b.head));
            }

            let mut holds = Vec::new();
            let mut gaps = Vec::new();
            let mut held_time = 0.0;
            let mut tight = 0usize;

            for column in &by_column {
                for (idx, note) in column.iter().enumerate() {
                    let Some(tail) = note.tail else {
                        continue;
                    };

                    let duration = tail - note.head;
                    holds.push(duration);
                    held_time += duration;

                    if let Some(next) = column.get(idx + 1) {
                        let gap = next.head - tail;

                        if gap >= 0.0 {
                            gaps.push(gap);

                            if gap < 45.0 {
                                tight += 1;
                            }
                        }
                    }
                }
            }

            if holds.is_empty() || gaps.is_empty() {
                continue;
            }

            let first = notes.first().map_or(0.0, |n| n.head);
            let last = notes.iter().map(|n| n.tail_or_head()).fold(0.0, f64::max);
            let span = (last - first).max(1.0);

            let median = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };

            let n_long = holds.len();
            let n_gaps = gaps.len();

            shapes.push(MapShape {
                id: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_owned(),
                keys: total_columns,
                od: map.od,
                median_hold: median(&mut holds),
                median_gap: median(&mut gaps),
                // Held time as a share of one column's worth of map time, which is what
                // "the key is down most of the time" means when averaged over columns.
                hold_share: held_time / (span * total_columns as f64),
                ln_share: n_long as f64 / notes.len() as f64,
                tight_gap_share: tight as f64 / n_gaps as f64,
            });
        }

        if shapes.is_empty() {
            println!("no parseable fixture maps; nothing to report");
            return;
        }

        // Sort by hold share: the top of this list is what 反键 charting looks like
        // numerically, if the fixture set contains any.
        shapes.sort_by(|a, b| b.hold_share.total_cmp(&a.hold_share));

        println!(
            "{} maps. Sorted by share of time held; the top rows are the inverse-style ones.",
            shapes.len()
        );
        println!(
            "{:>9} {:>4} {:>5} {:>11} {:>10} {:>10} {:>9} {:>10}",
            "map", "keys", "od", "median hold", "median gap", "hold share", "ln share", "gap<45ms"
        );

        for shape in shapes.iter().take(15) {
            println!(
                "{:>9} {:>4} {:>5.1} {:>10.0}ms {:>9.0}ms {:>9.1}% {:>8.1}% {:>9.1}%",
                shape.id,
                shape.keys,
                shape.od,
                shape.median_hold,
                shape.median_gap,
                shape.hold_share * 100.0,
                shape.ln_share * 100.0,
                shape.tight_gap_share * 100.0
            );
        }

        // Does hold duration predict gap? If the model's duration bins were a usable
        // proxy for gap tightness, long holds would come with long gaps and this
        // correlation would be strongly positive.
        let n = shapes.len() as f64;
        let log_hold: Vec<f64> = shapes.iter().map(|s| s.median_hold.max(1.0).ln()).collect();
        let log_gap: Vec<f64> = shapes.iter().map(|s| s.median_gap.max(1.0).ln()).collect();

        let mean_h = log_hold.iter().sum::<f64>() / n;
        let mean_g = log_gap.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_h = 0.0;
        let mut var_g = 0.0;

        for (h, g) in log_hold.iter().zip(&log_gap) {
            cov += (h - mean_h) * (g - mean_g);
            var_h += (h - mean_h).powi(2);
            var_g += (g - mean_g).powi(2);
        }

        let corr = cov / (var_h.sqrt() * var_g.sqrt()).max(1e-12);

        println!(
            "\ncorrelation of log median hold with log median gap: {corr:+.3} over {} maps",
            shapes.len()
        );

        let inverse: Vec<&MapShape> = shapes.iter().filter(|s| s.hold_share > 0.5).collect();
        let normal: Vec<&MapShape> = shapes.iter().filter(|s| s.hold_share <= 0.2).collect();

        let summarise = |label: &str, group: &[&MapShape]| {
            if group.is_empty() {
                println!("  {label}: none");
                return;
            }

            let k = group.len() as f64;
            println!(
                "  {label}: n={} median hold {:.0}ms  median gap {:.0}ms  gap<45ms {:.1}%  \
                 mean od {:.1}",
                group.len(),
                group.iter().map(|s| s.median_hold).sum::<f64>() / k,
                group.iter().map(|s| s.median_gap).sum::<f64>() / k,
                group.iter().map(|s| s.tight_gap_share).sum::<f64>() / k * 100.0,
                group.iter().map(|s| f64::from(s.od)).sum::<f64>() / k
            );
        };

        summarise("held >50% of the time", &inverse);
        summarise("held <20% of the time", &normal);

        // What the model charges these maps, to see whether the duration bins happen to
        // catch the inverse maps anyway.
        let model = ErrorModel::default();
        let scale_for = |duration: f64| {
            crate::mania_accuracy::ln_sigma_scale_for_duration(&model, duration)
        };

        println!(
            "\nthe model's LN spread multiplier at each duration bin's representative:"
        );
        for (idx, &rep) in LN_DURATION_REPRESENTATIVES.iter().enumerate() {
            println!("  bin {idx}: {rep:>4.0}ms -> {:.3}x", scale_for(rep));
        }
    }

    /// How often a release's judgement window reaches past the next press in the same
    /// column, which is the collision 反键 charting creates.
    ///
    /// The surface treats every judgement as an independent draw from a timing
    /// distribution. That assumption needs each judgement to have its own window to land
    /// in. When the gap between a release and the next press in the same column is
    /// smaller than the window the release is judged against, the two events compete for
    /// the same interval of time: releasing late enough to still score a 300 can push the
    /// press past its own window, so the player cannot place both independently and has
    /// to sacrifice one. Independent draws cannot represent that, and the model will read
    /// the resulting counts as a less skilled player rather than a harder map.
    ///
    /// Reports gaps against the map's *own* windows, since a low-OD map has wider windows
    /// and so collides at wider gaps — which is the opposite of the fixed reference's
    /// assumption that low OD means lenient.
    ///
    /// Run with `cargo test --release window_overlap_structure -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn window_overlap_structure() {
        use std::fs;

        struct Overlap {
            id: String,
            keys: usize,
            od: f32,
            great: f64,
            good: f64,
            /// Share of releases whose gap to the next press is under the GREAT window,
            /// so a 320-eligible release error can cost the next note its own 320.
            under_great: f64,
            /// Share under the GOOD window, the 200 boundary.
            under_good: f64,
            median_gap: f64,
            hold_share: f64,
            n_releases: usize,
        }

        let Ok(entries) = fs::read_dir("local-fixtures/maps") else {
            println!("no fixture maps present; nothing to report");
            return;
        };

        let mut rows = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("osu") {
                continue;
            }

            let Some(path_str) = path.to_str() else {
                continue;
            };
            let Some(map) = parse(path_str) else {
                continue;
            };

            let total_columns = map.cs.round_ties_even().max(1.0) as usize;
            let (notes, _) = build_notes(1.0, map.hit_objects.iter(), total_columns);

            if notes.len() < 2 {
                continue;
            }

            // The map's own windows, no mods — the same set `reference_windows` now
            // prices against.
            let windows = hit_windows(&map, &GameMods::default(), 1.0, true);

            let mut by_column: Vec<Vec<Note>> = vec![Vec::new(); total_columns];

            for note in &notes {
                if note.column < total_columns {
                    by_column[note.column].push(*note);
                }
            }

            for column in &mut by_column {
                column.sort_by(|a, b| a.head.total_cmp(&b.head));
            }

            let mut gaps = Vec::new();
            let mut held_time = 0.0;

            for column in &by_column {
                for (idx, note) in column.iter().enumerate() {
                    let Some(tail) = note.tail else {
                        continue;
                    };

                    held_time += tail - note.head;

                    if let Some(next) = column.get(idx + 1) {
                        let gap = next.head - tail;

                        if gap >= 0.0 {
                            gaps.push(gap);
                        }
                    }
                }
            }

            if gaps.len() < 20 {
                continue;
            }

            let n = gaps.len();
            let under_great = gaps.iter().filter(|&&g| g < windows.great).count();
            let under_good = gaps.iter().filter(|&&g| g < windows.good).count();

            gaps.sort_by(f64::total_cmp);

            let first = notes.first().map_or(0.0, |n| n.head);
            let last = notes.iter().map(|n| n.tail_or_head()).fold(0.0, f64::max);
            let span = (last - first).max(1.0);

            rows.push(Overlap {
                id: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_owned(),
                keys: total_columns,
                od: map.od,
                great: windows.great,
                good: windows.good,
                under_great: under_great as f64 / n as f64,
                under_good: under_good as f64 / n as f64,
                median_gap: gaps[n / 2],
                hold_share: held_time / (span * total_columns as f64),
                n_releases: n,
            });
        }

        if rows.is_empty() {
            println!("no parseable fixture maps; nothing to report");
            return;
        }

        rows.sort_by(|a, b| b.under_good.total_cmp(&a.under_good));

        println!(
            "{} maps with at least 20 releases, sorted by the share of releases whose \
             next press falls inside the release's own GOOD window.",
            rows.len()
        );
        println!(
            "{:>9} {:>4} {:>5} {:>7} {:>6} {:>10} {:>11} {:>10} {:>10}",
            "map", "keys", "od", "great", "good", "median gap", "gap<great", "gap<good", "held"
        );

        for row in rows.iter().take(15) {
            println!(
                "{:>9} {:>4} {:>5.1} {:>6.1}ms {:>5.1}ms {:>9.0}ms {:>10.1}% {:>9.1}% {:>9.1}%",
                row.id,
                row.keys,
                row.od,
                row.great,
                row.good,
                row.median_gap,
                row.under_great * 100.0,
                row.under_good * 100.0,
                row.hold_share * 100.0
            );
        }

        let summarise = |label: &str, group: &[&Overlap]| {
            if group.is_empty() {
                println!("  {label}: none");
                return;
            }

            let k = group.len() as f64;
            println!(
                "  {label}: n={} mean od {:.1}  median gap {:.0}ms  gap<great {:.1}%  \
                 gap<good {:.1}%",
                group.len(),
                group.iter().map(|r| f64::from(r.od)).sum::<f64>() / k,
                group.iter().map(|r| r.median_gap).sum::<f64>() / k,
                group.iter().map(|r| r.under_great).sum::<f64>() / k * 100.0,
                group.iter().map(|r| r.under_good).sum::<f64>() / k * 100.0
            );
        };

        println!("\nby keymode:");
        for keys in [4, 7] {
            let group: Vec<&Overlap> = rows.iter().filter(|r| r.keys == keys).collect();
            summarise(&format!("{keys}K"), &group);
        }

        println!("\nby how much of the map is spent holding:");
        let held_high: Vec<&Overlap> = rows.iter().filter(|r| r.hold_share > 0.35).collect();
        let held_low: Vec<&Overlap> = rows.iter().filter(|r| r.hold_share <= 0.15).collect();
        summarise("held >35%", &held_high);
        summarise("held <15%", &held_low);

        // Total exposure: how many releases across the whole set are in collision, which
        // decides whether this is a niche correction or a broad one.
        let total: usize = rows.iter().map(|r| r.n_releases).sum();
        let colliding: f64 = rows
            .iter()
            .map(|r| r.under_good * r.n_releases as f64)
            .sum();

        println!(
            "\n{colliding:.0} of {total} releases across the set ({:.1}%) have their next \
             press inside the release's GOOD window.",
            colliding / total as f64 * 100.0
        );
    }

    /// What sunny's own release term says about 反键 spacing.
    ///
    /// [`compute_rbar`] is the one place in the codebase that already reads the
    /// release-to-next-press gap: `i_t = |next_head - tail - 80| / leniency`, combined
    /// with the hold's own `i_h` through
    /// `2 / (2 + exp(-5(i_h - 0.75)) + exp(-5(i_t - 0.75)))`, and the result *multiplies*
    /// the release difficulty. The `- 80.0` centres it, so the term is extremal at a
    /// gap of 80 ms — and 80 ms is 1/4 at 187 bpm, i.e. exactly the spacing dense 反键
    /// charting uses.
    ///
    /// Prints the multiplier against gap to establish which direction it points, since a
    /// term minimised at 反键 spacing would be actively cancelling the difficulty the
    /// pattern creates.
    ///
    /// Run with `cargo test --release rbar_gap_response -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints a report rather than asserting"]
    fn rbar_gap_response() {
        // The same combination `compute_rbar` applies, extracted so the shape can be
        // read off directly.
        let combined = |i_h: f64, i_t: f64| {
            2.0 / (2.0 + (-5.0 * (i_h - 0.75)).exp() + (-5.0 * (i_t - 0.75)).exp())
        };

        println!(
            "sunny's rbar release multiplier `1 + 0.8*i` against release-to-next-press gap,\n\
             at a fixed 150ms hold. Higher = sunny charges more."
        );

        for od in [0.0, 5.0, 8.0] {
            let window = if od <= 0.0 { 64.5 } else { 34.0 + 3.0 * (10.0 - od) };
            let leniency = hit_leniency_from_window(window);

            println!("\n  OD {od:.0} (great {window:.1}ms, leniency {leniency:.4}s):");
            println!("  {:>8} {:>10} {:>12}", "gap", "i", "1 + 0.8i");

            let i_h = 0.001 * (150.0 - 80.0_f64).abs() / leniency;

            for gap in [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 250.0, 500.0, 1000.0] {
                let i_t = 0.001 * (gap - 80.0_f64).abs() / leniency;
                let i = combined(i_h, i_t);

                println!("  {gap:>6.0}ms {i:>10.4} {:>12.4}", 1.0 + 0.8 * i);
            }
        }

        println!(
            "\nFor reference, the accuracy surface's LN spread multiplier over the same\n\
             range of hold durations, to show whether it varies at all by default:"
        );

        let model = ErrorModel::default();

        for duration in [34.0, 84.0, 175.0, 419.0, 900.0] {
            println!(
                "  hold {duration:>4.0}ms -> {:.4}x",
                crate::mania_accuracy::ln_sigma_scale_for_duration(&model, duration)
            );
        }
    }

    /// Whether pp is under-predicted, and the fit is worse, on maps whose
    /// release-to-next-press gaps are tight.
    ///
    /// [`window_overlap_structure`] already showed that a release's gap to the next
    /// press in the same column can fall inside the release's own GOOD window, which
    /// breaks the independent-judgement assumption the whole surface rests on. This
    /// harness asks whether that collision actually shows up as mispricing: it joins
    /// each of `local-fixtures/multiuser.tsv`'s scored plays (via [`load_multiuser`],
    /// the 143-score set with live pp) to its map's median release gap and collision
    /// share, then buckets by each axis and reports mean predicted/live pp ratio and
    /// median `g_timing` per bucket. A monotone drop in the ratio, or a rise in
    /// `g_timing`, toward the tight-gap end would say the model under-rates 反键
    /// charting; a flat table would say the collision is priced fine, or at least not
    /// through pp or fit quality.
    ///
    /// Run with `cargo test --release gap_vs_fit_sweep -- --ignored --nocapture`.
    /// A map's release-gap shape, keyed by map id so every score on the same map
    /// reuses one computation instead of re-parsing the `.osu` per row.
    ///
    /// Shared between [`gap_vs_fit_sweep`] and [`collision_skill_slope`], which both
    /// need the same collision share against the map's own GOOD window.
    struct GapShape {
        median_gap: f64,
        collision_share: f64,
    }

    fn gap_shape_for(map_id: &str) -> Option<GapShape> {
        let map = parse(&format!("local-fixtures/maps/{map_id}.osu"))?;

        let total_columns = map.cs.round_ties_even().max(1.0) as usize;
        let (notes, _) = build_notes(1.0, map.hit_objects.iter(), total_columns);

        if notes.len() < 2 {
            return None;
        }

        // The map's own windows, no mods: the same reference `window_overlap_structure`
        // prices collisions against.
        let windows = hit_windows(&map, &GameMods::default(), 1.0, true);

        let mut by_column: Vec<Vec<Note>> = vec![Vec::new(); total_columns];
        for note in &notes {
            if note.column < total_columns {
                by_column[note.column].push(*note);
            }
        }
        for column in &mut by_column {
            column.sort_by(|a, b| a.head.total_cmp(&b.head));
        }

        let mut gaps = Vec::new();

        for column in &by_column {
            for (idx, note) in column.iter().enumerate() {
                let Some(tail) = note.tail else {
                    continue;
                };

                if let Some(next) = column.get(idx + 1) {
                    let gap = next.head - tail;

                    if gap >= 0.0 {
                        gaps.push(gap);
                    }
                }
            }
        }

        if gaps.len() < 20 {
            return None;
        }

        gaps.sort_by(f64::total_cmp);
        let n = gaps.len();
        let under_good = gaps.iter().filter(|&&g| g < windows.good).count();

        Some(GapShape {
            median_gap: gaps[n / 2],
            collision_share: under_good as f64 / n as f64,
        })
    }

    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn gap_vs_fit_sweep() {
        use std::collections::HashMap;

        let scores = load_multiuser();
        if scores.is_empty() {
            println!("no fixtures present (local-fixtures/multiuser.tsv); nothing to report");
            return;
        }

        struct Point {
            median_gap: f64,
            collision_share: f64,
            pp_ratio: f64,
            g_timing: f64,
        }

        let mut cache: HashMap<String, Option<GapShape>> = HashMap::new();
        let mut points = Vec::new();

        for s in &scores {
            if s.row.live_pp <= 0.0 {
                continue;
            }

            let shape = cache
                .entry(s.row.map_id.clone())
                .or_insert_with(|| gap_shape_for(&s.row.map_id));

            let Some(shape) = shape else {
                continue;
            };

            points.push(Point {
                median_gap: shape.median_gap,
                collision_share: shape.collision_share,
                pp_ratio: s.after_pp / s.row.live_pp,
                g_timing: s.g_timing,
            });
        }

        if points.is_empty() {
            println!("no scores with both live pp and a fitted gap shape; nothing to report");
            return;
        }

        println!(
            "{} scores with live pp, joined to their map's release-gap shape \
             (maps with fewer than 20 releases skipped).",
            points.len()
        );

        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };

        let summarise = |label: &str, group: &[&Point]| {
            if group.is_empty() {
                println!("  {label:>14}: n=0");
                return;
            }

            let n = group.len() as f64;
            let mut g_timings: Vec<f64> = group.iter().map(|p| p.g_timing).collect();

            println!(
                "  {label:>14}: n={:<4} mean gap {:>7.0}ms  mean collision {:>6.1}%  \
                 mean pred/live {:>6.3}  median g_timing {:>7.1}",
                group.len(),
                group.iter().map(|p| p.median_gap).sum::<f64>() / n,
                group.iter().map(|p| p.collision_share).sum::<f64>() / n * 100.0,
                group.iter().map(|p| p.pp_ratio).sum::<f64>() / n,
                median(&mut g_timings),
            );
        };

        println!("\nby median release-to-next-press gap:");
        let gap_edges = [
            ("<80ms", 0.0, 80.0),
            ("80-120ms", 80.0, 120.0),
            ("120-200ms", 120.0, 200.0),
            ("200-400ms", 200.0, 400.0),
            (">=400ms", 400.0, f64::INFINITY),
        ];
        for (label, lo, hi) in gap_edges {
            let group: Vec<&Point> = points
                .iter()
                .filter(|p| p.median_gap >= lo && p.median_gap < hi)
                .collect();
            summarise(label, &group);
        }

        println!("\nby collision share (releases under the map's own GOOD window):");
        let collision_edges = [
            ("<5%", 0.0, 0.05),
            ("5-15%", 0.05, 0.15),
            ("15-30%", 0.15, 0.30),
            (">=30%", 0.30, f64::INFINITY),
        ];
        for (label, lo, hi) in collision_edges {
            let group: Vec<&Point> = points
                .iter()
                .filter(|p| p.collision_share >= lo && p.collision_share < hi)
                .collect();
            summarise(label, &group);
        }
    }

    /// Whether fitted skill reads as *lower* on maps where releases collide with the
    /// next press, within one player.
    ///
    /// [`window_overlap_structure`] found that a meaningful share of releases have
    /// their next same-column press land inside the release's own GOOD window, which
    /// breaks the independent-judgement assumption the fit rests on. [`gap_vs_fit_sweep`]
    /// asked whether that shows up in pp/fit-quality pooled across players; this asks
    /// the sharper question directly of skill, and *within* each player rather than
    /// pooled, for the same reason [`ln_skill_slope`] does: a player's true skill is
    /// roughly constant across their own top plays, so if the fit reads them as *less*
    /// skilled specifically on their higher-collision maps, that is the collision
    /// difficulty being underrated, not the player being worse. Pooling across players
    /// naively — treating every score as one observation of the same slope — would
    /// confound this with players of different ability simply preferring different map
    /// styles, exactly the error the per-player framing avoids.
    ///
    /// The first version of this test avoided that confound by fitting one slope per
    /// player and averaging the three slopes. That is *correct* but wasteful: with 3
    /// players it reports on 2 degrees of freedom (n_players - 1) while sitting on top
    /// of 87 scores. A single outlier player dominates the average completely, which is
    /// exactly what happened — uid 10107 alone swung the headline number.
    ///
    /// The fix pools all scores while still absorbing between-player ability, by
    /// "demeaning" each score against its own player's mean before pooling: for score i
    /// belonging to player p, x_i = collision_share_i - mean_collision_share_p and
    /// y_i = skill_i - mean_skill_p. Averaging out to zero within each player is exactly
    /// what a player fixed effect (an intercept per player) does in a regression — this
    /// is the "within" or fixed-effects estimator, computed by hand instead of via a
    /// matrix library because with one regressor it reduces to an OLS-through-the-origin
    /// on the demeaned pool: slope = sum(x_i * y_i) / sum(x_i^2). It uses up n_players
    /// degrees of freedom for the intercepts (one mean subtracted per player) plus 1 for
    /// the slope itself, leaving n - n_players - 1 residual degrees of freedom — about
    /// 83 here instead of the 2 the per-player average was implicitly resting on, for
    /// the same 87 scores. The per-player table is kept below since it is still useful
    /// to see the raw shape per player; the pooled estimate is the headline because it
    /// is the one with enough power to say anything.
    ///
    /// Also prints the identical pooled fixed-effects estimate against `attrs.stars` as
    /// a control: if skill trends with star rating within a player too, the collision
    /// slope may just be picking up difficulty misestimation in general rather than
    /// collision specifically.
    ///
    /// Running those two univariate regressions side by side is not enough to settle
    /// that, though: collision share and star rating both track map style (denser,
    /// jack-heavy charts tend to run both higher collision and higher stars), so
    /// whichever trend stars is really carrying will partly load onto the collision
    /// coefficient when the two are fit separately, and vice versa. The fix is a
    /// two-variable OLS on the same within-player-demeaned pool — x1 = collision share,
    /// x2 = stars, y = skill, all demeaned against their own player's mean — solved by
    /// hand via the 2x2 normal equations rather than a matrix library, the direct
    /// generalisation of the univariate case's OLS-through-the-origin. It is printed
    /// alongside the univariate numbers, for both (a) all scores and (b) the
    /// no-window-mod subset, so the shift from univariate to joint is visible rather
    /// than replacing the old numbers outright. The correlation between the two
    /// demeaned regressors is printed with it, because that correlation is the real
    /// diagnostic: if it is high, the joint fit cannot actually separate the two
    /// effects, and both coefficients should be read as unstable rather than trusted at
    /// face value just because the arithmetic produced a number.
    ///
    /// The stars trend is also worth characterising on its own, separately from
    /// collision entirely. `sigma = sigma_ref * ((d + difficulty_floor) / skill)^
    /// skill_exponent` means a mis-set `skill_exponent` will make fitted skill drift
    /// with local difficulty *by construction*, for reasons that have nothing to do
    /// with collisions — so a stars trend is exactly the symptom a wrong exponent would
    /// produce. Below, the pooled skill-vs-stars relationship is broken out per player
    /// and star-rating bin to see whether it is monotone or driven by one bin, and then
    /// the no-window-mod joint regression is re-run under [`ErrorModel::default`] with
    /// `skill_exponent` swept over 1.3-2.1 around the shipped 1.7, to see whether some
    /// other exponent would flatten the stars coefficient toward zero on this fixture
    /// set.
    ///
    /// A second confound, caught only after the pooled estimate above was written: uid
    /// 10107 (documented at the `REAL_SCORES` fixture above as an "EZ pp exploiter") runs
    /// most of their scores under `EZ`, which multiplies hit windows — and therefore the
    /// absolute gap needed to avoid a collision — by 1.4x. [`gap_shape_for`] always
    /// computes collision share against the map's *own*, no-mod windows, so an EZ score's
    /// true collision exposure is understated on the x-axis: the same chart is easier to
    /// avoid colliding on than its no-mod collision share suggests, for a player who
    /// abnormally favours it. That taints any slope pooled across mod states, so the
    /// pooled estimate below is printed three times: all scores, no-window-mod scores
    /// (excluding `EZ` and `HR`, both of which rescale windows), and window-mod scores
    /// only. This deliberately does not try to rescale the modded collision share to
    /// compensate — that rescaling needs its own care (whether it is windows-only or also
    /// changes hold/gap geometry) and is future work, not this report's job.
    ///
    /// Only players with at least 4 scores and a collision-share range of at least
    /// 0.15 are reported in the per-player table — a player whose maps all sit at
    /// similar collision share cannot inform a slope, and would only add noise. The
    /// pooled estimate does not apply that filter: it uses every player with at least 2
    /// scores, since the fixed-effects demeaning itself down-weights players with little
    /// internal spread (their demeaned x_i cluster near zero and contribute little to
    /// sum(x_i^2)).
    ///
    /// Run with `cargo test --release collision_skill_slope -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads gitignored fixtures; prints a report rather than asserting"]
    fn collision_skill_slope() {
        use std::collections::{BTreeMap, HashMap};

        let scores = load_multiuser();
        if scores.is_empty() {
            println!("no fixtures present (local-fixtures/multiuser.tsv); nothing to report");
            return;
        }

        struct Point {
            uid: String,
            collision_share: f64,
            stars: f64,
            skill: f64,
            /// Whether this score used a mod that rescales hit windows (`EZ` or `HR`),
            /// which makes the no-mod `collision_share` an understatement (`EZ`) or
            /// overstatement (`HR`) of the score's true collision exposure.
            window_mod: bool,
            /// The map id and raw judgement counts, kept only so the `skill_exponent`
            /// sweep below can refit this score's skill under a non-default
            /// `ErrorModel` without re-reading `local-fixtures/multiuser.tsv`.
            map_id: String,
            mods: String,
            counts: [u32; 6],
        }

        let mut cache: HashMap<String, Option<GapShape>> = HashMap::new();
        let mut points = Vec::new();

        for s in &scores {
            if s.row.live_pp <= 0.0 || s.skill <= 0.0 {
                continue;
            }

            let shape = cache
                .entry(s.row.map_id.clone())
                .or_insert_with(|| gap_shape_for(&s.row.map_id));

            let Some(shape) = shape else {
                continue;
            };

            points.push(Point {
                uid: s.row.uid.clone(),
                collision_share: shape.collision_share,
                stars: s.stars,
                skill: s.skill,
                window_mod: s.row.mods.contains("EZ") || s.row.mods.contains("HR"),
                map_id: s.row.map_id.clone(),
                mods: s.row.mods.clone(),
                counts: s.row.counts,
            });
        }

        // Refit a point's skill from scratch under `model` instead of the default used
        // by `load_multiuser`. Re-parses the map and recomputes mods/attrs rather than
        // reusing anything cached on `MultiPriced`, since that struct only ever holds
        // the default-model fit. Mirrors `load_multiuser`'s own calculate -> units ->
        // fit_with_quality pipeline exactly, so the only thing that changes is `model`.
        fn refit_skill(point: &Point, model: &ErrorModel) -> Option<f64> {
            let map = parse(&format!("local-fixtures/maps/{}.osu", point.map_id))?;
            let (mods, clock_rate) = mods_for(&point.mods);
            let attrs = calculate(&map, &mods, clock_rate, Some(false), None)?;
            let total = point.counts.iter().sum::<u32>();
            let units = judgement_units(&attrs, f64::from(total), model);
            let fit = fit_with_quality(&point.counts, &units, &attrs.hit_windows, model);
            (fit.skill > 0.0).then_some(fit.skill)
        }

        if points.is_empty() {
            println!(
                "no scores with both a fitted skill and a fitted gap shape; nothing to report"
            );
            return;
        }

        // Pooled within-player (fixed-effects) OLS slope of `get_y` on `get_x`, computed
        // by demeaning each point against its own player's mean and running a single
        // OLS-through-the-origin over the pooled, demeaned points. Returns the slope, its
        // standard error, the t-statistic, n, and n_players. See the doc comment above
        // for why this beats averaging per-player slopes.
        fn pooled_fixed_effects(
            by_uid: &BTreeMap<&str, Vec<&Point>>,
            get_x: impl Fn(&Point) -> f64,
            get_y: impl Fn(&Point) -> f64,
        ) -> Option<(f64, f64, f64, usize, usize)> {
            let mut demeaned = Vec::new();
            let mut n_players = 0;

            for rows in by_uid.values() {
                if rows.len() < 2 {
                    continue;
                }
                n_players += 1;

                let n = rows.len() as f64;
                let mean_x = rows.iter().map(|p| get_x(p)).sum::<f64>() / n;
                let mean_y = rows.iter().map(|p| get_y(p)).sum::<f64>() / n;

                for p in rows {
                    demeaned.push((get_x(p) - mean_x, get_y(p) - mean_y));
                }
            }

            let n = demeaned.len();
            let sum_xx: f64 = demeaned.iter().map(|(x, _)| x * x).sum();
            if n == 0 || sum_xx <= 1e-9 {
                return None;
            }

            let sum_xy: f64 = demeaned.iter().map(|(x, y)| x * y).sum();
            let slope = sum_xy / sum_xx;

            let residual_df = n as isize - n_players as isize - 1;
            if residual_df <= 0 {
                return None;
            }

            let ss_res: f64 = demeaned
                .iter()
                .map(|(x, y)| (y - slope * x).powi(2))
                .sum();
            let s2 = ss_res / residual_df as f64;
            let se = (s2 / sum_xx).sqrt();
            let t = if se > 1e-12 { slope / se } else { f64::INFINITY };

            Some((slope, se, t, n, n_players))
        }

        struct JointFit {
            b1: f64,
            se1: f64,
            t1: f64,
            b2: f64,
            se2: f64,
            t2: f64,
            /// Correlation between the demeaned regressors, `S12 / sqrt(S11 * S22)`.
            /// The diagnostic that matters most: if this is large, `b1` and `b2`
            /// cannot be trusted individually no matter how big their `t` looks,
            /// because the two regressors barely vary independently once the
            /// player mean is taken out.
            corr: f64,
            n: usize,
            n_players: usize,
        }

        // Two-variable within-player (fixed-effects) OLS of `y` on `x1` and `x2`
        // jointly, by demeaning each of the three series against its own player's mean
        // and solving the pooled 2x2 normal equations by hand (see the doc comment
        // above for why this is needed rather than the two univariate fits above).
        // Takes `(x1, x2, y)` triples per player directly rather than `&Point`, so the
        // `skill_exponent` sweep below can reuse it with refit skills as `y` without
        // constructing throwaway `Point`s.
        fn pooled_joint(by_uid: &BTreeMap<&str, Vec<(f64, f64, f64)>>) -> Option<JointFit> {
            let mut demeaned: Vec<(f64, f64, f64)> = Vec::new();
            let mut n_players = 0;

            for rows in by_uid.values() {
                if rows.len() < 2 {
                    continue;
                }
                n_players += 1;

                let n = rows.len() as f64;
                let mean_x1 = rows.iter().map(|(x1, _, _)| x1).sum::<f64>() / n;
                let mean_x2 = rows.iter().map(|(_, x2, _)| x2).sum::<f64>() / n;
                let mean_y = rows.iter().map(|(_, _, y)| y).sum::<f64>() / n;

                for (x1, x2, y) in rows {
                    demeaned.push((x1 - mean_x1, x2 - mean_x2, y - mean_y));
                }
            }

            let n = demeaned.len();
            let s11: f64 = demeaned.iter().map(|(x1, _, _)| x1 * x1).sum();
            let s22: f64 = demeaned.iter().map(|(_, x2, _)| x2 * x2).sum();
            let s12: f64 = demeaned.iter().map(|(x1, x2, _)| x1 * x2).sum();
            let s1y: f64 = demeaned.iter().map(|(x1, _, y)| x1 * y).sum();
            let s2y: f64 = demeaned.iter().map(|(_, x2, y)| x2 * y).sum();

            let det = s11 * s22 - s12 * s12;
            if n == 0 || s11 <= 1e-9 || s22 <= 1e-9 || det.abs() <= 1e-9 {
                return None;
            }

            let b1 = (s22 * s1y - s12 * s2y) / det;
            let b2 = (s11 * s2y - s12 * s1y) / det;

            let residual_df = n as isize - n_players as isize - 2;
            if residual_df <= 0 {
                return None;
            }

            let ss_res: f64 = demeaned
                .iter()
                .map(|(x1, x2, y)| (y - b1 * x1 - b2 * x2).powi(2))
                .sum();
            let s2 = ss_res / residual_df as f64;
            let se1 = (s2 * s22 / det).sqrt();
            let se2 = (s2 * s11 / det).sqrt();
            let t1 = if se1 > 1e-12 { b1 / se1 } else { f64::INFINITY };
            let t2 = if se2 > 1e-12 { b2 / se2 } else { f64::INFINITY };
            let corr = s12 / (s11 * s22).sqrt();

            Some(JointFit {
                b1,
                se1,
                t1,
                b2,
                se2,
                t2,
                corr,
                n,
                n_players,
            })
        }

        let report_pooled = |label: &str, subset: &[&Point]| {
            if subset.is_empty() {
                println!("  {label}: n=0, nothing to report");
                return;
            }

            let mut grouped: BTreeMap<&str, Vec<&Point>> = BTreeMap::new();
            for p in subset {
                grouped.entry(p.uid.as_str()).or_default().push(p);
            }

            let mean_skill = subset.iter().map(|p| p.skill).sum::<f64>() / subset.len() as f64;

            println!("  {label}:");
            match pooled_fixed_effects(&grouped, |p| p.collision_share, |p| p.skill) {
                Some((slope, se, t, n, n_players)) => println!(
                    "    collision share: n={n:<4} n_players={n_players}  slope={slope:+.3}  \
                     ({:+.1}% per +100pp collision)  se={se:.3}  t={t:+.2}",
                    100.0 * slope / mean_skill
                ),
                None => println!("    collision share: not enough within-player spread"),
            }
            match pooled_fixed_effects(&grouped, |p| p.stars, |p| p.skill) {
                Some((slope, se, t, n, n_players)) => println!(
                    "    stars (control): n={n:<4} n_players={n_players}  slope={slope:+.3}  \
                     ({:+.1}% per +1 star)  se={se:.3}  t={t:+.2}",
                    100.0 * slope / mean_skill
                ),
                None => println!("    stars (control): not enough within-player spread"),
            }
            let joint_grouped: BTreeMap<&str, Vec<(f64, f64, f64)>> = grouped
                .iter()
                .map(|(&uid, rows)| {
                    let triples = rows
                        .iter()
                        .map(|p| (p.collision_share, p.stars, p.skill))
                        .collect();
                    (uid, triples)
                })
                .collect();
            match pooled_joint(&joint_grouped) {
                Some(fit) => {
                    println!(
                        "    joint (collision + stars): n={:<4} n_players={}  \
                         demeaned corr(collision, stars)={:+.3}",
                        fit.n, fit.n_players, fit.corr
                    );
                    println!(
                        "      collision: b1={:+.3}  ({:+.1}% per +100pp collision)  \
                         se={:.3}  t={:+.2}",
                        fit.b1,
                        100.0 * fit.b1 / mean_skill,
                        fit.se1,
                        fit.t1
                    );
                    println!(
                        "      stars:     b2={:+.3}  ({:+.1}% per +1 star)  se={:.3}  t={:+.2}",
                        fit.b2,
                        100.0 * fit.b2 / mean_skill,
                        fit.se2,
                        fit.t2
                    );
                    if fit.corr.abs() >= 0.7 {
                        println!(
                            "      warning: |corr| >= 0.7 — collision share and stars are too \
                             entangled in this subset for the joint estimate to separate them; \
                             read b1 and b2 as unstable, not as settled effects."
                        );
                    }
                }
                None => println!(
                    "    joint (collision + stars): skipped — det near zero or not enough \
                     within-player spread to solve the normal equations"
                ),
            }
        };

        println!(
            "{} scores with a fitted skill and a fitted gap shape, across {} players.",
            points.len(),
            points
                .iter()
                .map(|p| p.uid.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        );

        println!(
            "\nPooled within-player (fixed-effects) slope of fitted skill against \
             collision share, plus the identical estimate against `attrs.stars` as a \
             control. This is the headline: it pools all qualifying players' demeaned \
             scores into one regression instead of averaging three separate slopes."
        );

        let all: Vec<&Point> = points.iter().collect();
        let no_window_mod: Vec<&Point> = points.iter().filter(|p| !p.window_mod).collect();
        let window_mod: Vec<&Point> = points.iter().filter(|p| p.window_mod).collect();

        println!();
        report_pooled("(a) all scores", &all);
        println!();
        report_pooled("(b) no-window-mod scores (excludes EZ, HR)", &no_window_mod);
        println!();
        report_pooled("(c) window-mod scores only (EZ or HR)", &window_mod);

        // Least-squares slope of `ys` on `xs`. Shared by the collision regression and
        // the stars control so the two are computed identically.
        fn slope(pairs: &[(f64, f64)]) -> Option<f64> {
            let n = pairs.len() as f64;
            let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
            let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
            let covariance: f64 = pairs
                .iter()
                .map(|(x, y)| (x - mean_x) * (y - mean_y))
                .sum();
            let variance: f64 = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
            (variance > 1e-9).then_some(covariance / variance)
        }

        let mut by_uid: BTreeMap<&str, Vec<&Point>> = BTreeMap::new();
        for point in &points {
            by_uid.entry(point.uid.as_str()).or_default().push(point);
        }

        println!(
            "\nPer-player context (not the headline — see the pooled estimate above): each \
             player's own share of scores using a window mod, and their individual slope of \
             fitted skill against collision share, with fitted skill against `attrs.stars` \
             printed alongside as a control. Only players with >= 4 scores and a \
             collision-share range >= 0.15 are shown; the rest cannot inform a slope."
        );

        let mut excluded_n = 0;
        let mut excluded_range = 0;
        let mut collision_slopes = Vec::new();

        for (uid, rows) in &by_uid {
            if rows.len() < 4 {
                excluded_n += 1;
                continue;
            }

            let lo = rows
                .iter()
                .map(|p| p.collision_share)
                .fold(f64::INFINITY, f64::min);
            let hi = rows
                .iter()
                .map(|p| p.collision_share)
                .fold(f64::NEG_INFINITY, f64::max);
            let range = hi - lo;

            if range < 0.15 {
                excluded_range += 1;
                continue;
            }

            let mean_skill = rows.iter().map(|p| p.skill).sum::<f64>() / rows.len() as f64;
            let window_mod_share =
                rows.iter().filter(|p| p.window_mod).count() as f64 / rows.len() as f64;

            let collision_pairs: Vec<(f64, f64)> =
                rows.iter().map(|p| (p.collision_share, p.skill)).collect();
            let stars_pairs: Vec<(f64, f64)> = rows.iter().map(|p| (p.stars, p.skill)).collect();

            let Some(collision_slope) = slope(&collision_pairs) else {
                continue;
            };
            let stars_slope = slope(&stars_pairs);

            let collision_pct = 100.0 * collision_slope / mean_skill;
            collision_slopes.push(collision_pct);

            println!(
                "\n=== uid {uid} (n={}, collision share {lo:.2}-{hi:.2}, mean skill \
                 {mean_skill:.2}, {:.0}% EZ/HR)",
                rows.len(),
                window_mod_share * 100.0
            );
            println!(
                "  d(skill)/d(collision share) = {collision_slope:+.3}  \
                 ({collision_pct:+.1}% per +100pp collision share)"
            );
            match stars_slope {
                Some(stars_slope) => println!(
                    "  d(skill)/d(stars)            = {stars_slope:+.3}  \
                     ({:+.1}% per +1 star, control)",
                    100.0 * stars_slope / mean_skill
                ),
                None => println!("  d(skill)/d(stars)            = n/a (no star-rating spread)"),
            }
        }

        println!(
            "\n{} players excluded for fewer than 4 scores, {} for a collision-share range \
             under 0.15.",
            excluded_n, excluded_range
        );

        if collision_slopes.is_empty() {
            println!("\nno player had enough spread to compute a slope; nothing to summarise.");
            return;
        }

        let mut sorted = collision_slopes.clone();
        sorted.sort_by(f64::total_cmp);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let negative = sorted.iter().filter(|&&s| s < 0.0).count();
        let positive = sorted.iter().filter(|&&s| s > 0.0).count();

        println!(
            "\nOverall, {} qualifying players: mean collision slope {mean:+.1}% per +100pp, \
             median {median:+.1}% per +100pp, {negative} negative vs {positive} positive.",
            sorted.len()
        );

        // Characterising the stars trend on its own, as promised in the doc comment:
        // a per-player, per-bin table first, to see whether it is monotone or driven
        // by one bin, then a `skill_exponent` sweep to see whether the default 1.7 is
        // what is producing it.
        println!(
            "\nFitted skill by star-rating bin, per player (all scores, no mod filter). \
             A monotone climb across bins within a player is what a wrong \
             `skill_exponent` predicts; a single outlier bin would point elsewhere."
        );

        let star_bins: [(&str, f64, f64); 5] = [
            ("<5", f64::NEG_INFINITY, 5.0),
            ("5-6", 5.0, 6.0),
            ("6-7", 6.0, 7.0),
            ("7-8", 7.0, 8.0),
            (">=8", 8.0, f64::INFINITY),
        ];

        for (uid, rows) in &by_uid {
            print!("  uid {uid:<8}");
            for (label, lo, hi) in &star_bins {
                let group: Vec<&&Point> = rows
                    .iter()
                    .filter(|p| p.stars >= *lo && p.stars < *hi)
                    .collect();
                if group.is_empty() {
                    print!("  {label:>4}: n=0          ");
                } else {
                    let mean_skill =
                        group.iter().map(|p| p.skill).sum::<f64>() / group.len() as f64;
                    print!(
                        "  {label:>4}: n={:<3} skill={mean_skill:6.2}",
                        group.len()
                    );
                }
            }
            println!();
        }

        // `skill_exponent` sweep on the no-window-mod subset: refit every score's
        // skill under each candidate exponent (all other `ErrorModel` fields left at
        // their default) and re-run the joint regression, watching only the stars
        // coefficient. If some exponent other than the shipped 1.7 drives it toward
        // zero, that is this fixture set's evidence about the right value; if none do,
        // or the sweep is flat, that is itself the finding.
        println!(
            "\nskill_exponent sweep (no-window-mod scores, joint regression, stars \
             coefficient only):"
        );

        let mut best_exponent = None;
        let mut best_abs_t = f64::INFINITY;

        for exponent in [1.3, 1.5, 1.7, 1.9, 2.1] {
            let model = ErrorModel {
                skill_exponent: exponent,
                ..Default::default()
            };

            let mut refit_by_uid: BTreeMap<&str, Vec<(f64, f64, f64)>> = BTreeMap::new();
            let mut n_failed = 0;

            for p in &no_window_mod {
                match refit_skill(p, &model) {
                    Some(skill) => refit_by_uid
                        .entry(p.uid.as_str())
                        .or_default()
                        .push((p.collision_share, p.stars, skill)),
                    None => n_failed += 1,
                }
            }

            match pooled_joint(&refit_by_uid) {
                Some(fit) => {
                    println!(
                        "  skill_exponent={exponent:.1}: n={:<4} n_players={}  \
                         b2(stars)={:+.4}  se={:.4}  t={:+.2}{}",
                        fit.n,
                        fit.n_players,
                        fit.b2,
                        fit.se2,
                        fit.t2,
                        if n_failed > 0 {
                            format!("  ({n_failed} refits failed)")
                        } else {
                            String::new()
                        }
                    );
                    if fit.t2.abs() < best_abs_t {
                        best_abs_t = fit.t2.abs();
                        best_exponent = Some(exponent);
                    }
                }
                None => println!(
                    "  skill_exponent={exponent:.1}: joint regression skipped (det near zero \
                     or not enough spread)"
                ),
            }
        }

        match best_exponent {
            Some(exponent) => println!(
                "\nOf {{1.3, 1.5, 1.7, 1.9, 2.1}}, skill_exponent={exponent:.1} drives the stars \
                 coefficient closest to zero on this fixture set (|t|={best_abs_t:.2})."
            ),
            None => println!(
                "\nno exponent in the sweep produced a joint fit; nothing to conclude about \
                 skill_exponent from this fixture set."
            ),
        }
    }
}
