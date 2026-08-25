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

use crate::mania_accuracy::{fit_with_quality, ErrorModel, JudgementUnit};
use crate::mania_windows::{hit_windows, ManiaHitWindows};

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
        max_combo,
        n_objects: data.notes.len(),
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

/// The reference window set that [`window_scalar`] is measured against: OD 8
/// classic non-convert, the modal mania OD. The choice only fixes where the scalar
/// equals 1, not the size of its response.
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

/// How much the windows a score was played under change what it is worth.
///
/// This is where mods get priced, and it is the whole point of widening the windows
/// *before* grading the score. The same judgement counts are fitted twice: once
/// against the windows actually in effect, once against [`REFERENCE_WINDOWS`]. A
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

    // Uniform local difficulty for now: every note carries the map's star rating.
    // Per-note difficulty replaces this, and is why the response is currently the
    // same on every map.
    let units = [JudgementUnit::repeated(attrs.stars, f64::from(total))];
    let model = ErrorModel::default();

    let played = fit_with_quality(&counts, &units, &attrs.hit_windows, &model);
    let reference = fit_with_quality(&counts, &units, &REFERENCE_WINDOWS, &model);

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
    let sv2 = has_mod(mods, "V2");
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

    /// A score played on exactly the reference windows must be priced at 1: the two
    /// fits are then the same fit. Guards the scalar against picking up an offset
    /// from anything other than the windows.
    #[test]
    fn a_reference_od_no_mod_score_is_priced_at_one() {
        let map = synthetic_map(8.0, 2000, 120.0);
        let mods = GameMods::default();
        let attrs = calculate(&map, &mods, 1.0, Some(true), None).unwrap();

        let state = SunnyScoreState {
            n320: 1400,
            n300: 480,
            n200: 90,
            n100: 20,
            n50: 5,
            misses: 5,
        };

        let perf = calculate_performance(&attrs, &mods, state);

        assert!(
            (perf.window_scalar - 1.0).abs() < 1e-6,
            "reference windows must be neutral, got {}",
            perf.window_scalar
        );
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
}
