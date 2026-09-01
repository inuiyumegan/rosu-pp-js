//! WASM bindings for the sunny (Star-Rating-Rebirth) mania algorithm.

use std::borrow::Cow;

use rosu_mods::GameMods;
use rosu_pp::model::{beatmap::Beatmap, mode::GameMode};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{
    JsError, JsResult,
    args::difficulty::{DifficultyArgs, JsDifficultyArgs},
    args::performance::{JsPerformanceArgs, PerformanceArgs},
    beatmap::JsBeatmap,
    sunny::{
        self, INPUT_STATE_BINS, InputClass, InputStateBin, NOTE_DIFFICULTY_BINS, NoteDifficultyBin,
        SunnyManiaDifficultyAttributes, SunnyScoreState,
    },
    util,
};

// ---------------------------------------------------------------------------
// Difficulty attributes
// ---------------------------------------------------------------------------

/// The result of a sunny mania difficulty calculation.
#[wasm_bindgen(js_name = SunnyManiaDifficultyAttributes, inspectable)]
#[derive(Clone, Default, serde::Deserialize)]
#[serde(rename = "SunnyManiaDifficultyAttributes", rename_all = "camelCase")]
pub struct JsSunnyManiaDifficultyAttributes {
    /// The final star rating.
    #[wasm_bindgen(readonly)]
    pub stars: f64,
    /// The variety measure of the map.
    #[wasm_bindgen(readonly)]
    pub variety: f64,
    /// The accuracy scalar `0.5 * spikiness + 0.5 * switches`.
    #[wasm_bindgen(js_name = "accScalar", readonly)]
    pub acc_scalar: f64,
    /// How much the difficulty spikes within the map.
    #[wasm_bindgen(readonly)]
    pub spikiness: f64,
    /// How much the playstyle switches between jack and stream-like patterns.
    #[wasm_bindgen(readonly)]
    pub switches: f64,
    /// The GREAT hit window used for the calculation (incl. mods).
    #[wasm_bindgen(js_name = "greatHitWindow", readonly)]
    pub great_hit_window: f64,
    /// The max combo of the map.
    #[wasm_bindgen(js_name = "maxCombo", readonly)]
    pub max_combo: u32,
    /// The amount of hit objects taken into account.
    #[wasm_bindgen(js_name = "nObjects", readonly)]
    pub n_objects: u32,
    /// How many of those hit objects are long notes.
    ///
    /// Exposed because the judgement model needs it: under ScoreV1 a long note is
    /// graded on the summed head and release offsets, so it carries more timing
    /// spread than a plain note and an LN-heavy map is a mixture of the two.
    #[wasm_bindgen(js_name = "nLongNotes", readonly)]
    pub n_long_notes: u32,
    /// Versioned flattened input-state bins retained by cached JS attributes.
    #[serde(default)]
    pub(crate) input_state_bins: Vec<f64>,
    /// Exact played judgement windows retained by cached JS attributes.
    #[serde(default, rename = "hitWindows")]
    pub(crate) serialized_hit_windows: Vec<f64>,
    /// Exact natural judgement windows retained by cached JS attributes.
    #[serde(default, rename = "mapWindows")]
    pub(crate) serialized_map_windows: Vec<f64>,
    /// Exact long-note duration histogram retained by cached JS attributes.
    #[serde(default, rename = "lnDurationBuckets")]
    pub(crate) serialized_ln_duration_buckets: Vec<f64>,
    /// Exact per-note difficulty bins retained by cached JS attributes.
    #[serde(default, rename = "noteDifficultyBins")]
    pub(crate) serialized_note_difficulty_bins: Vec<f64>,
    /// Explicit scoring mode retained by cached JS attributes.
    #[serde(default)]
    pub(crate) ln_judged_as_one: Option<bool>,
    /// The long-note duration histogram, kept for the performance calc.
    ///
    /// The exact Rust-side copy. JS receives the flattened
    /// `serialized_ln_duration_buckets` payload because `wasm_bindgen` cannot carry a
    /// fixed-size array as a field.
    #[serde(skip)]
    pub(crate) ln_duration_buckets: [usize; crate::mania_accuracy::LN_DURATION_BUCKETS],
    /// The mods used for the calculation, kept for the performance calc.
    #[serde(skip)]
    pub(crate) mods: rosu_mods::GameMods,
    /// The judgement windows the score will be graded against, kept for the
    /// performance calc. JS receives the flattened `serialized_hit_windows` payload.
    #[serde(skip)]
    pub(crate) hit_windows: crate::mania_windows::ManiaHitWindows,
    /// The map's windows with mods stripped, kept for the performance calc. JS receives
    /// the flattened `serialized_map_windows` payload.
    #[serde(skip)]
    pub(crate) map_windows: crate::mania_windows::ManiaHitWindows,
}

impl From<SunnyManiaDifficultyAttributes> for JsSunnyManiaDifficultyAttributes {
    fn from(attrs: SunnyManiaDifficultyAttributes) -> Self {
        Self {
            stars: attrs.stars,
            variety: attrs.variety,
            acc_scalar: attrs.acc_scalar,
            spikiness: attrs.spikiness,
            switches: attrs.switches,
            great_hit_window: attrs.great_hit_window,
            max_combo: attrs.max_combo,
            n_objects: attrs.n_objects as u32,
            n_long_notes: attrs.n_long_notes as u32,
            input_state_bins: encode_input_state_bins(attrs.input_state_bins.as_ref()),
            serialized_hit_windows: encode_windows(attrs.hit_windows),
            serialized_map_windows: encode_windows(attrs.map_windows),
            serialized_ln_duration_buckets: encode_ln_duration_buckets(attrs.ln_duration_buckets),
            serialized_note_difficulty_bins: encode_note_difficulty_bins(
                attrs.note_difficulty_bins.as_ref(),
            ),
            ln_judged_as_one: Some(attrs.ln_judged_as_one),
            ln_duration_buckets: attrs.ln_duration_buckets,
            mods: GameMods::default(),
            hit_windows: attrs.hit_windows,
            map_windows: attrs.map_windows,
        }
    }
}

#[wasm_bindgen(js_class = SunnyManiaDifficultyAttributes)]
impl JsSunnyManiaDifficultyAttributes {
    /// Compact input-state metadata used by the timing surface.
    #[wasm_bindgen(getter = inputStateBins)]
    pub fn input_state_bins(&self) -> Box<[f64]> {
        self.input_state_bins.clone().into_boxed_slice()
    }

    #[wasm_bindgen(getter = hitWindows)]
    pub fn hit_windows(&self) -> Box<[f64]> {
        self.serialized_hit_windows.clone().into_boxed_slice()
    }

    #[wasm_bindgen(getter = mapWindows)]
    pub fn map_windows(&self) -> Box<[f64]> {
        self.serialized_map_windows.clone().into_boxed_slice()
    }

    #[wasm_bindgen(getter = lnDurationBuckets)]
    pub fn ln_duration_buckets(&self) -> Box<[f64]> {
        self.serialized_ln_duration_buckets
            .clone()
            .into_boxed_slice()
    }

    #[wasm_bindgen(getter = noteDifficultyBins)]
    pub fn note_difficulty_bins(&self) -> Box<[f64]> {
        self.serialized_note_difficulty_bins
            .clone()
            .into_boxed_slice()
    }

    #[wasm_bindgen(getter = lnJudgedAsOne)]
    pub fn ln_judged_as_one(&self) -> Option<bool> {
        self.ln_judged_as_one
    }
}

fn encode_windows(windows: crate::mania_windows::ManiaHitWindows) -> Vec<f64> {
    vec![
        windows.perfect,
        windows.great,
        windows.good,
        windows.ok,
        windows.meh,
        windows.miss,
    ]
}

fn decode_windows(encoded: &[f64]) -> Option<crate::mania_windows::ManiaHitWindows> {
    (encoded.len() == 6
        && encoded
            .iter()
            .all(|value| value.is_finite() && *value > 0.0))
    .then(|| crate::mania_windows::ManiaHitWindows {
        perfect: encoded[0],
        great: encoded[1],
        good: encoded[2],
        ok: encoded[3],
        meh: encoded[4],
        miss: encoded[5],
    })
}

fn encode_ln_duration_buckets(
    buckets: [usize; crate::mania_accuracy::LN_DURATION_BUCKETS],
) -> Vec<f64> {
    buckets.into_iter().map(|count| count as f64).collect()
}

fn decode_ln_duration_buckets(
    encoded: &[f64],
) -> Option<[usize; crate::mania_accuracy::LN_DURATION_BUCKETS]> {
    if encoded.len() != crate::mania_accuracy::LN_DURATION_BUCKETS
        || encoded
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || value.fract() != 0.0)
    {
        return None;
    }

    Some(std::array::from_fn(|idx| encoded[idx] as usize))
}

const NOTE_DIFFICULTY_FIELDS_PER_BIN: usize = 4;

fn encode_note_difficulty_bins(
    bins: Option<&[NoteDifficultyBin; NOTE_DIFFICULTY_BINS]>,
) -> Vec<f64> {
    let Some(bins) = bins else {
        return Vec::new();
    };

    let mut encoded = Vec::with_capacity(NOTE_DIFFICULTY_BINS * NOTE_DIFFICULTY_FIELDS_PER_BIN);
    for bin in bins {
        encoded.extend([
            bin.difficulty,
            f64::from(bin.rice),
            f64::from(bin.long),
            bin.mean_duration,
        ]);
    }

    encoded
}

fn decode_note_difficulty_bins(
    encoded: &[f64],
) -> Option<[NoteDifficultyBin; NOTE_DIFFICULTY_BINS]> {
    if encoded.len() != NOTE_DIFFICULTY_BINS * NOTE_DIFFICULTY_FIELDS_PER_BIN
        || encoded.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let bins = std::array::from_fn(|idx| {
        let offset = idx * NOTE_DIFFICULTY_FIELDS_PER_BIN;
        NoteDifficultyBin {
            difficulty: encoded[offset],
            rice: encoded[offset + 1].clamp(0.0, f64::from(u32::MAX)) as u32,
            long: encoded[offset + 2].clamp(0.0, f64::from(u32::MAX)) as u32,
            mean_duration: encoded[offset + 3],
        }
    });

    bins.iter()
        .all(|bin| bin.difficulty >= 0.0 && bin.mean_duration >= 0.0)
        .then_some(bins)
}

const INPUT_STATE_SERIAL_VERSION: f64 = 2.0;
const INPUT_STATE_FIELDS_PER_BIN: usize = 10;

fn encode_input_state_bins(bins: Option<&[InputStateBin; INPUT_STATE_BINS]>) -> Vec<f64> {
    let Some(bins) = bins else {
        return Vec::new();
    };

    let mut encoded = Vec::with_capacity(1 + INPUT_STATE_BINS * INPUT_STATE_FIELDS_PER_BIN);
    encoded.push(INPUT_STATE_SERIAL_VERSION);

    for bin in bins {
        encoded.extend([
            f64::from(bin.count),
            f64::from(bin.long_count),
            f64::from(bin.predecessor_count),
            bin.mean_difficulty,
            bin.mean_duration_ms,
            bin.mean_gap_ms,
            bin.mean_next_gap_ms,
            f64::from(bin.next_operation_count),
            bin.mean_chord_width,
            bin.mean_other_held,
        ]);
    }

    encoded
}

fn decode_input_state_bins(encoded: &[f64]) -> Option<[InputStateBin; INPUT_STATE_BINS]> {
    let (fields_per_bin, version) = match encoded.first().copied() {
        Some(1.0) => (8, 1.0),
        Some(INPUT_STATE_SERIAL_VERSION) => {
            (INPUT_STATE_FIELDS_PER_BIN, INPUT_STATE_SERIAL_VERSION)
        }
        _ => return None,
    };
    if encoded.len() != 1 + INPUT_STATE_BINS * fields_per_bin
        || encoded.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let bins = std::array::from_fn(|idx| {
        let offset = 1 + idx * fields_per_bin;
        InputStateBin {
            class: match idx / NOTE_DIFFICULTY_BINS {
                0 => InputClass::FreshPress,
                1 => InputClass::RapidRepress,
                2 => InputClass::Jack,
                3 => InputClass::Release,
                4 => InputClass::ReleaseToPress,
                5 => InputClass::PressUnderHold,
                _ => InputClass::ChordEntryOrExit,
            },
            count: encoded[offset].clamp(0.0, f64::from(u32::MAX)) as u32,
            long_count: encoded[offset + 1].clamp(0.0, f64::from(u32::MAX)) as u32,
            predecessor_count: encoded[offset + 2].clamp(0.0, f64::from(u32::MAX)) as u32,
            mean_difficulty: encoded[offset + 3],
            mean_duration_ms: encoded[offset + 4],
            mean_gap_ms: encoded[offset + 5],
            mean_next_gap_ms: if version >= 2.0 {
                encoded[offset + 6]
            } else {
                0.0
            },
            next_operation_count: if version >= 2.0 {
                encoded[offset + 7].clamp(0.0, f64::from(u32::MAX)) as u32
            } else {
                0
            },
            mean_chord_width: encoded[offset + if version >= 2.0 { 8 } else { 6 }],
            mean_other_held: encoded[offset + if version >= 2.0 { 9 } else { 7 }],
        }
    });

    bins.iter()
        .all(|bin| bin.long_count <= bin.count && bin.predecessor_count <= bin.count)
        .then_some(bins)
}

// ---------------------------------------------------------------------------
// Performance attributes
// ---------------------------------------------------------------------------

/// The result of a sunny mania performance calculation.
#[wasm_bindgen(js_name = SunnyManiaPerformanceAttributes, inspectable)]
#[derive(Clone, Default, serde::Deserialize)]
#[serde(rename = "SunnyManiaPerformanceAttributes", rename_all = "camelCase")]
pub struct JsSunnyManiaPerformanceAttributes {
    /// The total performance points.
    #[wasm_bindgen(readonly)]
    pub pp: f64,
    /// The difficulty portion of the PP.
    #[wasm_bindgen(js_name = "ppDifficulty", readonly)]
    pub pp_difficulty: f64,
    /// The variety multiplier applied to the difficulty portion.
    #[wasm_bindgen(js_name = "varietyMultiplier", readonly)]
    pub variety_multiplier: f64,
    /// The accuracy multiplier applied to the difficulty portion.
    #[wasm_bindgen(js_name = "accMultiplier", readonly)]
    pub acc_multiplier: f64,
    /// The length multiplier applied to the difficulty portion.
    #[wasm_bindgen(js_name = "lengthMultiplier", readonly)]
    pub length_multiplier: f64,
    /// Timing skill inferred from the played hit-result surface, normalized by the
    /// established non-input-state surface through the map's natural windows.
    ///
    /// The played fit uses the judgement windows actually in effect, including EZ/HR.
    #[wasm_bindgen(js_name = "windowScalar", readonly)]
    pub window_scalar: f64,
    /// PP contribution from pattern difficulty (sunny's base calculation).
    #[wasm_bindgen(js_name = "ppPattern", readonly)]
    pub pp_pattern: f64,
    /// PP contribution from timing difficulty (accuracy surface).
    #[wasm_bindgen(js_name = "ppTiming", readonly)]
    pub pp_timing: f64,
    /// Fitted timing skill through actual windows (with mods and input-state).
    #[wasm_bindgen(js_name = "timingSkillPlayed", readonly)]
    pub timing_skill_played: f64,
    /// Fitted timing skill through natural windows with input-state recovery disabled.
    #[wasm_bindgen(js_name = "timingSkillBaseline", readonly)]
    pub timing_skill_baseline: f64,
}

impl From<sunny::SunnyManiaPerformanceAttributes> for JsSunnyManiaPerformanceAttributes {
    fn from(attrs: sunny::SunnyManiaPerformanceAttributes) -> Self {
        Self {
            pp: attrs.pp,
            pp_difficulty: attrs.pp_difficulty,
            variety_multiplier: attrs.variety_multiplier,
            acc_multiplier: attrs.acc_multiplier,
            length_multiplier: attrs.length_multiplier,
            window_scalar: attrs.window_scalar,
            pp_pattern: attrs.pp_pattern,
            pp_timing: attrs.pp_timing,
            timing_skill_played: attrs.timing_skill_played,
            timing_skill_baseline: attrs.timing_skill_baseline,
        }
    }
}

// ---------------------------------------------------------------------------
// Difficulty
// ---------------------------------------------------------------------------

/// Builder for a sunny mania difficulty calculation.
#[wasm_bindgen(js_name = SunnyManiaDifficulty)]
#[derive(Clone)]
pub struct JsSunnyManiaDifficulty {
    pub(crate) args: DifficultyArgs,
}

#[wasm_bindgen(js_class = SunnyManiaDifficulty)]
impl JsSunnyManiaDifficulty {
    /// Create a new sunny mania difficulty calculator.
    #[wasm_bindgen(constructor)]
    pub fn new(args: Option<JsDifficultyArgs>) -> JsResult<Self> {
        let args = args
            .as_deref()
            .map(util::from_value::<DifficultyArgs>)
            .transpose()?
            .unwrap_or_default();

        Ok(Self { args })
    }

    /// Perform the sunny mania difficulty calculation.
    pub fn calculate(&self, map: &JsBeatmap) -> JsResult<JsSunnyManiaDifficultyAttributes> {
        let map = prepare_map(&self.args, map)?;
        let clock_rate = clock_rate(&self.args);

        let attrs = sunny::calculate(
            &map,
            &self.args.mods,
            clock_rate,
            self.args.lazer,
            self.args.passed_objects,
        )
        .ok_or_else(|| JsError::new("sunny calculation requires at least 2 hit objects"))?;

        let mut js_attrs = JsSunnyManiaDifficultyAttributes::from(attrs);
        js_attrs.mods = self.args.mods.clone();

        Ok(js_attrs)
    }

    #[wasm_bindgen(setter)]
    pub fn set_mods(&mut self, mods: Option<crate::mods::JsGameMods>) -> JsResult<()> {
        self.args.mods = mods
            .as_deref()
            .map(crate::deserializer::JsDeserializer::from_ref)
            .map(util::deserialize_mods)
            .transpose()?
            .unwrap_or_default();

        Ok(())
    }

    #[wasm_bindgen(setter)]
    pub fn set_lazer(&mut self, lazer: Option<bool>) {
        self.args.lazer = lazer;
    }

    #[wasm_bindgen(setter = clockRate)]
    pub fn set_clock_rate(&mut self, clock_rate: Option<f64>) {
        self.args.clock_rate = clock_rate;
    }

    #[wasm_bindgen(setter = passedObjects)]
    pub fn set_passed_objects(&mut self, passed_objects: Option<u32>) {
        self.args.passed_objects = passed_objects;
    }
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

/// Builder for a sunny mania performance calculation.
#[wasm_bindgen(js_name = SunnyManiaPerformance)]
pub struct JsSunnyManiaPerformance {
    pub(crate) args: PerformanceArgs,
}

#[wasm_bindgen(js_class = SunnyManiaPerformance)]
impl JsSunnyManiaPerformance {
    /// Create a new sunny mania performance calculator.
    #[wasm_bindgen(constructor)]
    pub fn new(args: Option<JsPerformanceArgs>) -> JsResult<Self> {
        let args = args
            .as_deref()
            .map(util::from_value::<PerformanceArgs>)
            .transpose()?
            .unwrap_or_default();

        Ok(Self { args })
    }

    /// Perform the sunny mania performance calculation.
    ///
    /// The argument must either be the attributes of a previous sunny mania
    /// difficulty calculation or a beatmap.
    pub fn calculate(
        &self,
        value: &wasm_bindgen::JsValue,
    ) -> JsResult<JsSunnyManiaPerformanceAttributes> {
        let (attrs, mods) = self.attrs_and_mods(value)?;
        let state = self.score_state(attrs.n_objects as u32)?;

        let perf_attrs = sunny::calculate_performance(&attrs, &mods, state);

        Ok(perf_attrs.into())
    }

    fn attrs_and_mods(
        &self,
        value: &wasm_bindgen::JsValue,
    ) -> JsResult<(SunnyManiaDifficultyAttributes, rosu_mods::GameMods)> {
        if let Ok(js_attrs) = util::from_value::<JsSunnyManiaDifficultyAttributes>(value) {
            let mods = if !js_attrs.mods.is_empty() {
                js_attrs.mods.clone()
            } else {
                self.args.mods.clone()
            };

            let attrs = reconstruct_attributes(js_attrs, &mods);

            return Ok((attrs, mods));
        }

        if let Ok(map) =
            JsBeatmap::deserialize(crate::deserializer::JsDeserializer::from_ref(value))
        {
            let map = prepare_map_for_perf(&self.args, &map)?;
            let clock_rate = self
                .args
                .clock_rate
                .unwrap_or_else(|| self.args.mods.clock_rate().unwrap_or(1.0));

            let attrs = sunny::calculate(
                &map,
                &self.args.mods,
                clock_rate,
                self.args.lazer,
                self.args.passed_objects,
            )
            .ok_or_else(|| JsError::new("sunny calculation requires at least 2 hit objects"))?;

            return Ok((attrs, self.args.mods.clone()));
        }

        Err(JsError::new(
            "Expected either sunny mania difficulty attributes or a beatmap",
        ))
    }

    fn score_state(&self, n_objects: u32) -> JsResult<SunnyScoreState> {
        let mut state = SunnyScoreState {
            n320: self.args.n_geki.unwrap_or(0),
            n300: self.args.n300.unwrap_or(0),
            n200: self.args.n_katu.unwrap_or(0),
            n100: self.args.n100.unwrap_or(0),
            n50: self.args.n50.unwrap_or(0),
            misses: self.args.misses.unwrap_or(0),
        };

        // If no hitresults were given but an accuracy was, generate the most
        // favorable combination of 320s and 300s that matches the accuracy.
        if state.total_hits() == 0 {
            if let Some(accuracy) = self.args.accuracy {
                let acc = (accuracy / 100.0).clamp(0.0, 1.0);
                let total = n_objects;

                // 305-based weighting: n320 * 305 + n300 * 300 = acc * total * 305
                // assuming only 320s and 300s with n320 + n300 = total.
                let n320 = ((acc * 305.0 * total as f64 - 300.0 * total as f64) / 5.0)
                    .round()
                    .max(0.0)
                    .min(total as f64) as u32;
                let n300 = total - n320;

                state.n320 = n320;
                state.n300 = n300;
            }
        }

        Ok(state)
    }

    #[wasm_bindgen(setter)]
    pub fn set_mods(&mut self, mods: Option<crate::mods::JsGameMods>) -> JsResult<()> {
        self.args.mods = mods
            .as_deref()
            .map(crate::deserializer::JsDeserializer::from_ref)
            .map(util::deserialize_mods)
            .transpose()?
            .unwrap_or_default();

        Ok(())
    }

    #[wasm_bindgen(setter)]
    pub fn set_lazer(&mut self, lazer: Option<bool>) {
        self.args.lazer = lazer;
    }

    #[wasm_bindgen(setter = clockRate)]
    pub fn set_clock_rate(&mut self, clock_rate: Option<f64>) {
        self.args.clock_rate = clock_rate;
    }

    #[wasm_bindgen(setter = passedObjects)]
    pub fn set_passed_objects(&mut self, passed_objects: Option<u32>) {
        self.args.passed_objects = passed_objects;
    }

    #[wasm_bindgen(setter)]
    pub fn set_accuracy(&mut self, accuracy: Option<f64>) {
        self.args.accuracy = accuracy;
    }

    #[wasm_bindgen(setter = nGeki)]
    pub fn set_n_geki(&mut self, n_geki: Option<u32>) {
        self.args.n_geki = n_geki;
    }

    #[wasm_bindgen(setter = nKatu)]
    pub fn set_n_katu(&mut self, n_katu: Option<u32>) {
        self.args.n_katu = n_katu;
    }

    #[wasm_bindgen(setter)]
    pub fn set_n300(&mut self, n300: Option<u32>) {
        self.args.n300 = n300;
    }

    #[wasm_bindgen(setter)]
    pub fn set_n100(&mut self, n100: Option<u32>) {
        self.args.n100 = n100;
    }

    #[wasm_bindgen(setter)]
    pub fn set_n50(&mut self, n50: Option<u32>) {
        self.args.n50 = n50;
    }

    #[wasm_bindgen(setter)]
    pub fn set_misses(&mut self, misses: Option<u32>) {
        self.args.misses = misses;
    }
}

fn reconstruct_attributes(
    js_attrs: JsSunnyManiaDifficultyAttributes,
    mods: &GameMods,
) -> SunnyManiaDifficultyAttributes {
    // New cached attributes carry exact values. The reconstruction branches remain for
    // objects cached by older package versions, which only exposed `greatHitWindow`.
    let hit_windows = decode_windows(&js_attrs.serialized_hit_windows).unwrap_or_else(|| {
        if js_attrs.hit_windows == Default::default() {
            crate::mania_windows::windows_from_great(js_attrs.great_hit_window)
        } else {
            js_attrs.hit_windows
        }
    });
    let map_windows = decode_windows(&js_attrs.serialized_map_windows).unwrap_or_else(|| {
        if js_attrs.map_windows == Default::default() {
            let unmodded =
                js_attrs.great_hit_window * crate::mania_windows::difficulty_multiplier(mods);
            crate::mania_windows::windows_from_great(unmodded)
        } else {
            js_attrs.map_windows
        }
    });
    let ln_duration_buckets = decode_ln_duration_buckets(&js_attrs.serialized_ln_duration_buckets)
        .filter(|buckets| buckets.iter().sum::<usize>() == js_attrs.n_long_notes as usize)
        .or_else(|| {
            (js_attrs.ln_duration_buckets.iter().sum::<usize>() > 0)
                .then_some(js_attrs.ln_duration_buckets)
        })
        .unwrap_or_else(|| sunny::modal_ln_duration_histogram(js_attrs.n_long_notes as usize));

    SunnyManiaDifficultyAttributes {
        stars: js_attrs.stars,
        variety: js_attrs.variety,
        acc_scalar: js_attrs.acc_scalar,
        spikiness: js_attrs.spikiness,
        switches: js_attrs.switches,
        great_hit_window: js_attrs.great_hit_window,
        hit_windows,
        map_windows,
        max_combo: js_attrs.max_combo,
        n_objects: js_attrs.n_objects as usize,
        n_long_notes: js_attrs.n_long_notes as usize,
        ln_duration_buckets,
        note_difficulty_bins: decode_note_difficulty_bins(
            &js_attrs.serialized_note_difficulty_bins,
        ),
        input_state_bins: decode_input_state_bins(&js_attrs.input_state_bins),
        // Missing means an old cached object. Preserve its historical fallback while
        // ensuring every newly produced object carries the difficulty calculation's
        // explicit stable/lazer decision.
        ln_judged_as_one: js_attrs
            .ln_judged_as_one
            .unwrap_or_else(|| sunny::is_classic(None, mods)),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ensure that the beatmap is converted to mania.
fn prepare_map<'m>(args: &DifficultyArgs, map: &'m JsBeatmap) -> JsResult<Cow<'m, Beatmap>> {
    if map.inner.mode == GameMode::Mania {
        return Ok(Cow::Borrowed(&map.inner));
    }

    convert_map(args, &map.inner)
}

/// Like [`prepare_map`] but only used when a beatmap was passed to the
/// performance calculator directly.
fn prepare_map_for_perf<'m>(
    args: &PerformanceArgs,
    map: &'m JsBeatmap,
) -> JsResult<Cow<'m, Beatmap>> {
    if map.inner.mode == GameMode::Mania {
        return Ok(Cow::Borrowed(&map.inner));
    }

    let difficulty_args = DifficultyArgs {
        mods: args.mods.clone(),
        clock_rate: args.clock_rate,
        ar: args.ar,
        ar_with_mods: args.ar_with_mods,
        cs: args.cs,
        cs_with_mods: args.cs_with_mods,
        hp: args.hp,
        hp_with_mods: args.hp_with_mods,
        od: args.od,
        od_with_mods: args.od_with_mods,
        passed_objects: args.passed_objects,
        hardrock_offsets: args.hardrock_offsets,
        lazer: args.lazer,
    };

    convert_map(&difficulty_args, &map.inner)
}

fn convert_map<'m>(args: &DifficultyArgs, map: &'m Beatmap) -> JsResult<Cow<'m, Beatmap>> {
    let mods = rosu_pp::GameMods::from(args.mods.clone());

    map.convert_ref(GameMode::Mania, &mods)
        .map_err(|err| JsError::new(&format!("converting the map to mania failed: {err:?}")))
}

/// The clock rate to use: a custom one if given, otherwise the one from the
/// rate-adjusting mods (defaults to 1.0).
fn clock_rate(args: &DifficultyArgs) -> f64 {
    args.clock_rate
        .unwrap_or_else(|| args.mods.clock_rate().unwrap_or(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosu_mods::{GameMod, GameMods};

    #[test]
    fn input_state_bins_round_trip_and_old_attributes_fall_back() {
        let mut bins = std::array::from_fn(|idx| InputStateBin {
            class: match idx / NOTE_DIFFICULTY_BINS {
                0 => InputClass::FreshPress,
                1 => InputClass::RapidRepress,
                2 => InputClass::Jack,
                3 => InputClass::Release,
                4 => InputClass::ReleaseToPress,
                5 => InputClass::PressUnderHold,
                _ => InputClass::ChordEntryOrExit,
            },
            ..InputStateBin::default()
        });
        bins[2 * NOTE_DIFFICULTY_BINS] = InputStateBin {
            class: InputClass::Jack,
            count: 17,
            long_count: 3,
            predecessor_count: 12,
            mean_difficulty: 8.25,
            mean_duration_ms: 123.0,
            mean_gap_ms: 91.0,
            mean_next_gap_ms: 140.0,
            next_operation_count: 11,
            mean_chord_width: 1.5,
            mean_other_held: 0.25,
        };

        let encoded = encode_input_state_bins(Some(&bins));
        assert_eq!(decode_input_state_bins(&encoded), Some(bins));
        assert_eq!(decode_input_state_bins(&[]), None);

        let mut malformed = encoded;
        malformed[0] = 99.0;
        assert_eq!(decode_input_state_bins(&malformed), None);
    }

    fn ln_heavy_map() -> Beatmap {
        Beatmap::from_bytes(
            br#"osu file format v14

[General]
Mode: 3

[Difficulty]
CircleSize: 4
OverallDifficulty: 7.6
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
0,500,4,2,1,100,1,0

[HitObjects]
64,192,1000,128,0,1180:0:0:0:0:
192,192,1120,128,0,1400:0:0:0:0:
320,192,1260,128,0,1710:0:0:0:0:
448,192,1430,128,0,1510:0:0:0:0:
64,192,1600,128,0,2050:0:0:0:0:
192,192,1760,128,0,1940:0:0:0:0:
320,192,1910,128,0,2190:0:0:0:0:
448,192,2080,128,0,2530:0:0:0:0:
"#,
        )
        .expect("inline mania map must parse")
    }

    fn mods(gamemods: impl IntoIterator<Item = GameMod>) -> GameMods {
        gamemods.into_iter().collect()
    }

    fn assert_close(label: &str, direct: f64, cached: f64) {
        let tolerance = 1e-12 * direct.abs().max(cached.abs()).max(1.0);
        assert!(
            (direct - cached).abs() <= tolerance,
            "{label}: direct={direct:?}, cached={cached:?}"
        );
    }

    #[test]
    fn cached_attributes_preserve_scoring_windows_and_pp() {
        let map = ln_heavy_map();
        let cases = [
            ("v1 nm", GameMods::default(), Some(false), 1.0),
            (
                "v1 ez custom rate",
                mods([GameMod::EasyMania(Default::default())]),
                Some(false),
                1.17,
            ),
            (
                "v1 hr",
                mods([GameMod::HardRockMania(Default::default())]),
                Some(false),
                1.0,
            ),
            (
                "stable sv2 nm",
                mods([GameMod::ScoreV2Mania(Default::default())]),
                Some(false),
                1.0,
            ),
            (
                "stable sv2 ez",
                mods([
                    GameMod::ScoreV2Mania(Default::default()),
                    GameMod::EasyMania(Default::default()),
                ]),
                Some(false),
                1.0,
            ),
            (
                "stable sv2 hr custom rate",
                mods([
                    GameMod::ScoreV2Mania(Default::default()),
                    GameMod::HardRockMania(Default::default()),
                ]),
                Some(false),
                0.91,
            ),
        ];

        for (label, mods, lazer, clock_rate) in cases {
            let direct = sunny::calculate(&map, &mods, clock_rate, lazer, None)
                .unwrap_or_else(|| panic!("{label}: difficulty calculation failed"));
            let mut cached = JsSunnyManiaDifficultyAttributes::from(direct);

            // Model a plain JS object: serde-visible getter payloads survive while the
            // Rust-only fields and attached mods do not.
            cached.hit_windows = Default::default();
            cached.map_windows = Default::default();
            cached.ln_duration_buckets = Default::default();
            cached.mods = Default::default();

            let round_tripped = reconstruct_attributes(cached, &mods);
            assert_eq!(direct.hit_windows, round_tripped.hit_windows, "{label}");
            assert_eq!(direct.map_windows, round_tripped.map_windows, "{label}");
            assert_eq!(
                direct.ln_judged_as_one, round_tripped.ln_judged_as_one,
                "{label}"
            );
            assert_eq!(
                direct.ln_duration_buckets, round_tripped.ln_duration_buckets,
                "{label}"
            );
            assert_eq!(
                direct.note_difficulty_bins, round_tripped.note_difficulty_bins,
                "{label}"
            );
            assert_eq!(
                direct.input_state_bins, round_tripped.input_state_bins,
                "{label}"
            );

            let total = if direct.ln_judged_as_one { 8 } else { 16 };
            let state = SunnyScoreState {
                n320: total - 5,
                n300: 2,
                n200: 1,
                n100: 1,
                n50: 0,
                misses: 1,
            };
            let direct_pp = sunny::calculate_performance(&direct, &mods, state);
            let cached_pp = sunny::calculate_performance(&round_tripped, &mods, state);

            for (field, direct, cached) in [
                (
                    "timing_skill_played",
                    direct_pp.timing_skill_played,
                    cached_pp.timing_skill_played,
                ),
                (
                    "timing_skill_baseline",
                    direct_pp.timing_skill_baseline,
                    cached_pp.timing_skill_baseline,
                ),
                (
                    "window_scalar",
                    direct_pp.window_scalar,
                    cached_pp.window_scalar,
                ),
                ("pp_pattern", direct_pp.pp_pattern, cached_pp.pp_pattern),
                ("pp_timing", direct_pp.pp_timing, cached_pp.pp_timing),
                (
                    "pp_difficulty",
                    direct_pp.pp_difficulty,
                    cached_pp.pp_difficulty,
                ),
                ("pp", direct_pp.pp, cached_pp.pp),
            ] {
                assert_close(&format!("{label} {field}"), direct, cached);
            }
        }
    }
}
