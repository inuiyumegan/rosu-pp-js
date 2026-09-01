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
        self, INPUT_STATE_BINS, InputClass, InputStateBin, NOTE_DIFFICULTY_BINS,
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
    /// The long-note duration histogram, kept for the performance calc.
    ///
    /// Not exposed to JS for the same reason as [`Self::hit_windows`]: it is an
    /// implementation detail of how long notes are priced, and `wasm_bindgen` cannot
    /// carry a fixed-size array as a field anyway. Reconstructed when absent — see the
    /// performance path, which explains what that costs.
    #[serde(skip)]
    pub(crate) ln_duration_buckets: [usize; crate::mania_accuracy::LN_DURATION_BUCKETS],
    /// The mods used for the calculation, kept for the performance calc.
    #[serde(skip)]
    pub(crate) mods: rosu_mods::GameMods,
    /// The judgement windows the score will be graded against, kept for the
    /// performance calc. Not exposed to JS: it is an implementation detail of how
    /// mods are priced, and it is recomputed when absent.
    #[serde(skip)]
    pub(crate) hit_windows: crate::mania_windows::ManiaHitWindows,
    /// The map's windows with mods stripped, kept for the performance calc. Not exposed
    /// for the same reason as [`Self::hit_windows`], and reconstructed the same way.
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
                js_attrs.mods
            } else {
                self.args.mods.clone()
            };

            // Attributes that came back through JS lose the window set, since it is
            // not part of the public shape. Rebuild it from the OD-equivalent of the
            // GREAT window that *is* carried, so a cached-attributes call prices mods
            // the same as a from-beatmap one.
            let hit_windows = if js_attrs.hit_windows == Default::default() {
                crate::mania_windows::windows_from_great(js_attrs.great_hit_window)
            } else {
                js_attrs.hit_windows
            };

            // Likewise the mod-stripped window set, which is what the score is priced
            // *against*, so getting it wrong misprices mods rather than merely blurring
            // them. `great_hit_window` already has the multiplier folded in, and
            // `hit_windows` folds it in by *dividing*, so undo it by multiplying:
            // `EZ`'s 1/1.4 multiplied the played window by 1.4, and multiplying by 1/1.4
            // takes it back. Dividing here instead would widen an already-widened window
            // and hand `EZ` a bonus.
            // Pinned by `stripping_the_mod_multiplier_recovers_the_maps_own_window`.
            let map_windows = if js_attrs.map_windows == Default::default() {
                let unmodded =
                    js_attrs.great_hit_window * crate::mania_windows::difficulty_multiplier(&mods);
                crate::mania_windows::windows_from_great(unmodded)
            } else {
                js_attrs.map_windows
            };

            let attrs = SunnyManiaDifficultyAttributes {
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
                // The histogram survives a Rust-side clone but not a JS round-trip,
                // where it is `serde(skip)` and comes back zeroed. An all-zero
                // histogram on a map that has long notes means "lost", not "no long
                // notes", so fall back to the modal bucket rather than dropping the LN
                // population: the count is the first-order term, and a wrong bucket
                // costs less than pricing a 90% LN map as pure rice. A caller that
                // wants the exact figure should pass the beatmap.
                ln_duration_buckets: if js_attrs.ln_duration_buckets.iter().sum::<usize>() > 0 {
                    js_attrs.ln_duration_buckets
                } else {
                    sunny::modal_ln_duration_histogram(js_attrs.n_long_notes as usize)
                },
                // Lost on a JS round-trip for the same reason as the histogram, and *not*
                // reconstructed: unlike the LN buckets there is no defensible stand-in,
                // since the whole content of this field is how per-note difficulty spreads
                // around `stars` and a round-tripped attribute set carries no trace of it.
                // Inventing a spread would price maps on a guess. `None` falls back to the
                // uniform list, which is what this path already did.
                note_difficulty_bins: None,
                input_state_bins: decode_input_state_bins(&js_attrs.input_state_bins),
                // Not carried through JS: it is a property of how the score was
                // played, not of the map, so it is re-derived from the mods that
                // came back with the attributes. `lazer` is not part of the shape
                // either, so this follows the same default the difficulty calc uses.
                ln_judged_as_one: sunny::is_classic(None, &mods),
            };

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
}
