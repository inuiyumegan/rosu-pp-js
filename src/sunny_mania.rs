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
    sunny::{self, SunnyManiaDifficultyAttributes, SunnyScoreState},
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
    /// The mods used for the calculation, kept for the performance calc.
    #[serde(skip)]
    pub(crate) mods: rosu_mods::GameMods,
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
            mods: GameMods::default(),
        }
    }
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
}

impl From<sunny::SunnyManiaPerformanceAttributes> for JsSunnyManiaPerformanceAttributes {
    fn from(attrs: sunny::SunnyManiaPerformanceAttributes) -> Self {
        Self {
            pp: attrs.pp,
            pp_difficulty: attrs.pp_difficulty,
            variety_multiplier: attrs.variety_multiplier,
            acc_multiplier: attrs.acc_multiplier,
            length_multiplier: attrs.length_multiplier,
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
    pub fn calculate(&self, value: &wasm_bindgen::JsValue) -> JsResult<JsSunnyManiaPerformanceAttributes> {
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

            let attrs = SunnyManiaDifficultyAttributes {
                stars: js_attrs.stars,
                variety: js_attrs.variety,
                acc_scalar: js_attrs.acc_scalar,
                spikiness: js_attrs.spikiness,
                switches: js_attrs.switches,
                great_hit_window: js_attrs.great_hit_window,
                max_combo: js_attrs.max_combo,
                n_objects: js_attrs.n_objects as usize,
            };

            return Ok((attrs, mods));
        }

        if let Ok(map) = JsBeatmap::deserialize(crate::deserializer::JsDeserializer::from_ref(value))
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
fn prepare_map<'m>(
    args: &DifficultyArgs,
    map: &'m JsBeatmap,
) -> JsResult<Cow<'m, Beatmap>> {
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
