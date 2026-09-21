//! Minimal Node/Wasm bridge for the public rosu-pp v4 mania APIs.
//!
//! This deliberately does not mirror rosu-pp-js v3's generic surface. It keeps
//! the server boundary small and ensures both algorithms remain implemented only
//! in rosu-pp.

use js_sys::Reflect;
use rosu_pp::{
    Beatmap, Difficulty, GameMods, Performance,
    mania::{SunnyScoreState, sunny},
    model::mode::GameMode,
};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

fn error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn property(args: &JsValue, name: &str) -> Option<JsValue> {
    if args.is_undefined() || args.is_null() {
        return None;
    }
    Reflect::get(args, &JsValue::from_str(name))
        .ok()
        .filter(|value| !value.is_undefined() && !value.is_null())
}

fn number(args: &JsValue, name: &str) -> Option<f64> {
    property(args, name)?.as_f64()
}
fn u32_value(args: &JsValue, name: &str) -> u32 {
    number(args, name).unwrap_or(0.0).max(0.0) as u32
}
fn bool_value(args: &JsValue, name: &str) -> Option<bool> {
    property(args, name)?.as_bool()
}

#[derive(Clone)]
struct CalculationArgs {
    mods: GameMods,
    clock_rate: f64,
    lazer: bool,
    passed_objects: Option<u32>,
}

impl CalculationArgs {
    fn from_js(args: &JsValue) -> Self {
        let bits = number(args, "mods").unwrap_or(0.0).max(0.0) as u32;
        let mods = GameMods::from(bits);
        let default_rate = if bits & (64 | 512) != 0 {
            1.5
        } else if bits & 256 != 0 {
            0.75
        } else {
            1.0
        };
        let clock_rate = number(args, "clockRate").unwrap_or(default_rate);
        Self {
            mods,
            clock_rate,
            lazer: bool_value(args, "lazer").unwrap_or(false),
            passed_objects: number(args, "passedObjects").map(|v| v.max(0.0) as u32),
        }
    }
}

fn mania_map(map: &JsBeatmap, mods: &GameMods) -> Result<Beatmap, JsValue> {
    let mut map = map.inner.clone();
    map.convert_mut(GameMode::Mania, mods).map_err(error)?;
    Ok(map)
}

#[wasm_bindgen(js_name = Beatmap)]
pub struct JsBeatmap {
    inner: Beatmap,
}

#[wasm_bindgen(js_class = Beatmap)]
impl JsBeatmap {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Self, JsValue> {
        Ok(Self {
            inner: Beatmap::from_bytes(bytes).map_err(error)?,
        })
    }
    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> u8 {
        match self.inner.mode {
            GameMode::Osu => 0,
            GameMode::Taiko => 1,
            GameMode::Catch => 2,
            GameMode::Mania => 3,
        }
    }
    #[wasm_bindgen(js_name = isSuspicious)]
    pub fn is_suspicious(&self) -> bool {
        self.inner.check_suspicion().is_err()
    }
    pub fn convert(&mut self, mode: u8) -> Result<(), JsValue> {
        let mode = match mode {
            0 => GameMode::Osu,
            1 => GameMode::Taiko,
            2 => GameMode::Catch,
            3 => GameMode::Mania,
            _ => return Err(JsValue::from_str("invalid game mode")),
        };
        self.inner
            .convert_mut(mode, &GameMods::default())
            .map_err(error)
    }
    pub fn free(self) {}
}

#[wasm_bindgen]
pub struct ManiaResult {
    #[wasm_bindgen(readonly)]
    pub pp: f64,
    #[wasm_bindgen(readonly)]
    pub stars: f64,
}

fn sunny_state(args: &JsValue) -> SunnyScoreState {
    SunnyScoreState {
        n320: u32_value(args, "nGeki"),
        n300: u32_value(args, "n300"),
        n200: u32_value(args, "nKatu"),
        n100: u32_value(args, "n100"),
        n50: u32_value(args, "n50"),
        misses: u32_value(args, "misses"),
    }
}

#[wasm_bindgen(js_name = RebirthManiaDifficulty)]
pub struct RebirthManiaDifficulty {
    args: CalculationArgs,
}

#[wasm_bindgen(js_class = RebirthManiaDifficulty)]
impl RebirthManiaDifficulty {
    #[wasm_bindgen(constructor)]
    pub fn new(args: JsValue) -> Self {
        Self {
            args: CalculationArgs::from_js(&args),
        }
    }

    pub fn calculate(&self, map: &JsBeatmap) -> Result<ManiaResult, JsValue> {
        let map = mania_map(map, &self.args.mods)?;
        let attrs = Difficulty::new()
            .mods(self.args.mods.clone())
            .clock_rate(self.args.clock_rate)
            .lazer(self.args.lazer)
            .calculate(&map);

        Ok(ManiaResult {
            pp: 0.0,
            stars: attrs.stars(),
        })
    }
}

#[wasm_bindgen(js_name = RebirthManiaPerformance)]
pub struct RebirthManiaPerformance {
    args: CalculationArgs,
    raw: JsValue,
}

#[wasm_bindgen(js_class = RebirthManiaPerformance)]
impl RebirthManiaPerformance {
    #[wasm_bindgen(constructor)]
    pub fn new(args: JsValue) -> Self {
        Self {
            args: CalculationArgs::from_js(&args),
            raw: args,
        }
    }

    pub fn calculate(&self, map: &JsBeatmap) -> Result<ManiaResult, JsValue> {
        let map = mania_map(map, &self.args.mods)?;
        let attrs = Performance::new(&map)
            .mods(self.args.mods.clone())
            .clock_rate(self.args.clock_rate)
            .lazer(self.args.lazer)
            .n_geki(u32_value(&self.raw, "nGeki"))
            .n300(u32_value(&self.raw, "n300"))
            .n_katu(u32_value(&self.raw, "nKatu"))
            .n100(u32_value(&self.raw, "n100"))
            .n50(u32_value(&self.raw, "n50"))
            .misses(u32_value(&self.raw, "misses"))
            .calculate();

        Ok(ManiaResult {
            pp: attrs.pp(),
            stars: attrs.stars(),
        })
    }
}

#[wasm_bindgen(js_name = SunnyManiaDifficulty)]
pub struct SunnyManiaDifficulty {
    args: CalculationArgs,
}
#[wasm_bindgen(js_class = SunnyManiaDifficulty)]
impl SunnyManiaDifficulty {
    #[wasm_bindgen(constructor)]
    pub fn new(args: JsValue) -> Self {
        Self {
            args: CalculationArgs::from_js(&args),
        }
    }
    pub fn calculate(&self, map: &JsBeatmap) -> Result<ManiaResult, JsValue> {
        let map = mania_map(map, &self.args.mods)?;
        let attrs = sunny::calculate(
            &map,
            &self.args.mods,
            self.args.clock_rate,
            Some(self.args.lazer),
            self.args.passed_objects,
        )
        .ok_or_else(|| JsValue::from_str("sunny calculation requires at least 2 hit objects"))?;
        Ok(ManiaResult {
            pp: 0.0,
            stars: attrs.stars,
        })
    }
}

#[wasm_bindgen(js_name = SunnyManiaPerformance)]
pub struct SunnyManiaPerformance {
    args: CalculationArgs,
    raw: JsValue,
}
#[wasm_bindgen(js_class = SunnyManiaPerformance)]
impl SunnyManiaPerformance {
    #[wasm_bindgen(constructor)]
    pub fn new(args: JsValue) -> Self {
        Self {
            args: CalculationArgs::from_js(&args),
            raw: args,
        }
    }
    pub fn calculate(&self, map: &JsBeatmap) -> Result<ManiaResult, JsValue> {
        let map = mania_map(map, &self.args.mods)?;
        let attrs = sunny::calculate(
            &map,
            &self.args.mods,
            self.args.clock_rate,
            Some(self.args.lazer),
            self.args.passed_objects,
        )
        .ok_or_else(|| JsValue::from_str("sunny calculation requires at least 2 hit objects"))?;
        let perf = sunny::calculate_performance(&attrs, &self.args.mods, sunny_state(&self.raw));
        Ok(ManiaResult {
            pp: perf.pp,
            stars: attrs.stars,
        })
    }
}
