// End-to-end test for the sunny mania algorithm (wasm build).
// Usage: node test-sunny.mjs <path-to-pkg> <path-to-osu>

import * as rosu from "./pkg/rosu_pp_js.js";
import * as fs from "fs";

const MAP = process.argv[2] ?? "C:\\Users\\uuzof\\AppData\\Local\\Temp\\opencode\\rosu-pp\\resources\\1638954.osu";

const bytes = fs.readFileSync(MAP);
const map = new rosu.Beatmap(bytes);
console.log("mode:", map.mode, "| cs:", map.cs, "| od:", map.od, "| objects:", map.nObjects);

// --- Difficulty ---
const diff = new rosu.SunnyManiaDifficulty();
const attrs = diff.calculate(map);
console.log("\n[SunnyManiaDifficulty NM]");
console.log("stars:", attrs.stars.toFixed(6));
console.log("variety:", attrs.variety.toFixed(6));
console.log("accScalar:", attrs.accScalar.toFixed(6));
console.log("spikiness:", attrs.spikiness.toFixed(6));
console.log("switches:", attrs.switches.toFixed(6));
console.log("greatHitWindow:", attrs.greatHitWindow.toFixed(6));
console.log("maxCombo:", attrs.maxCombo, "| nObjects:", attrs.nObjects);

// --- EZ / HR effect on SR ---
const diffEZ = new rosu.SunnyManiaDifficulty({ mods: "EZ" });
const attrsEZ = diffEZ.calculate(map);
console.log("\nEZ stars:", attrsEZ.stars.toFixed(6), "(expect < NM)");

const diffHR = new rosu.SunnyManiaDifficulty({ mods: "HR" });
const attrsHR = diffHR.calculate(map);
console.log("HR stars:", attrsHR.stars.toFixed(6), "(expect > NM)");

// --- Performance (SS) ---
const perf = new rosu.SunnyManiaPerformance({
    mods: "NM",
    nGeki: attrs.nObjects,
    n300: 0,
    misses: 0,
});
const perfAttrs = perf.calculate(attrs);
console.log("\n[SunnyManiaPerformance SS]");
console.log("pp:", perfAttrs.pp.toFixed(6));
console.log("ppDifficulty:", perfAttrs.ppDifficulty.toFixed(6));
console.log("varietyMultiplier:", perfAttrs.varietyMultiplier.toFixed(6));
console.log("accMultiplier:", perfAttrs.accMultiplier.toFixed(6));
console.log("lengthMultiplier:", perfAttrs.lengthMultiplier.toFixed(6));

// --- Performance with EZ and NF mods (multipliers 0.9 * 0.75) ---
const perfEZNF = new rosu.SunnyManiaPerformance({
    mods: "EZNF",
    nGeki: attrs.nObjects,
});
const perfAttrsEZNF = perfEZNF.calculate(attrs);
console.log("\nEZNF pp:", perfAttrsEZNF.pp.toFixed(6), "(expect * 0.675 of NM)");

// --- Performance from beatmap directly ---
const perfFromMap = new rosu.SunnyManiaPerformance({ nGeki: attrs.nObjects });
const perfFromMapAttrs = perfFromMap.calculate(map);
console.log("\npp from map directly:", perfFromMapAttrs.pp.toFixed(6));

// --- JSON serialization ---
console.log("\nattrs.toJSON():", JSON.stringify(attrs));

map.free();
