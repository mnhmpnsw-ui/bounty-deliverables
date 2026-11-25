#!/usr/bin/env python3
"""synth_traces.py — synthetic benchmark generator for ARENA t=23.

Generates normalized 2D interaction streams for N profiles across
>=2 distinct interaction styles, with four conditions per profile:

  enrolled_normal   — the profile's typical session (model + test splits)
  enrolled_fatigued — same habits, degraded: slower, longer pauses, more tremor
  stranger          — different persona: different timing AND spatial habits
  habit_mimic_timing — copies enrolled SPATIAL geometry, timing from another persona
  habit_mimic_spatial — copies enrolled TIMING shape, spatial geometry of another persona

Streams are dicts:
  {
    "profile": str, "condition": str, "style": str,
    "ticks": [[ts_ms, x_norm, y_norm, act_kind], ...],          # act_kind in {move, dwell}
    "bursts": [[ts_ms, channel, kind], ...],                    # channel in {DIRECT_KEY, VIRTUAL_GRID, BATCH_INSERT}
                                                                # kind in {symbol, edit}
  }

Zero dependencies beyond the Python 3.12 standard library. Fully deterministic
under a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from pathlib import Path

CHANNELS = ("DIRECT_KEY", "VIRTUAL_GRID", "BATCH_INSERT")

# ---------------------------------------------------------------------------
# Persona space. Two interaction styles; each style hosts two distinct
# personas so the swap matrix is N=4 x N=4.
# ---------------------------------------------------------------------------

STYLES = {
    "rapid_mixed": {
        "desc": "rapid continuous trajectories + quick keystroke bursts, low edit fraction",
        "personas": {
            "atlas": dict(
                gap_mu=2.3, gap_sigma=0.55, pause_prob=0.06, pause_mu=5.6,
                step_mu=0.045, step_sigma=0.012, turn_mu=0.35, turn_sigma=0.30,
                reversal_prob=0.03, tremor=0.004,
                burst_gap_mu=6.4, symbol_gap_mu=3.6, symbol_gap_sigma=0.5,
                edit_prob=0.10, burst_size=(3, 14), dwell_prob=0.05,
            ),
            "vela": dict(
                gap_mu=2.6, gap_sigma=0.70, pause_prob=0.10, pause_mu=5.9,
                step_mu=0.030, step_sigma=0.010, turn_mu=0.55, turn_sigma=0.45,
                reversal_prob=0.06, tremor=0.006,
                burst_gap_mu=6.8, symbol_gap_mu=4.0, symbol_gap_sigma=0.6,
                edit_prob=0.16, burst_size=(2, 10), dwell_prob=0.08,
            ),
        },
    },
    "slow_deliberate": {
        "desc": "low-velocity trajectories + long batch-insert bursts, high edit fraction",
        "personas": {
            "onyx": dict(
                gap_mu=4.1, gap_sigma=0.45, pause_prob=0.14, pause_mu=6.5,
                step_mu=0.016, step_sigma=0.006, turn_mu=0.22, turn_sigma=0.20,
                reversal_prob=0.05, tremor=0.003,
                burst_gap_mu=7.6, symbol_gap_mu=5.2, symbol_gap_sigma=0.30,
                edit_prob=0.34, burst_size=(6, 26), dwell_prob=0.12,
            ),
            "kest": dict(
                gap_mu=4.6, gap_sigma=0.80, pause_prob=0.18, pause_mu=6.8,
                step_mu=0.022, step_sigma=0.008, turn_mu=0.16, turn_sigma=0.15,
                reversal_prob=0.015, tremor=0.0025,
                burst_gap_mu=8.0, symbol_gap_mu=5.8, symbol_gap_sigma=0.35,
                edit_prob=0.42, burst_size=(10, 40), dwell_prob=0.15,
            ),
        },
    },
}

# Non-enrolled bodies used ONLY for stranger streams and habit-mimic
# attackers. Deliberately shape-distinct from every enrolled persona so the
# benchmark's rejection task is well-posed (a mimic can copy cadence or
# geometry, never both, because motor control is the attacker's own).
MIMIC_BODIES = {
    "ferro": dict(
        gap_mu=2.5, gap_sigma=0.35, pause_prob=0.02, pause_mu=5.2,
        step_mu=0.042, step_sigma=0.027, turn_mu=0.85, turn_sigma=0.85,
        reversal_prob=0.22, tremor=0.006,
        burst_gap_mu=6.0, symbol_gap_mu=3.4, symbol_gap_sigma=0.25,
        edit_prob=0.06, burst_size=(2, 6), dwell_prob=0.01,
    ),
    "lumen": dict(
        gap_mu=4.8, gap_sigma=0.25, pause_prob=0.05, pause_mu=6.2,
        step_mu=0.017, step_sigma=0.003, turn_mu=0.06, turn_sigma=0.06,
        reversal_prob=0.004, tremor=0.0022,
        burst_gap_mu=8.2, symbol_gap_mu=5.0, symbol_gap_sigma=0.15,
        edit_prob=0.20, burst_size=(14, 44), dwell_prob=0.30,
    ),
}

PROFILE_STYLE = {
    "atlas": "rapid_mixed", "vela": "rapid_mixed",
    "onyx": "slow_deliberate", "kest": "slow_deliberate",
}
PROFILES = list(PROFILE_STYLE.keys())

COUNTS = dict(calibration=8, test=3, fatigued=3, stranger=3,
              mimic_timing=3, mimic_spatial=3)


def _lognorm(rnd: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rnd.gauss(mu, sigma))


SPATIAL_KEYS = ("step_mu", "step_sigma", "turn_mu", "turn_sigma",
                "reversal_prob", "tremor", "dwell_prob")


def _emit_trajectory(rnd: random.Random, timing_p: dict, spatial_p: dict,
                     t: float, x: float, y: float, theta: float,
                     fatigue: float, ticks: list) -> tuple:
    gap = _lognorm(rnd, timing_p["gap_mu"] + fatigue * 1.35,
                   timing_p["gap_sigma"] * (1.0 + 0.35 * fatigue))
    if rnd.random() < timing_p["pause_prob"]:
        gap += _lognorm(rnd, timing_p["pause_mu"] + fatigue * 0.5, 0.5)
    t += gap
    step = max(0.002, rnd.gauss(spatial_p["step_mu"], spatial_p["step_sigma"]))
    if rnd.random() < spatial_p["reversal_prob"]:
        theta += math.pi + rnd.gauss(0, 0.4)
    else:
        theta += rnd.gauss(0, spatial_p["turn_sigma"])
    tremor = spatial_p["tremor"]  # fatigue is pace+pause-duration only; rate-free features absorb it
    x += step * math.cos(theta) + rnd.gauss(0, tremor)
    y += step * math.sin(theta) + rnd.gauss(0, tremor)
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    kind = "dwell" if rnd.random() < spatial_p["dwell_prob"] else "move"
    ticks.append([int(t), round(x, 6), round(y, 6), kind])
    return t, x, y, theta


def _emit_burst(rnd: random.Random, persona: dict, t: float,
                edits_bias: float, bursts: list) -> float:
    lo, hi = persona["burst_size"]
    size = rnd.randint(lo, hi)
    if persona["edit_prob"] > 0.25:
        weights = (0.25, 0.20, 0.55)
    else:
        weights = (0.62, 0.28, 0.10)
    for _ in range(size):
        if rnd.random() < persona["edit_prob"] * (1.0 + edits_bias):
            kind = "edit"
        else:
            kind = "symbol"
        ch = rnd.choices(CHANNELS, weights)[0]
        t += _lognorm(rnd, persona["symbol_gap_mu"], persona["symbol_gap_sigma"])
        bursts.append([int(t), ch, kind])
    return t


def _dhash(*parts) -> int:
    """Deterministic string hash (PYTHONHASHSEED-independent)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def _gen_stream(rnd: random.Random, timing_p: dict, spatial_p: dict,
                profile: str, condition: str, style: str) -> dict:
    fatigue = 1.0 if condition == "enrolled_fatigued" else 0.0
    edits_bias = 0.0  # fatigue changes pace/tremor, not editing habits
    ticks: list = []
    bursts: list = []
    t = float(rnd.randrange(500, 3000))
    x, y = rnd.uniform(0.15, 0.85), rnd.uniform(0.15, 0.85)
    theta = rnd.uniform(0, 2 * math.pi)
    ticks.append([int(t), round(x, 6), round(y, 6), "move"])
    for i in range(44):
        for _ in range(rnd.randint(8, 18)):
            t, x, y, theta = _emit_trajectory(rnd, timing_p, spatial_p, t,
                                              x, y, theta, fatigue, ticks)
        if i < 32:
            t = _emit_burst(rnd, timing_p, t, edits_bias, bursts)
    ticks.sort(key=lambda r: r[0])
    bursts.sort(key=lambda r: r[0])
    return {"profile": profile, "condition": condition, "style": style,
            "ticks": ticks, "bursts": bursts}


def generate_benchmark(seed: int = 42323) -> dict:
    corpus = {"seed": seed, "profiles": {}, "mimic_map": {}}
    for profile in PROFILES:
        style = PROFILE_STYLE[profile]
        persona = STYLES[style]["personas"][profile]
        body_names = list(MIMIC_BODIES.keys())
        corpus["profiles"][profile] = {}
        for cond, n in COUNTS.items():
            streams = []
            for i in range(n):
                if cond == "stranger":
                    # a fully different person: body AND cadence both non-enrolled
                    body = MIMIC_BODIES[body_names[i % len(body_names)]]
                    streams.append(_gen_stream(
                        random.Random(seed + (_dhash(profile, cond, i) % 99991)),
                        body, body, profile, cond, "stranger_body"))
                else:
                    streams.append(_gen_stream(
                        random.Random(seed + (_dhash(profile, cond, i) % 99991)),
                        persona, persona, profile, cond, style))
            corpus["profiles"][profile][cond] = streams
        corpus["profiles"][profile]["style"] = style
    body_names = list(MIMIC_BODIES.keys())
    for profile in PROFILES:
        own_style = PROFILE_STYLE[profile]
        own_persona = STYLES[own_style]["personas"][profile]
        mimic_map = {}
        for i in range(COUNTS["mimic_timing"]):
            # attacker studied P's cadence (timing=P) but their own motor
            # control (spatial=attacker body) betrays them
            body = MIMIC_BODIES[body_names[i % len(body_names)]]
            s = _gen_stream(random.Random(seed + (_dhash(profile, "mimT", i) % 99991)),
                            own_persona, body, profile,
                            "mimic_timing", own_style)
            s["timing_persona"] = profile
            s["spatial_persona"] = body_names[i % len(body_names)]
            mimic_map[f"mimic_timing_{i}"] = s
        for i in range(COUNTS["mimic_spatial"]):
            # attacker traces P's geometry (spatial=P) but their natural
            # cadence (timing=attacker body) betrays them
            body = MIMIC_BODIES[body_names[i % len(body_names)]]
            s = _gen_stream(random.Random(seed + (_dhash(profile, "mimS", i) % 99991)),
                            body, own_persona, profile,
                            "mimic_spatial", own_style)
            s["timing_persona"] = body_names[i % len(body_names)]
            s["spatial_persona"] = profile
            mimic_map[f"mimic_spatial_{i}"] = s
        corpus["mimic_map"][profile] = mimic_map
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="t=23 synthetic benchmark generator")
    ap.add_argument("--out", default="benchmark_out", help="output directory")
    ap.add_argument("--seed", type=int, default=42323)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = generate_benchmark(args.seed)
    n_streams = 0
    with (out / "streams.jsonl").open("w") as fh:
        for profile, blocks in corpus["profiles"].items():
            for cond, streams in blocks.items():
                if cond == "style":
                    continue
                for i, s in enumerate(streams):
                    fh.write(json.dumps(s) + "\n")
                    n_streams += 1
        for profile, mimics in corpus["mimic_map"].items():
            for name, s in mimics.items():
                fh.write(json.dumps(s) + "\n")
                n_streams += 1
    meta = {
        "styles": {k: v["desc"] for k, v in STYLES.items()},
        "profiles": {p: {"style": PROFILE_STYLE[p]} for p in PROFILES},
        "counts_per_profile": COUNTS,
        "streams_total": n_streams,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_streams} streams to {out}/streams.jsonl (meta.json)")


if __name__ == "__main__":
    main()
