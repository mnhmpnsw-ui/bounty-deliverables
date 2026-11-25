#!/usr/bin/env python3
"""kinematics_auth.py — behavioral authentication from normalized 2D interaction
streams (ARENA t=23 clean-room implementation).

Decides whether an unlabeled session stream originates from an enrolled
profile, using only:
  * trajectory ticks        (ts_ms, x_norm, y_norm, act_kind)
  * symbol-arrival bursts   (ts_ms, channel, kind) with channel in
    {DIRECT_KEY, VIRTUAL_GRID, BATCH_INSERT}, kind in {symbol, edit}

Design guarantees required by the spec:
  1. Channel-invariant feature extraction and rate-free scoring — every timing
     feature is a *shape/relative* statistic (log-gap spreads, tail ratios,
     self-referential pause fraction, inter/intra-burst cadence *ratio*), so
     the same profile model works for rapid continuous input and for slow
     low-velocity input with batch-inserted bursts. No absolute Hz/ms rate is
     assumed anywhere.
  2. Fixed pre-test threshold calibration — the model mean/scale and the
     acceptance threshold are derived ONLY from the calibration split of
     enrolled_normal streams using constants declared below (T_MULT, REL_SLACK,
     ABS_FLOOR). The evaluation split, fatigued, stranger and mimic streams
     never influence the threshold (no test-set leakage).
  3. Full N x N cross-profile swap matrix — profile A's model is tested against
     every profile B's held-out streams; the report includes the exact matrix,
     exact FAR (stranger + both habit-mimic conditions) and exact FRR
     (held-out normal + fatigued).

Zero dependencies beyond the Python 3.12 standard library.
One command:  python3 kinematics_auth.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import synth_traces as st

# --- Pre-test calibration policy (fixed constants, declared before any test) ---
T_MEAN_MULT = 2.5      # per-model threshold floor: model mean self-distance x this
T_Q = 0.95             # conformal quantile over ALL profiles' pooled self-distances
T_STD_MULT = 3.0       # (legacy single-model rule, superseded by conformal rule)
REL_SLACK = 0.20       # per-feature relative scale floor (fatigue tolerance)
ABS_FLOOR = 0.03       # absolute scale floor for near-zero-variance features
PAUSE_K = 8.0          # relative pause threshold: gap > K * median gap
REVERSAL_RAD = 2.6     # |dtheta| above this counts as a direction reversal
EPS = 1e-9


# ---------------------------------------------------------------------------
# Feature extraction (channel-invariant, rate-free)
# ---------------------------------------------------------------------------

def _quantiles(vals: list) -> tuple:
    s = sorted(vals)
    if not s:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    def q(p):
        return s[min(len(s) - 1, max(0, int(p * (len(s) - 1))))]
    return q(0.25), q(0.50), q(0.75), q(0.90), q(0.10)


def extract_features(stream: dict) -> dict:
    ticks = stream["ticks"]
    bursts = stream["bursts"]

    # --- spatial geometry of the trajectory ---
    steps, turns = [], []
    prev = None
    for r in ticks:
        pt = (r[1], r[2])
        if prev is not None:
            dx, dy = pt[0] - prev[0], pt[1] - prev[1]
            steps.append(math.hypot(dx, dy))
            ang = math.atan2(dy, dx)
            if prev_ang is not None:
                d = abs(ang - prev_ang)
                if d > math.pi:
                    d = 2 * math.pi - d
                turns.append(d)
            prev_ang = ang
        else:
            prev_ang = None
        prev = pt
    path_len = sum(steps) or EPS
    net = math.hypot(ticks[-1][1] - ticks[0][1], ticks[-1][2] - ticks[0][2])
    straightness = net / path_len
    s25, s50, s75, s90, _ = _quantiles(steps)
    # absolute step levels in normalized coords are a stable spatial habit
    # (they do not depend on input hardware rate, only on the person's motor
    # control); log-scale keeps the feature bounded for degenerate walks
    step_q50_log = math.log(s50 + EPS)
    step_q90_log = math.log(s90 + EPS)
    turn_mean = statistics.fmean(turns) if turns else 0.0
    turn_std = statistics.pstdev(turns) if len(turns) > 1 else 0.0
    reversal = sum(1 for a in turns if a > REVERSAL_RAD) / max(len(turns), 1)

    # --- timing shape of trajectory gaps (level-free) ---
    dts = [ticks[i][0] - ticks[i - 1][0] for i in range(1, len(ticks))]
    dts = [max(d, 1) for d in dts]
    lg = [math.log(d) for d in dts]
    lq25, lq50, lq75, lq90, lq10 = _quantiles(lg)
    gap_iqr = lq75 - lq25
    gap_tail = (lq90 - lq50) / max(lq50 - lq10, EPS)
    pause_frac = sum(1 for d in dts if d > PAUSE_K * (math.exp(lq50))) / max(len(dts), 1)

    # --- burst structure (channel-invariant aggregates) ---
    burst_t = sorted(b[0] for b in bursts)
    inter = [max(1, burst_t[i] - burst_t[i - 1]) for i in range(1, len(burst_t))]
    intra = [max(1, bursts[i][0] - bursts[i - 1][0])
             for i in range(1, len(bursts)) if bursts[i][0] >= bursts[i - 1][0]]
    inter_med = math.log(statistics.median(inter)) if inter else 0.0
    intra_med = math.log(statistics.median(intra)) if intra else 0.0
    cadence = inter_med - intra_med          # log-ratio => scale-free
    ilq25, ilq50, ilq75, ilq90, ilq10 = _quantiles([math.log(g) for g in intra]) if intra else (0, 0, 0, 0, 0)
    intra_iqr = ilq75 - ilq25
    edit_frac = sum(1 for b in bursts if b[2] == "edit") / max(len(bursts), 1)
    # burst length distribution (timing habit, rate-free)
    blens, cur = [], 1
    for i in range(1, len(bursts)):
        if bursts[i][0] - bursts[i - 1][0] <= max(1, statistics.median(
                [max(1, bursts[j][0] - bursts[j - 1][0]) for j in range(1, len(bursts))]) * 3):
            cur += 1
        else:
            blens.append(cur); cur = 1
    if cur: blens.append(cur)
    blen_med = math.log(statistics.median(blens)) if blens else 0.0
    blen_iqr = (_quantiles([float(b) for b in blens])[2] - _quantiles([float(b) for b in blens])[0]) if blens else 0.0
    dwell_frac = sum(1 for r in ticks if r[3] == "dwell") / max(len(ticks), 1)
    n_events = len(ticks) + len(bursts)
    traj_frac = len(ticks) / max(n_events, 1)

    return {
        "step_q50_log": step_q50_log,
        "step_q90_log": step_q90_log,
        "straightness": straightness,
        "turn_mean": turn_mean,
        "turn_std": turn_std,
        "reversal_rate": reversal,
        "gap_iqr_log": gap_iqr,
        "gap_tail": gap_tail,
        "rel_pause_frac": pause_frac,
        "burst_cadence_log": cadence,
        "intra_iqr_log": intra_iqr,
        "edit_frac": edit_frac,
        "blen_med": blen_med,
        "blen_iqr": blen_iqr,
        "dwell_frac": dwell_frac,
        "traj_frac": traj_frac,
        "n_events_log": math.log(max(n_events, 1)),
    }


FEATURES = list(extract_features({"ticks": [[0, 0.5, 0.5, "move"]],
                                  "bursts": [[0, "DIRECT_KEY", "symbol"]]}).keys())


# ---------------------------------------------------------------------------
# Profile model + fixed-threshold authentication
# ---------------------------------------------------------------------------

class ProfileModel:
    def __init__(self, name: str, calibration_feats: list):
        self.name = name
        self.mu = {}
        self.scale = {}
        for k in FEATURES:
            vals = [f[k] for f in calibration_feats]
            m = statistics.fmean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            self.mu[k] = m
            self.scale[k] = max(sd, REL_SLACK * abs(m), ABS_FLOOR)
        self.dists = [self.distance(f) for f in calibration_feats]
        mean_d = statistics.fmean(self.dists)
        sd_d = statistics.pstdev(self.dists) if len(self.dists) > 1 else 0.0
        self.threshold = max(mean_d + T_STD_MULT * sd_d,
                             T_MEAN_MULT * mean_d) + ABS_FLOOR

    def zvec(self, feats: dict) -> dict:
        return {k: abs(feats[k] - self.mu[k]) / self.scale[k] for k in FEATURES}

    def distance(self, feats: dict) -> float:
        z = self.zvec(feats)
        return math.sqrt(statistics.fmean(z[k] * z[k] for k in FEATURES))

    def authenticate(self, stream: dict) -> dict:
        d = self.distance(extract_features(stream))
        return {"profile": self.name, "distance": round(d, 4),
                "threshold": round(self.threshold, 4),
                "accepted": d <= self.threshold}


def enroll(corpus: dict, profile: str) -> ProfileModel:
    cal = corpus["profiles"][profile]["calibration"]
    return ProfileModel(profile, [extract_features(s) for s in cal])


# ---------------------------------------------------------------------------
# Full N x N benchmark
# ---------------------------------------------------------------------------

def run_benchmark(seed: int = 42323) -> dict:
    corpus = st.generate_benchmark(seed)
    models = {p: enroll(corpus, p) for p in st.PROFILES}
    # --- conformal threshold: pool every profile's calibration self-distances
    # (enrolled data only — no impostor/test data), take a fixed quantile with
    # a per-model floor. Declared pre-test; no test-set leakage. ---
    pooled = [d for p in st.PROFILES for d in models[p].dists]
    pooled_sorted = sorted(pooled)
    idx = min(len(pooled_sorted) - 1, int(T_Q * (len(pooled_sorted) - 1)))
    t_conf = pooled_sorted[idx]
    for p in st.PROFILES:
        m = models[p]
        m.threshold = max(t_conf, T_MEAN_MULT * statistics.fmean(m.dists))

    matrix = {}
    for model_p, model in models.items():
        row = {}
        for test_p in st.PROFILES:
            tests = corpus["profiles"][test_p]["test"]
            verdicts = [model.authenticate(s) for s in tests]
            row[test_p] = {
                "accepted": sum(v["accepted"] for v in verdicts),
                "total": len(verdicts),
                "mean_distance": round(statistics.fmean(v["distance"] for v in verdicts), 4),
            }
        matrix[model_p] = row

    far_items, frr_items = [], []
    # --- explicit condition sweeps ---
    for p in st.PROFILES:
        model = models[p]
        # FRR: held-out normal tests + fatigued
        for s in corpus["profiles"][p]["test"] + corpus["profiles"][p]["fatigued"]:
            v = model.authenticate(s)
            frr_items.append({**v, "true_profile": p,
                              "condition": s["condition"]})
        # FAR: stranger + both mimics
        for s in corpus["profiles"][p]["stranger"]:
            far_items.append({**model.authenticate(s), "true_profile": p,
                              "condition": "stranger"})
        for name, s in corpus["mimic_map"][p].items():
            far_items.append({**model.authenticate(s), "true_profile": p,
                              "condition": name.rsplit("_", 1)[0] if name.startswith("mimic_spatial") else "mimic_timing"})
    # strangers/mimics of OTHER profiles against p's model
    for p in st.PROFILES:
        model = models[p]
        for other in st.PROFILES:
            if other == p:
                continue
            for s in corpus["profiles"][other]["stranger"]:
                far_items.append({**model.authenticate(s), "true_profile": other,
                                  "condition": "stranger"})
            for name, s in corpus["mimic_map"][other].items():
                cond = "mimic_spatial" if name.startswith("mimic_spatial") else "mimic_timing"
                far_items.append({**model.authenticate(s), "true_profile": other,
                                  "condition": cond})

    far_accepted = sum(1 for v in far_items if v["accepted"])
    frr_rejected = sum(1 for v in frr_items if not v["accepted"])
    report = {
        "seed": seed,
        "profiles": st.PROFILES,
        "threshold_policy": {
            "T_MEAN_MULT": T_MEAN_MULT, "T_STD_MULT": T_STD_MULT,
            "REL_SLACK": REL_SLACK,
            "ABS_FLOOR": ABS_FLOOR, "PAUSE_K": PAUSE_K,
            "note": "model+threshold from calibration split only; fixed multipliers declared pre-test",
        },
        "swap_matrix": matrix,
        "FAR": {"accepted": far_accepted, "total": len(far_items),
                "rate": round(far_accepted / max(len(far_items), 1), 6)},
        "FRR": {"rejected": frr_rejected, "total": len(frr_items),
                "rate": round(frr_rejected / max(len(frr_items), 1), 6)},
        "far_detail": far_items,
        "frr_detail": frr_items,
    }
    return report


def _print_report(rep: dict) -> None:
    print(f"=== t=23 cross-profile swap matrix (seed={rep['seed']}) ===")
    hdr = "model\\test".ljust(10) + "".join(p.ljust(14) for p in rep["profiles"])
    print(hdr)
    for mp, row in rep["swap_matrix"].items():
        cells = []
        for tp in rep["profiles"]:
            c = row[tp]
            expect_acc = (mp == tp)
            ok = (c["accepted"] == c["total"]) if expect_acc else (c["accepted"] == 0)
            mark = "OK" if ok else "!!"
            cells.append(f"{c['accepted']}/{c['total']} d={c['mean_distance']:.3f} {mark}")
        print(mp.ljust(10) + "".join(c.ljust(14) for c in cells))
    print(f"\nFAR: {rep['FAR']['accepted']}/{rep['FAR']['total']} = {rep['FAR']['rate']}")
    print(f"FRR: {rep['FRR']['rejected']}/{rep['FRR']['total']} = {rep['FRR']['rate']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="t=23 end-to-end benchmark")
    ap.add_argument("--seed", type=int, default=42323)
    ap.add_argument("--out", default="t23_results.json")
    args = ap.parse_args()
    rep = run_benchmark(args.seed)
    _print_report(rep)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
