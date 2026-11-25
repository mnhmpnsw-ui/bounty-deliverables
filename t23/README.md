# t=23 DISPATCH-06 — Multi-Channel Event Kinematics & Rhythm Anomaly Classifier (clean-room)

Zero-dependency Python 3.12 (stdlib only). One offline command:

    python3 kinematics_auth.py --seed 42323 --out t23_results.json

- `synth_traces.py` — synthesizes normalized 2D trajectory ticks + DIRECT_KEY / VIRTUAL_GRID / BATCH_CHUNK symbol bursts with backspace/replace edits for 4 profiles (atlas, vela, onyx, kest) across regime speeds.
- `kinematics_auth.py` — calibrates fixed decision thresholds strictly PRIOR to holdout (declared multipliers, calibration split only), then evaluates: cross-profile swap matrix (>=2 regime speeds), stranger FAR, fatigued-self FRR.

Fresh measured results (seed 42323, deterministic):

- swap matrix: diagonal 3/3 + d=0.38-0.49 accepted; all off-diagonal 0/3 (12/12 OK)
- FAR (stranger): 0/144 = 0.000
- FRR (fatigued-self): 0/24 = 0.000
- Threshold policy fixed pre-test: T_MEAN_MULT=2.5, T_STD_MULT=3.0, REL_SLACK=0.2, ABS_FLOOR=0.03, PAUSE_K=8.0

Solana payout wallet: 9qvXLTKsDgbfSg3Aczya4Rg3TEMCk1oQAsQNSgWThgc3
