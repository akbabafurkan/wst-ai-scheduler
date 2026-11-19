"""
WST scheduler experiment:
  1) Build the ephemeris + programme pool
  2) Train the learning scheduler with CEM (training weather seeds)
  3) Compare 4 policies on unseen weather/date blocks
  4) Produce the figures

Usage:  python run_experiment.py            (full run, a few minutes)
        python run_experiment.py --quick    (quick trial)
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_timeline import plot_night

from wstsim import (Ephemeris, Survey, Policy, run_block, train_cem, checkpoint,
                    FEATURE_NAMES, HANDTUNED_W, SLOT_MIN, CLOUD_NAMES)

QUICK = "--quick" in sys.argv
BLOCK = 28                      # one lunation
N_TEST = 10 if QUICK else 30
OUT = "figures"

import os
os.makedirs(OUT, exist_ok=True)

eph = Ephemeris()
sv = Survey(seed=42)
Pok = sv.calibrate_scarcity(eph, np.random.default_rng(1))
checkpoint("Probability that programme constraints are met: " +
           ", ".join(f"{n.split(' ')[0]}={p:.2f}" for n, p in zip(sv.names, Pok)))

# ---------------------------------------------------------------- training
checkpoint("Starting CEM training...")
w_learned, hist = train_cem(sv, eph, n_iter=6 if QUICK else 25, pop=16 if QUICK else 32,
                            n_eval=2 if QUICK else 3, block_nights=BLOCK, seed=0)
json.dump(dict(features=FEATURE_NAMES, w=w_learned.tolist()),
          open("learned_weights.json", "w"), indent=2)
checkpoint("Learned weights written to learned_weights.json")

# ---------------------------------------------------------------- test
policies = {
    "Random": Policy("random", rng=np.random.default_rng(5)),
    "Greedy priority": Policy("greedy", rng=np.random.default_rng(5)),
    "Hand-tuned merit": Policy("handtuned"),
    "Learned (CEM)": Policy("learned", w=w_learned),
}
rng_t = np.random.default_rng(12345)
test_starts = np.linspace(0, 364, N_TEST).astype(int)      # spread over the year
test_seeds = rng_t.integers(100_000, 200_000, N_TEST)      # not seen in training
res = {k: [] for k in policies}
checkpoint(f"Test: {N_TEST} blocks x {BLOCK} nights, 4 policies")
for st, sd in zip(test_starts, test_seeds):
    for k, pol in policies.items():
        res[k].append(run_block(sv, eph, pol, st, BLOCK, sd))
checkpoint("Test finished")

summary = {}
ref = np.array([r["score"] for r in res["Greedy priority"]])
for k, rs in res.items():
    sc = np.array([r["score"] for r in rs])
    summary[k] = dict(
        score_rel=float(np.mean(sc / ref)), score_rel_std=float(np.std(sc / ref)),
        win_vs_greedy=float(np.mean(sc > ref)),
        completion=np.mean([r["completion"] for r in rs], axis=0).tolist(),
        wasted=float(np.mean([r["n_wasted"] / max(r["n_obs"], 1) for r in rs])),
        idle=float(np.mean([r["n_idle"] / r["n_dark"] for r in rs])),
        airmass=float(np.nanmean([r["mean_airmass"] for r in rs])),
        too=float(np.sum([r["too_done"] for r in rs]) / max(np.sum([r["too_total"] for r in rs]), 1)),
        too_lat=float(np.nanmedian([r["too_latency"] for r in rs])),
    )
json.dump(summary, open("results_summary.json", "w"), indent=2)
for k, s in summary.items():
    checkpoint(f"{k:17s} score={s['score_rel']:.3f}±{s['score_rel_std']:.3f} x greedy | "
               f"wins={s['win_vs_greedy']:.0%} | wasted={s['wasted']:.1%} | "
               f"X={s['airmass']:.2f} | ToO={s['too']:.0%} (median latency {s['too_lat']:.0f} min)")

# ---------------------------------------------------------------- figures
cols = {"Random": "#9e9e9e", "Greedy priority": "#4c72b0",
        "Hand-tuned merit": "#dd8452", "Learned (CEM)": "#55a868"}
names = list(policies)

# Fig 1: overall comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i, k in enumerate(names):
    v = [r["score"] for r in res[k]] / ref
    ax[0].boxplot(v, positions=[i], widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor=cols[k], alpha=0.8), medianprops=dict(color="k"))
ax[0].axhline(1, ls="--", c="k", lw=0.8)
ax[0].set_xticks(range(len(names)), names, rotation=20, ha="right")
ax[0].set_ylabel("Priority-weighted science score / greedy")
ax[0].set_title(f"Science return ({N_TEST} unseen {BLOCK}-night blocks)")

prog_labels = [n.split(" (")[0] for n in sv.names]
x = np.arange(len(prog_labels))
bw = 0.2
for i, k in enumerate(names):
    ax[1].bar(x + (i - 1.5) * bw, np.array(summary[k]["completion"]) * 100, bw,
              color=cols[k], label=k)
ax[1].set_xticks(x, prog_labels, rotation=25, ha="right")
ax[1].set_ylabel("Completion per block [%]")
ax[1].set_title("Programme completion")
ax[1].legend(fontsize=8)

metrics = [("wasted", "Wasted exposures [%]", 100), ("idle", "Idle dark time [%]", 100),
           ("too", "ToO success [%]", 100)]
xm = np.arange(len(metrics))
for i, k in enumerate(names):
    ax[2].bar(xm + (i - 1.5) * bw, [summary[k][m] * f for m, _, f in metrics], bw,
              color=cols[k])
ax[2].set_xticks(xm, [m[1] for m in metrics], rotation=15, ha="right")
ax[2].set_title("Operational metrics")
for i, k in enumerate(names):
    ax[2].text(2 + (i - 1.5) * bw, summary[k]["too"] * 100 + 1,
               f"X={summary[k]['airmass']:.2f}", rotation=90, fontsize=7, ha="center", va="bottom")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_policy_comparison.png", dpi=140)

# Fig 2: training curve + learned weights
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
it = [h["it"] + 1 for h in hist]
ax[0].plot(it, [h["best"] for h in hist], "o-", c="#55a868", label="best in population")
ax[0].plot(it, [h["mean"] for h in hist], "s-", c="#8172b3", label="population mean")
ax[0].plot(it, [h["mu_fit"] for h in hist], "^-", c="k", label="current mean policy")
ax[0].axhline(1, ls="--", c="#4c72b0", label="greedy (same conditions)")
ax[0].set_xlabel("CEM iteration")
ax[0].set_ylabel("Score / greedy")
ax[0].set_title("Training (simulated weather, random start dates)")
ax[0].legend(fontsize=8)
y = np.arange(len(FEATURE_NAMES))
ax[1].barh(y + 0.2, HANDTUNED_W, 0.4, color="#dd8452", label="hand-tuned")
ax[1].barh(y - 0.2, w_learned, 0.4, color="#55a868", label="learned")
ax[1].set_yticks(y, FEATURE_NAMES)
ax[1].axvline(0, c="k", lw=0.8)
ax[1].invert_yaxis()
ax[1].set_xlabel("Merit weight")
ax[1].set_title("What the scheduler learned to value")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_training_and_weights.png", dpi=140)

# Fig 3: single-night timeline (greedy vs learned), same weather
plot_night(sv, eph, policies, int(test_starts[N_TEST // 2]), int(test_seeds[N_TEST // 2]),
           BLOCK, f"{OUT}/fig3_one_night_timeline.png")
checkpoint(f"Figures saved to {OUT}/")
