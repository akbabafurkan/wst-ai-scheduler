"""
Single-night timeline: greedy vs learned scheduler under identical weather, starting from
the same survey state. A night with moonrise/moonset and changing clouds is preferred.

Standalone use (requires learned_weights.json):  python plot_timeline.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from wstsim import (Ephemeris, Survey, Policy, run, night_weather, checkpoint,
                    CLOUD_NAMES, STEP_MIN, STEPS_PER_NIGHT)


def pick_interesting_night(eph, start, n_nights, seed):
    best, best_n = -1, start
    for n in range(start, start + n_nights):
        d = eph.dark[n]
        if d.sum() == 0:
            continue
        closed, cloud, _ = night_weather(seed, n)
        if closed or np.mean(cloud[d] == 2) > 0.3:
            continue
        fli_eff = np.where(eph.moon_alt[n] > 0, eph.fli[n], 0)[d]
        up = np.mean(fli_eff > 0)
        val = 4 * up * (1 - up) * eph.fli[n][d].mean() + 0.05 * np.sum(np.diff(cloud[d]) != 0)
        if val > best:
            best, best_n = val, n
    return best_n


def plot_night(sv, eph, policies, night, seed, out, state=None):
    cmap = plt.get_cmap("tab20")
    pcols = [cmap(i % 20) for i in range(len(sv.names))]
    runs = {k: run(sv, eph, pol, night, 1, seed, state=state, log_nights=(0,))
            for k, pol in policies.items()}
    fig, ax = plt.subplots(2 + len(runs), 1, figsize=(13, 8.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1] + [1.1] * len(runs)))
    log0 = runs[list(runs)[0]]["logs"][0]
    hh = (np.arange(STEPS_PER_NIGHT) + 0.5) * STEP_MIN / 60
    d = log0["dark"]
    ax[0].plot(hh[d], log0["seeing"][d], "o-", c="#c44e52", ms=3)
    ax[0].set_ylabel('Zenith seeing ["]')
    ax[0].set_title(f"One simulated night (night {night} of the survey, from 2041-01-01); "
                    "hatched = 180 s exposure, plain = 1000 s, black = no usable spectra, * = ToO")
    ax[1].step(hh[d], log0["cloud"][d], where="mid", c="#4c72b0")
    ax[1].set_yticks([0, 1, 2], CLOUD_NAMES)
    ax[1].set_ylabel("Cloud", color="#4c72b0")
    ax2 = ax[1].twinx()
    ax2.plot(hh[d], log0["fli"][d], c="#ccb974", lw=2)
    ax2.set_ylabel("Moon FLI (0 if below horizon)", color="#a08c2c", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    for row, (k, r) in enumerate(runs.items(), start=2):
        for e in r["logs"][0]["exps"]:
            col = "k" if e["prog"] < 0 else pcols[e["prog"]]
            ax[row].barh(0.5, e["dur"], left=e["t"], height=0.8, color=col,
                         alpha=0.6 if e["prog"] < 0 else 1.0, edgecolor="w", lw=0.4,
                         hatch="///" if e["mode"] == 0 else None)
            if e["too"]:
                ax[row].plot(e["t"] + e["dur"] / 2, 0.5, "*", c="gold", ms=12, mec="k")
        a = r["acc"]
        ax[row].set_ylim(0, 1)
        ax[row].set_yticks([])
        ax[row].set_ylabel(f"{k}\nscore +{r['gain'] * 1e3:.2f}e-3", fontsize=9)
    ax[-1].set_xlabel("Hours since 21:00 UTC")
    used = sorted({e["prog"] for r in runs.values() for e in r["logs"][0]["exps"] if e["prog"] >= 0})
    handles = [Patch(color=pcols[p], label=sv.labels[p]) for p in used]
    handles += [Patch(color="k", alpha=0.6, label="no usable spectra")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    checkpoint(f"Timeline saved: {out} (night {night})")
    return runs


if __name__ == "__main__":
    eph = Ephemeris()
    sv = Survey(seed=42)
    sv.calibrate_scarcity(eph)
    w = np.array(json.load(open("learned_weights.json"))["w"])
    pols = {"Greedy (gain rate)": Policy("greedy"), "Learned (CEM)": Policy("learned", w=w)}
    # warm up the survey state for one lunation with the greedy policy, then plot a night
    warm = run(sv, eph, Policy("greedy"), 0, 60, seed=777)
    night = pick_interesting_night(eph, 60, 28, 777)
    plot_night(sv, eph, pols, night, 777, "figures/fig3_one_night_timeline.png",
               state=warm["state"])
