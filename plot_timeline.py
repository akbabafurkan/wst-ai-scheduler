"""
Single-night timeline: greedy vs learned scheduler under identical weather.
An "interesting" night with moonrise/moonset and cloud changes is picked from the block.

Standalone use (requires learned_weights.json):  python plot_timeline.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from wstsim import (Ephemeris, Survey, Policy, run_block, checkpoint,
                    SLOT_MIN, CLOUD_NAMES, N_NIGHTS_YEAR)


def pick_interesting_night(eph, timeline, start, n_nights):
    """Picks the night where the Moon rises/sets during the night and clouds change."""
    best, best_k = -1, 0
    for k in range(n_nights):
        n = (start + k) % N_NIGHTS_YEAR
        d = eph.dark[n]
        fli_eff = np.where(eph.moon_alt[n] > 0, eph.fli[n], 0)[d]
        moon_frac = np.mean(fli_eff > 0)
        moon_mix = moon_frac * (1 - moon_frac) * eph.fli[n][d].mean()  # mixed moon
        cl = np.array([e["cloud"] for e in timeline if e["night"] == k])
        if len(cl) == 0 or np.mean(cl == 2) > 0.3:
            continue
        val = 4 * moon_mix + 0.05 * np.sum(np.diff(cl) != 0)
        if val > best:
            best, best_k = val, k
    return best_k


def plot_night(sv, eph, policies, start, seed, block, out):
    pcols = plt.cm.tab10(np.arange(len(sv.names) + 1))
    runs = {k: run_block(sv, eph, policies[k], start, block, seed, log=True)
            for k in ["Greedy priority", "Learned (CEM)"]}
    night = pick_interesting_night(eph, runs["Greedy priority"]["timeline"], start, block)
    fig, ax = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.2, 1.2]))
    for row, k in [(2, "Greedy priority"), (3, "Learned (CEM)")]:
        tl = [e for e in runs[k]["timeline"] if e["night"] == night]
        for e in tl:
            h = (e["slot"] * SLOT_MIN) / 60
            if e["kind"] == "idle":
                ax[row].bar(h, 1, SLOT_MIN / 60, align="edge", color="#eeeeee")
            elif not e["progs"]:
                ax[row].bar(h, 1, SLOT_MIN / 60, align="edge", color="k", alpha=0.6)
            else:
                m = len(e["progs"])
                for j, p in enumerate(e["progs"]):
                    ax[row].bar(h, 1 / m, SLOT_MIN / 60, bottom=j / m, align="edge",
                                color=pcols[p], edgecolor="w", lw=0.3)
            if np.isfinite(e["am"]):
                ax[row].plot(h + SLOT_MIN / 120, e["am"] - 1, "k.", ms=4)
        ax[row].set_ylim(0, 1)
        ax[row].set_ylabel(f"{k}\n(fibre share)", fontsize=9)
        if row == 2:
            tl_ref = tl
    hh = np.array([(e["slot"] * SLOT_MIN) / 60 + SLOT_MIN / 120 for e in tl_ref])
    ax[0].plot(hh, [e["seeing"] for e in tl_ref], "o-", c="#c44e52", ms=3)
    ax[0].set_ylabel("Zenith seeing [\"]")
    ax[0].set_title(f"One simulated night (day {(start + night) % N_NIGHTS_YEAR} of 2041); "
                    "black dots = airmass - 1 of chosen field")
    ax[1].step(hh, [e["cloud"] for e in tl_ref], where="mid", c="#4c72b0")
    ax[1].set_yticks([0, 1, 2], CLOUD_NAMES)
    ax[1].set_ylabel("Cloud", color="#4c72b0")
    ax2 = ax[1].twinx()
    ax2.plot(hh, [e["fli"] for e in tl_ref], c="#ccb974", lw=2)
    ax2.set_ylabel("Moon FLI (0 if below horizon)", color="#a08c2c", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax[3].set_xlabel("Hours since 21:00 UTC")
    handles = [Patch(color=pcols[p], label=n.split(" (")[0]) for p, n in enumerate(sv.names)]
    handles += [Patch(color=pcols[len(sv.names)], label="ToO"),
                Patch(color="k", alpha=0.6, label="failed (conditions changed)"),
                Patch(color="#eeeeee", label="idle")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, dpi=140)
    checkpoint(f"Timeline saved: {out} (night {night})")


if __name__ == "__main__":
    eph = Ephemeris()
    sv = Survey(seed=42)
    sv.calibrate_scarcity(eph, np.random.default_rng(1))
    w = np.array(json.load(open("learned_weights.json"))["w"])
    pols = {"Greedy priority": Policy("greedy", rng=np.random.default_rng(5)),
            "Learned (CEM)": Policy("learned", w=w)}
    plot_night(sv, eph, pols, start=60, seed=777, block=28,
               out="figures/fig3_one_night_timeline.png")
