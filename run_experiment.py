"""
WST scheduler experiment (v1):
  1) Build the ephemeris, tiling, footprints and programme pool
  2) Cosmology-only validation runs vs the analytic Poisson pass model (Payerne et al. 2026)
  3) Reference multi-year greedy run -> survey-state snapshots for training
  4) Train the merit weights with CEM on 28-night blocks starting from those snapshots
  5) Compare 5 policies over the full survey on unseen weather
  6) Produce the figures, results_summary.json and learned_weights.json

Usage:  python run_experiment.py            (full run: 5-year survey, ~50 min on 8 cores)
        python run_experiment.py --quick    (1-year survey, short training, ~15 min)
"""
import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wstsim import (Ephemeris, Survey, Policy, run_many, make_pool, train_cem, checkpoint,
                    poisson_pass_model, time_category_fractions, FEATURE_NAMES, HANDTUNED_W,
                    CAT_NAMES, SURVEY_NIGHTS, CLOSED_NIGHT_P, _job)
from programmes import COSMO_ONLY, FOV_DEG2
from plot_timeline import plot_night, pick_interesting_night

QUICK = "--quick" in sys.argv
YEARS = 1 if QUICK else 5
N_NIGHTS = int(round(365.25 * YEARS))
VAL_YEARS = 2 if QUICK else 7
BLOCK = 28
TRAIN_SEED = 101
TEST_SEEDS = [900001] if QUICK else [900001, 900002]
# policies evaluated on every test seed; the others (cheap baselines) on the first one only,
# so that the full evaluation fits in one round of 8 worker processes
ALL_SEEDS = {"greedy", "learned", "eso"}
SNAP_EVERY = 30 if QUICK else 61
CEM = dict(n_iter=4, pop=8, n_eval=2) if QUICK else dict(n_iter=10, pop=12, n_eval=2)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

POLICY_NAMES = {"random": "Random", "eso": "ESO time-category plan", "greedy": "Greedy (gain rate)",
                "handtuned": "Hand-tuned merit", "learned": "Learned (CEM)"}
COLS = {"random": "#9e9e9e", "eso": "#8172b3", "greedy": "#4c72b0",
        "handtuned": "#dd8452", "learned": "#55a868"}


def plot_cosmology_validation(sv, val, res1, cat_frac, phase_frac, dark_h_yr, n_nights, val_years,
                              out):
    """Fig. 4. val: cosmology-only runs {"eso", "greedy"}; res1: full-survey runs
    {"greedy", "learned"} (one seed each)."""
    from matplotlib.lines import Line2D
    years_axis = lambda r: r["hist_n"] / 365.25  # noqa: E731
    cosmo_idx = [sv.names.index(n) for n in COSMO_ONLY]
    n_wfd = int(sv.nfields[cosmo_idx[0]])
    f_paper = {"COSMO-BRIGHT": 0.2, "COSMO-GREY": 0.35, "COSMO-DARK": 0.35}
    f_eph = {"COSMO-BRIGHT": cat_frac[2], "COSMO-GREY": cat_frac[1], "COSMO-DARK": cat_frac[0]}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8]))
    ccol = {"COSMO-BRIGHT": "#e1a100", "COSMO-GREY": "#7f7f7f", "COSMO-DARK": "#1f3b73"}
    for p in cosmo_idx:
        nm = sv.names[p]
        spec = sv.progs[p]
        y1, c1 = poisson_pass_model(spec["density"], spec["t_req"], n_wfd, f_paper[nm])
        y2, c2 = poisson_pass_model(spec["density"], spec["t_req"], n_wfd, f_eph[nm],
                                    f_obs=1 - CLOSED_NIGHT_P, hours_per_night=dark_h_yr / 365.25)
        ax[0].plot(y1, c1 * 100, "D:", c=ccol[nm], ms=3, lw=1)
        ax[0].plot(y2, c2 * 100, "o--", c=ccol[nm], ms=3, lw=1)
        r = val["eso"]
        ax[0].plot(years_axis(r), r["hist_C"][:, p] * 100, "-", c=ccol[nm], lw=2.2, label=nm)
        r = val["greedy"]
        ax[0].plot(years_axis(r), r["hist_C"][:, p] * 100, "-", c=ccol[nm], lw=1, alpha=0.6)
    ax[0].axhline(95, c="k", lw=0.6, ls=":")
    ax[0].set_xlim(0, val_years)
    ax[0].set_ylim(0, 102)
    ax[0].set_xlabel("Years since survey start")
    ax[0].set_ylabel("Target completeness [%]")
    ax[0].set_title("Cosmology only (all LR fibres): simulation vs Poisson pass model")
    extra = [Line2D([], [], c="k", lw=2.2, label="simulated, ESO time-category plan"),
             Line2D([], [], c="k", lw=1, alpha=0.6, label="simulated, greedy gain rate"),
             Line2D([], [], c="k", ls=":", marker="D", ms=3, lw=1,
                    label="analytic, paper's time split (0.2/0.35/0.35)"),
             Line2D([], [], c="k", ls="--", marker="o", ms=3, lw=1,
                    label="analytic, this ephemeris' time split")]
    h, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles=h + extra, fontsize=7, loc="lower right")
    for k in ["greedy", "learned"]:
        r = res1[k]
        for p in cosmo_idx:
            ax[1].plot(years_axis(r), r["hist_C"][:, p] * 100, c=ccol[sv.names[p]],
                       lw=2.2 if k == "learned" else 1, alpha=1 if k == "learned" else 0.6,
                       label=sv.names[p] if k == "learned" else None)
    for p in cosmo_idx:
        r = val["eso"]
        m = r["hist_n"] <= n_nights
        ax[1].plot(years_axis(r)[m], r["hist_C"][m, p] * 100, ":", c=ccol[sv.names[p]], lw=1)
    ax[1].set_xlabel("Years since survey start")
    ax[1].set_ylabel("Target completeness [%]")
    ax[1].set_title("Cosmology inside the shared survey\n(thick: learned, thin: greedy, "
                    "dotted: cosmology-only)")
    ax[1].legend(fontsize=8)
    splits = {"Payerne+26\nassumption": [0.35, 0.35, 0.20],
              "This ephemeris,\nMoon phase only": list(phase_frac),
              "This ephemeris,\nMoon down = dark": list(cat_frac)}
    bottom = np.zeros(len(splits))
    for j, (cname, col) in enumerate(zip(CAT_NAMES, ["#1f3b73", "#7f7f7f", "#e1a100"])):
        v = np.array([s[j] for s in splits.values()])
        ax[2].bar(range(len(splits)), v * 100, bottom=bottom * 100, color=col, label=cname)
        for x, (vv, bb) in enumerate(zip(v, bottom)):
            ax[2].text(x, (bb + vv / 2) * 100, f"{vv:.0%}", ha="center", va="center",
                       color="w", fontsize=8)
        bottom += v
    ax[2].set_xticks(range(len(splits)), list(splits), fontsize=8)
    ax[2].set_ylabel("% of observing time")
    ax[2].set_title("Dark / grey / bright split (2041-45)")
    ax[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def elapsed():
    return f"[{(time.time() - T0) / 60:.1f} min]"


if __name__ == "__main__":
    eph = Ephemeris()
    sv = Survey(seed=42)
    P_ok = sv.calibrate_scarcity(eph)
    checkpoint(f"Tiles: {sv.n_tiles} x {FOV_DEG2} deg^2; WFD area {sv.area['wfd']:.0f} deg^2 "
               f"({sv.nfields[0]} tiles)")
    checkpoint("P(conditions good enough): " +
               ", ".join(f"{n}={p:.2f}" for n, p in zip(sv.names, P_ok)))
    cat_frac, dark_h_yr = time_category_fractions(eph)
    phase_frac, _ = time_category_fractions(eph, moon_altitude=False)
    checkpoint(f"Dark time split (this ephemeris): dark={cat_frac[0]:.2f}, grey={cat_frac[1]:.2f}, "
               f"bright={cat_frac[2]:.2f}; {dark_h_yr:.0f} h/yr astronomically dark")
    cosmo_mask = np.isin(sv.names, COSMO_ONLY)
    pool = make_pool(sv, eph)

    # ------------------------------------------------ stage 1: validation + reference run
    val_n = int(round(365.25 * VAL_YEARS))
    rec = 7
    val_jobs = {
        k: pool.submit(_job, (Policy(k), 0, val_n, TEST_SEEDS[0], None,
                              dict(prog_subset=cosmo_mask, record_every=rec)))
        for k in ["eso", "greedy"]
    }
    ref_fut = pool.submit(_job, (Policy("greedy"), 0, N_NIGHTS, TRAIN_SEED, None,
                                 dict(snapshot_every=SNAP_EVERY, record_every=rec)))
    checkpoint(f"{elapsed()} Reference run ({YEARS} yr) and cosmology-only validation "
               f"({VAL_YEARS} yr) submitted")
    ref = ref_fut.result()
    snapshots = [(0, sv.new_state())] + [s for s in ref["snaps"] if s[0] + BLOCK < N_NIGHTS]
    checkpoint(f"{elapsed()} Reference run done: {len(snapshots)} snapshots")

    # ------------------------------------------------ stage 2: training
    checkpoint(f"{elapsed()} Starting CEM training...")
    w_learned, hist = train_cem(pool, snapshots, block_nights=BLOCK, seed=0, **CEM)
    json.dump(dict(features=FEATURE_NAMES, w=w_learned.tolist()),
              open("learned_weights.json", "w"), indent=2)
    checkpoint(f"{elapsed()} Learned weights written to learned_weights.json")

    # ------------------------------------------------ stage 3: evaluation
    policies = {k: Policy(k) for k in ["random", "eso", "greedy", "handtuned"]}
    policies["learned"] = Policy("learned", w=w_learned)
    jobs, keys = [], []
    for sd in TEST_SEEDS:
        for k, pol in policies.items():
            if sd != TEST_SEEDS[0] and k not in ALL_SEEDS:
                continue
            jobs.append((pol, 0, N_NIGHTS, sd, None, dict(record_every=rec, keep_state=True)))
            keys.append((k, sd))
    checkpoint(f"{elapsed()} Test: {len(TEST_SEEDS)} weather seed(s) x {YEARS} yr x "
               f"{len(policies)} policies")
    res_list = run_many(pool, jobs)
    res = {k: [] for k in policies}
    for (k, sd), r in zip(keys, res_list):
        res[k].append(r)
    val = {k: f.result() for k, f in val_jobs.items()}
    pool.shutdown()
    checkpoint(f"{elapsed()} Test finished")

    # ------------------------------------------------ summary
    summary = dict(config=dict(years=YEARS, test_seeds=TEST_SEEDS, block_nights=BLOCK, cem=CEM,
                               n_tiles=int(sv.n_tiles), wfd_area_deg2=float(sv.area["wfd"]),
                               time_category_fraction_of_dark=dict(zip(CAT_NAMES, cat_frac.tolist())),
                               time_category_fraction_moon_phase_only=dict(zip(CAT_NAMES, phase_frac.tolist())),
                               dark_hours_per_year=float(dark_h_yr)),
                   programmes=sv.names, policies={})
    ref_score = {r["seed"]: r["score"] for r in res["greedy"]}
    for k, rs in res.items():
        sc = np.array([r["score"] for r in rs])
        rel = np.array([r["score"] / ref_score[r["seed"]] for r in rs])
        A = [r["acc"] for r in rs]
        summary["policies"][POLICY_NAMES[k]] = dict(
            score=float(sc.mean()), score_std=float(sc.std()),
            score_rel_greedy=float(rel.mean()), n_seeds=len(rs),
            score_frac_of_max=float(sc.mean() / sv.max_score()),
            utility=np.mean([r["hist_U"][-1] for r in rs], axis=0).tolist(),
            completeness=np.mean([r["hist_C"][-1] for r in rs], axis=0).tolist(),
            open_frac=float(np.mean([a["open_s"] / a["dark_s"] for a in A])),
            overhead_frac=float(np.mean([a["overhead_s"] / a["dark_s"] for a in A])),
            idle_frac=float(np.mean([a["idle_s"] / a["dark_s"] for a in A])),
            closed_frac=float(np.mean([a["closed_s"] / a["dark_s"] for a in A])),
            wasted_frac=float(np.mean([a["wasted_s"] / max(a["open_s"], 1) for a in A])),
            short_exp_frac=float(np.mean([a["n_short"] / max(a["n_exp"], 1) for a in A])),
            mean_airmass=float(np.mean([a["am_sum"] / max(a["am_n"], 1) for a in A])),
            too_success=float(np.sum([r["too_done"] for r in rs]) /
                              max(np.sum([r["too_total"] for r in rs]), 1)),
        )
    for k, s in summary["policies"].items():
        checkpoint(f"{k:20s} score={s['score']:.3f} ({s['score_rel_greedy']:.3f}x greedy, "
                   f"{s['score_frac_of_max']:.0%} of max) | open={s['open_frac']:.0%} "
                   f"ovh={s['overhead_frac']:.0%} idle={s['idle_frac']:.0%} "
                   f"wasted={s['wasted_frac']:.1%} short={s['short_exp_frac']:.0%} "
                   f"X={s['mean_airmass']:.2f} ToO={s['too_success']:.0%}")

    # cosmology validation numbers
    cosmo_idx = [sv.names.index(n) for n in COSMO_ONLY]
    n_wfd = int(sv.nfields[cosmo_idx[0]])
    f_paper = {"COSMO-BRIGHT": 0.2, "COSMO-GREY": 0.35, "COSMO-DARK": 0.35}
    f_eph = {"COSMO-BRIGHT": cat_frac[2], "COSMO-GREY": cat_frac[1], "COSMO-DARK": cat_frac[0]}
    open_frac_eph = 1 - 0.12
    h_night_eph = dark_h_yr / 365.25

    def t95(years, comp):
        i = np.where(np.asarray(comp) >= 0.95)[0]
        return float(years[i[0]]) if len(i) else None

    vt = {}
    for p in cosmo_idx:
        nm = sv.names[p]
        spec = sv.progs[p]
        y1, c1 = poisson_pass_model(spec["density"], spec["t_req"], n_wfd, f_paper[nm])
        y2, c2 = poisson_pass_model(spec["density"], spec["t_req"], n_wfd, f_eph[nm],
                                    f_obs=open_frac_eph, hours_per_night=h_night_eph)
        vt[nm] = dict(analytic_paper_assumptions=t95(y1, c1), analytic_this_ephemeris=t95(y2, c2))
        for k, r in val.items():
            vt[nm][f"simulated_{k}"] = t95(r["hist_n"] / 365.25, r["hist_C"][:, p])
    summary["cosmology_validation_years_to_95pc"] = vt
    for nm, v in vt.items():
        checkpoint(f"Years to 95% completeness, {nm}: " +
                   ", ".join(f"{k}={'>' + str(VAL_YEARS) if x is None else f'{x:.1f}'}"
                             for k, x in v.items()))
    summary["training_history"] = hist
    json.dump(summary, open("results_summary.json", "w"), indent=2)

    # ------------------------------------------------ figures
    names = list(policies)
    years_axis = lambda r: r["hist_n"] / 365.25  # noqa: E731

    # Fig 1: policy comparison
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    for k in names:
        for j, r in enumerate(res[k]):
            ax.plot(years_axis(r), r["hist_score"], c=COLS[k], lw=1.6,
                    label=POLICY_NAMES[k] if j == 0 else None)
    ax.set_xlabel("Years since survey start")
    ax.set_ylabel("Science score  (sum of priority x utility + ToO)")
    ax.set_title(f"Science return over the survey ({len(TEST_SEEDS)} unseen weather seed(s))")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    gref = res["greedy"][0]
    for k in names:
        r = res[k][0]
        ax.plot(years_axis(r), r["hist_score"] / np.maximum(gref["hist_score"], 1e-9),
                c=COLS[k], lw=1.6)
    ax.axhline(1, ls="--", c="k", lw=0.8)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Years since survey start")
    ax.set_ylabel("Score / greedy (same weather)")
    ax.set_title("Relative to the greedy gain-rate scheduler")
    ax = fig.add_subplot(gs[0, 2])
    metrics = [("open_frac", "Open shutter"), ("overhead_frac", "Overheads"),
               ("idle_frac", "Idle"), ("closed_frac", "Closed (weather)"),
               ("wasted_frac", "Wasted exposures*"), ("short_exp_frac", "180 s exposures*"),
               ("too_success", "ToO success*")]
    xm = np.arange(len(metrics))
    bw = 0.16
    for i, k in enumerate(names):
        s = summary["policies"][POLICY_NAMES[k]]
        ax.bar(xm + (i - 2) * bw, [s[m] * 100 for m, _ in metrics], bw, color=COLS[k])
    ax.set_xticks(xm, [m[1] for m in metrics], rotation=30, ha="right")
    ax.set_ylabel("% of dark time (* = % of exposures / alerts)")
    ax.set_title("Operational metrics")
    ax = fig.add_subplot(gs[1, :])
    xp = np.arange(len(sv.names))
    for i, k in enumerate(names):
        s = summary["policies"][POLICY_NAMES[k]]
        ax.bar(xp + (i - 2) * bw, np.array(s["completeness"]) * 100, bw, color=COLS[k],
               label=POLICY_NAMES[k])
    ax.set_xticks(xp, sv.labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"Completeness after {YEARS} yr [%]\n(targets observed, or epochs done)")
    ax.set_title("Programme completeness (fibre pools shared on the same focal plane)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_policy_comparison.png", dpi=130)
    plt.close(fig)

    # Fig 2: training curve + learned weights
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    it = [h["it"] + 1 for h in hist]
    ax[0].plot(it, [h["best"] for h in hist], "o-", c="#55a868", label="best in population")
    ax[0].plot(it, [h["mean"] for h in hist], "s-", c="#8172b3", label="population mean")
    ax[0].plot(it, [h["mu_fit"] for h in hist], "^-", c="k", label="current mean policy")
    ax[0].axhline(1, ls="--", c="#4c72b0", label="greedy (same blocks)")
    ax[0].set_xlabel("CEM iteration")
    ax[0].set_ylabel("Score gain over a 28-night block / greedy")
    ax[0].set_title("Training (blocks start from early-, mid- and late-survey states)")
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
    plt.close(fig)

    # Fig 3: one night, greedy vs learned, from a mid-survey state
    mid = snapshots[len(snapshots) // 2]
    night = pick_interesting_night(eph, mid[0], BLOCK, TEST_SEEDS[0])
    plot_night(sv, eph, {"Greedy (gain rate)": Policy("greedy"), "Learned (CEM)": policies["learned"]},
               night, TEST_SEEDS[0], f"{OUT}/fig3_one_night_timeline.png", state=mid[1])

    # Fig 4: cosmology validation (Fig. 14 of Payerne et al. 2026)
    plot_cosmology_validation(sv, val, {k: res[k][0] for k in ["greedy", "learned"]},
                              cat_frac, phase_frac, dark_h_yr, N_NIGHTS, VAL_YEARS,
                              f"{OUT}/fig4_cosmology_validation.png")

    # Fig 5: footprints and final completeness on the sky
    def moll(ax, vals, cmap, title, vmin=None, vmax=None, s=4):
        x = np.radians(-((sv.ra + 180) % 360 - 180))
        sc = ax.scatter(x, np.radians(sv.dec), c=vals, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                        linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels([])
        ax.grid(alpha=0.3)
        return sc
    fig = plt.figure(figsize=(17, 5.2))
    fp_code = np.full(sv.n_tiles, np.nan)
    fp_code[sv.footprints["south"]] = 0
    fp_code[sv.footprints["halo"] & ~sv.footprints["wfd"]] = 1
    fp_code[sv.footprints["wfd"]] = 2
    fp_code[sv.footprints["disc"]] = 3
    ax = fig.add_subplot(1, 3, 1, projection="mollweide")
    fp_cols = ["#c7c7c7", "#9ecae1", "#fdd49e", "#a1d99b"]
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    moll(ax, fp_code, ListedColormap(fp_cols), "Programme footprints", -0.5, 3.5)
    fp_handles = [Patch(color=c, label=lab) for c, lab in zip(
        fp_cols, ["all-sky HR / WDs", "halo (|b|>20)", "cosmology (LSST-WFD-like)", "disc (|b|<15)"])]
    for key, mk, col in [("dwarfs", "*", "r"), ("sfr", "^", "m"), ("ifs", "s", "k")]:
        m = sv.footprints[key]
        ax.scatter(np.radians(-((sv.ra[m] + 180) % 360 - 180)), np.radians(sv.dec[m]),
                   marker=mk, c=col, s=25)
        fp_handles.append(plt.Line2D([], [], ls="", marker=mk, c=col,
                                     label={"dwarfs": "dwarf galaxies", "sfr": "star-forming regions",
                                            "ifs": "IFS deep fields"}[key]))
    ax.legend(handles=fp_handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    stL = res["learned"][0]["state"]
    lr = [p for p in range(len(sv.names)) if sv.progs[p]["pool"] == "LR" and not sv.visit[p]]
    hr = [p for p in range(len(sv.names)) if sv.progs[p]["pool"] == "HR" and not sv.visit[p]]
    for j, (idx, title) in enumerate([(lr, "MOS-LR"), (hr, "MOS-HR")]):
        N = sv.N0[idx].sum(axis=0)
        comp = np.where(N > 0, 1 - stL.R[idx].sum(axis=0) / np.maximum(N, 1), np.nan)
        ax = fig.add_subplot(1, 3, 2 + j, projection="mollweide")
        sc = moll(ax, comp * 100, "viridis",
                  f"{title} target completeness after {YEARS} yr (learned) [%]", 0, 100)
        fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_sky_maps.png", dpi=130)
    plt.close(fig)
    checkpoint(f"{elapsed()} Figures saved to {OUT}/")
