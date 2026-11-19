# WST AI Scheduler — prototype (v1)

A toy simulation of a scheduler for the Wide-field Spectroscopic Telescope (WST) that
*learns* the decision "which pointing, with which exposure, should be taken now?". The goal is
to present the idea to the WST AI group in a concrete, testable form, and to provide a survey
simulator that the science working groups can use to check survey-time estimates.

v1 rebuilds the v0 toy model around the inputs in the WST cosmology paper (Payerne et al. 2026,
WP2.2.1), with illustrative assumptions for the Galactic, stellar and time-domain programmes.
v0 is kept in git (`git checkout v0`).

## Running

```bash
python run_experiment.py --quick   # 1-year survey, short training, ~15 min on 8 cores
python run_experiment.py           # 5-year survey, full training, ~50 min on 8 cores
python plot_timeline.py            # single-night figure only (needs learned_weights.json)
```

Requirements: numpy, astropy, matplotlib (nothing else). The first run computes 7 years of
Sun/Moon ephemeris (~1 min) and caches it in `cache/` (git-ignored). Runs are parallelised
over CPU cores; all randomness (weather, ToO alerts) is seeded per night, so every policy sees
exactly the same sky.

## Files

| File | Content |
|---|---|
| `programmes.py` | The programme pool, with the source of every number (all editable placeholders) |
| `wstsim.py` | Ephemeris, weather, tiling, fibre allocation, exposure model, simulator, policies, CEM |
| `run_experiment.py` | Validation, training, evaluation, figures, `results_summary.json` |
| `plot_timeline.py` | Single-night timeline figure |

## Model

| Component | What it does | Source |
|---|---|---|
| Time | Event-driven nights (Sun < −18°) over a 5-year survey starting 2041-01-01; exposures are 180 s (*short*, bright-time mode) or 1000 s (*long*), + 120 s fibre positioning + 1 s/deg slew | [C] Sec. 2.1 |
| Tiling | ~10,000 tiles of 3.1 deg² (Fibonacci sphere) south of δ=+30°; 5,805 tiles (18,000 deg²) in an LSST-WFD-like cosmology footprint | [C] Sec. 2.1 |
| Fibre pools | MOS-LR (30,000 fibres) and MOS-HR (2,000) shared by all programmes on the same focal plane, plus the IFS. Fibres are split between eligible programmes by priority-weighted share (capped by what each can use), leftovers by priority | [C] Sec. 2.1 |
| Fibre completeness | Poisson collisions: with F fibres on R remaining targets a tile completes R(1−e^(−F/R)) targets, so successive passes return less | [C] Eq. 37–39 |
| Exposure quality | Effective time t_eff = t_exp × speed, speed = transparency² × fibre-aperture loss² (seeing × X^0.6) × extinction² × (sky_ref/sky)^β. Sky includes airglow and moonlight (FLI, Moon altitude and separation). Credit ∝ t_eff / t_req; speed < 0.3 gives no usable spectra | — |
| Weather | Clouds: 3-state Markov chain (10-min steps), seeing: log-normal AR(1), 12% closed nights. The scheduler sees the previous step's conditions; true conditions decide the outcome | — |
| Programmes | 13 programmes (below) + ToO alerts, each with priority, footprint, target density, t_req, sky sensitivity β, airmass / seeing / Moon-distance limits, ESO time category, cadence and utility | `programmes.py` |
| Cadence | Visit programmes need N epochs separated by ≥ min_gap nights (dwarf-galaxy binarity, YSO variability) | [D] |
| IFS | The IFS deep-field programme sets the pointing; the MOS collects targets of that tile at the same time | — |
| Science utility | Per programme, normalised to 1 at completion. Cosmology: per-tile Fisher-like (c·nP/(1+c·nP))²; Galactic surveys: concave (1−e^(−c))/(1−e^(−1)); cadence/IFS: fraction of epochs done. Score = Σ priority × utility + ToO term | [C] Sec. 5.3 |

[C] Payerne et al. 2026 (WP2.2.1 cosmology); [D] illustrative assumptions (literature-inspired placeholders).

### Programme pool (hypothetical, illustrative numbers)

| Programme | Pool | Targets | t_req | Time cat. | Notes |
|---|---|---|---|---|---|
| Cosmo bright (BG r<20.74) | LR | 2,900 deg⁻² on WFD | 180 s | bright | [C] Table 1 |
| Cosmo grey (BG-Grey + LRG + ELG) | LR | 29,000 deg⁻² on WFD | 1000 s | grey | [C] Table 1 |
| Cosmo dark (QSO + LBG u/g/r) | LR | 17,300 deg⁻² on WFD | 2000 s | dark | [C] Table 1, faint LBGs need 1–3 passes |
| Halo chemo-kinematics | LR | 1,500 deg⁻², \|b\|>20 | 1000 s | grey | [D] |
| Disc/bulge MSTO | LR | 20,000 deg⁻², \|b\|<15 | 1000 s | grey | [D] |
| White dwarfs | LR | 10 deg⁻², all south | 1000 s | grey | [D] ~300,000 WDs |
| In-situ disc/bulge | HR | 1,000 (\|b\|<15) / 150 deg⁻² | 2000 s | grey | [D] ≥2 exposures per field |
| Halo chemical tagging | HR | 60 deg⁻², \|b\|>20 | 3000 s | dark | [D] G<18.5 |
| Exo hosts + n-capture | HR | 8 deg⁻², all south | 600 s | bright | [D] bright stars |
| Dwarf galaxies (HR) | HR | 500 per dwarf, 16 dwarfs | 4000 s × 3 epochs, ≥30 d apart | grey | [D] binarity |
| Dwarf galaxies (LR RVs) | LR | 3,000 per dwarf | 2000 s × 3 epochs, ≥30 d apart | dark | [D] |
| Young stars (time domain) | LR | 3,000 per tile, 40 regions | 1000 s × 4 epochs, ≥7 d apart | grey | [D] |
| IFS deep fields | IFS | 20 fields | 6000 s, seeing ≤ 0.8″ | dark | IFS drives the pointing |
| ToO alerts | — | 0.4 / night, 6 h window | 1000 s | — | LSST-like alerts |

## Policies

1. **Random**: random valid (tile, exposure) choice
2. **ESO time-category plan**: programmes and exposure length are fixed by the ESO time category
   of the moment (bright → 180 s and bright programmes; grey/dark → 1000 s and that category's
   programmes, falling back to all programmes only when nothing is available), and the tile is
   chosen by science gain per second. This is the plan assumed in [C] Sec. 6.1
3. **Greedy (gain rate)**: the (tile, exposure) with the largest expected gain in science score
   per second, given the fibre allocation and the last measured conditions. A strong myopic baseline
4. **Hand-tuned merit**: weighted sum of 10 features with hand-picked weights
5. **Learned (CEM)**: the same 10 features, weights learned with the Cross-Entropy Method on
   28-night blocks that start from early-, mid- and late-survey states (snapshots of a
   reference run), so the learner sees the whole survey

Features: science gain rate; scarcity × gain rate; lag × gain rate (favours programmes that are
behind); airmass; distance from tonight's best airmass; set urgency; slew; cadence gain rate;
ToO flag; short-exposure flag. The gain-type features are normalised to the best candidate.

For speed, when more than 600 tiles are visible a short list is pre-selected (the 200 best by
approximate gain, 200 best by priority × remaining work, 100 nearest to the last pointing,
100 closest to their best airmass tonight, plus tiles with an active ToO), and the full fibre
allocation is computed only for those.

## Results

Full run: 5-year survey (2041–2045), 2 unseen weather seeds (1 for the random and hand-tuned
baselines), CEM with 10 iterations × 12 candidates × 2 blocks. Max possible score = 11.3.

| Policy | Score | / greedy | % of max | Open shutter | Overheads | Wasted exp. | 180 s exp. | Mean X | ToO success |
|---|---|---|---|---|---|---|---|---|---|
| Random | 8.16 | 0.89 | 74% | 64% | 17% | 3.1% | 50% | 1.18 | 0% |
| ESO time-category plan | 7.34 | 0.80 | 66% | 64% | 17% | 3.1% | 65% | 1.18 | 24% |
| Greedy (gain rate) | 9.16 | 1.000 | 83% | 60% | 20% | 3.7% | 76% | 1.20 | 24% |
| Hand-tuned merit | 9.16 | 1.001 | 83% | 61% | 19% | 3.6% | 72% | 1.14 | 24% |
| **Learned (CEM)** | **9.16** | **1.000** | 83% | 62% | 19% | 3.6% | 72% | 1.15 | 24% |

Open shutter / overheads are fractions of astronomically dark time (the rest is closed nights
~11% and idle time ~9%).

**1. The learned scheduler does not beat the greedy gain-rate scheduler in v1.** Over the
whole survey it ties (1.000×, the seed-to-seed scatter is ±0.01). During training it was
0.96–1.10× greedy on individual 28-night blocks, but that did not carry over to the full
survey. v0's +4% came against a weaker greedy (priority only). In v1 the greedy already
optimises science gain per second, with the fibre allocation, airmass, slew and exposure time
built in. A linear merit function over myopic features has little left to add. Beating it
will need look-ahead (multi-night planning) or a stronger learner, not re-weighting.

**2. Same score, different survey.** The learned and greedy schedulers end with different
cosmology programmes:

| Cosmology programme after 5 yr | Greedy | Learned | ESO plan |
|---|---|---|---|
| Bright (BG) | 100% | 100% | 100% |
| Grey (BG-Grey + LRG + ELG) | 71% | 59% | 4.5% |
| Dark (QSO + LBG) | 24% | 33% | 68% |

The score treats these as equivalent. Choosing between them is a science decision for the
collaboration; the scheduler can't make it.

**3. A rigid ESO time-category plan costs ~20% of the science score.** Grey time is scarce:
- With this ephemeris, ESO grey (FLI 0.4–0.7 with the Moon up) is only **11%** of dark time.
  If the Moon's altitude is ignored (Moon phase only), it is 20%.
- The cosmology paper assumes 35%.

Programmes tied to grey time therefore starve: cosmo-grey reaches 4.5% and the disc LR survey
37% after 5 years. The flexible schedulers put grey programmes into dark time whenever that
pays off.

**4. Validation against the cosmology paper (Fig. 4, cosmology-only runs).** Years to reach
95% target completeness:

| Sub-survey | Analytic, paper's split (12 h nights, f_obs 0.8, 0.2/0.35/0.35) | Analytic, this ephemeris (9.2 h nights, 88% open, 0.33/0.11/0.56) | Simulated, ESO plan | Simulated, greedy |
|---|---|---|---|---|
| Bright | 0.7 | 0.5 | 1.4 | 0.6 |
| Grey | 5.9 | 22.5 | > 7 (≈35% at 7 yr) | 5.2 |
| Dark | 8.8 | 6.5 | > 7 (≈87% at 7 yr) | > 7 (≈50% at 7 yr) |

- **The analytic model reproduces the paper.** Run with the paper's assumptions, it gives
  ~6 years for the grey survey, the same as the paper.
- **The grey estimate moves from ~6 to ~22 years with realistic time categories.** With the
  real Moon statistics and night length (and the same model), the grey survey needs ~22
  years. The full simulation agrees with this corrected analytic curve (Fig. 4, left).
- **Dark takes longer than in the paper** because faint LBGs are given 2 × 1000 s here.
- **Weather, airmass limits and slews** slow the simulation further relative to the analytic
  curves.
- **A flexible scheduler can still finish grey in ~5 years** by using dark time, but then the
  dark (high-z) survey reaches only ~50% in 7 years.

The single time-category split behind the survey-time estimates is the assumption the WG
should revisit first.

**5. The equal-weight score saturates.**
- After 5 years, all the small Galactic and cadence programmes are ≳85–100% complete, even
  under the random policy. This is why random reaches 0.89× greedy.
- The large cosmology surveys (~900 million targets) decide the differences.
- Programme weights tied to real time allocations would make the comparison sharper.

**6. Exposure mix.**
- Greedy-type schedulers use 180 s exposures for ~3/4 of all exposures. A short exposure
  fully serves the bright programmes and gives partial credit to the others.
- The bright cosmology survey is finished in ~0.6 yr, but overheads rise to ~20% of dark time.

**What the scheduler learned (Fig. 2):** large weights on science gain, scarcity × gain
(spend rare conditions on the programmes that need them) and cadence gain; it penalises
distance from tonight's best airmass, slews and set urgency.

Figures:
- `figures/fig1_policy_comparison.png`: score over time, operational metrics, completeness per programme
- `figures/fig2_training_and_weights.png`: CEM training curve and learned weights
- `figures/fig3_one_night_timeline.png`: one night, greedy vs learned, from a mid-survey state
- `figures/fig4_cosmology_validation.png`: cosmology-only runs vs the Poisson pass model; cosmology inside the shared survey; dark/grey/bright split (paper's assumption vs this ephemeris)
- `figures/fig5_sky_maps.png`: footprints and final MOS-LR / MOS-HR completeness on the sky

## Limitations (honest list)

- Programme pool, densities, exposure times, weather and sky model are **hypothetical**.
  Numbers are for relative comparison.
- Fibre assignment is a Poisson approximation (no positioner geometry, no target clustering,
  no tile overlaps or dithers); targets are not tracked individually.
- A partial-depth exposure counts as a fraction of the fibre time ("fibre-time accounting"),
  not as a per-target signal-to-noise accumulation; the exposure model is not calibrated with
  the WST ETC (`pyetc_wst`).
- Equal normalisation of every programme's utility is a value judgement: it makes small or
  cheap programmes worth as much as the large cosmology surveys. Programme weights should come
  from the collaboration (e.g. time allocations per WP).
- Persistence is the only weather "forecast". No calibrations, twilight, engineering time.
- The learned model is linear (10 weights) and myopic; it is interpretable but limited.

## Next steps

1. Calibrate the exposure model with the WST ETC; replace placeholders with WG numbers
2. Agree programme weights / time allocations with the WPs; add fairness constraints
3. Per-target fibre assignment on real positioner geometry (joint scheduling + fibre assignment)
4. Look-ahead planning (multi-night) and deep RL (PPO/DQN); seeing/cloud nowcasting
5. Use the simulator to revisit the cosmology survey-time estimates with realistic time categories
