# WST AI Scheduler — prototype

A toy simulation of a scheduler for the Wide-field Spectroscopic Telescope (WST) that
*learns* the decision "which observation should be taken now?". The goal is to present
the idea to the WST AI group in a concrete, testable form.

## Running

```bash
python run_experiment.py --quick   # ~30 s, quick trial
python run_experiment.py           # ~4 min, full run (25 CEM iterations, 30 test blocks)
python plot_timeline.py            # single-night figure only (needs learned_weights.json)
```

Requirements: numpy, astropy, matplotlib (nothing else).

## Model

| Component | What it does |
|---|---|
| Ephemeris | Year 2041, 20-min slots; Sun/Moon positions, lunar FLI, field altitudes, airmass, Moon separation. Site: Paranal coordinates (placeholder) |
| Weather | Clouds: 3-state Markov chain (photometric/thin/thick); seeing: log-normal AR(1); 8% closed nights. Seeing scales with airmass as X^0.6 |
| Programmes | 6 hypothetical programmes (COSMO-LBG, COSMO-BG, GAL-HALO, GAL-DISC, DWARFS & MCs, IFS DEEP) + ToO alerts. Each has a priority, max airmass, max FLI, min Moon separation, max seeing, cloud tolerance, field list and number of visits |
| Shared focal plane | When a field is observed, **all** programmes whose constraints are met at that moment receive visit credit (WST's simultaneous MOS-LR + MOS-HR + IFS operation) |
| Uncertainty | The scheduler sees the previous slot's conditions; if the true conditions change, the exposure is wasted |
| Overheads | 120 s fibre positioning + 1 s/deg slew |

## Policies

1. **Random**: random choice among valid fields
2. **Greedy priority**: the field with the highest summed priority of programmes it would credit right now
3. **Hand-tuned merit**: weighted sum of 9 features with hand-picked weights (the logic of Rubin's Feature-Based Scheduler)
4. **Learned (CEM)**: the same 9 features, but the weights are **learned** over thousands of
   simulated nights with the Cross-Entropy Method

Features: priority, priority × constraint scarcity, airmass, distance from tonight's best airmass,
set urgency, programme lag, slew distance, seeing margin, ToO flag.

## Results (30 unseen 28-night blocks spread over the year)

| Policy | Science score / greedy | Beats greedy | Wasted exposures | Mean airmass | ToO success |
|---|---|---|---|---|---|
| Random | 0.77 ± 0.05 | 0% | 7.4% | 1.30 | 13% |
| Greedy priority | 1.00 | — | 8.8% | 1.29 | 36% |
| Hand-tuned merit | 1.00 ± 0.02 | 47% | 7.2% | 1.21 | 39% |
| **Learned (CEM)** | **1.04 ± 0.02** | **100%** | 8.6% | 1.27 | 38% |

- The learning scheduler beat greedy in every test block, by ~4% on average.
- The gain comes mostly from programmes that need rare conditions: IFS deep fields go from 12% → 27%,
  dwarf galaxies/MCs from 33% → 43%. The model put a large weight on the "priority × scarcity" feature,
  i.e. it discovered on its own the rule "good seeing / dark time is rare; when it comes, spend it on
  the programme that needs it most".
- The hand-tuned merit function is no better than greedy. This shows that tuning the weights by hand
  is hard and that learning them is valuable.

Figures: `figures/fig1_policy_comparison.png`, `fig2_training_and_weights.png`,
`fig3_one_night_timeline.png`.

## Limitations (honest list)

- The programme pool, weather model and costs are **hypothetical**. The numbers are for relative
  comparison, not absolute predictions.
- ~500 fields are used (in reality ~6000+ pointings); fibre assignment (which fibre goes to which target) is not modelled.
- No weather forecasting (persistence only). A real system would use seeing/cloud forecasts and all-sky camera input.
- Most ToOs are never observable within their 6-hour window, which is why ToO success looks low.
- Idle dark time is higher for the learned model than for greedy (30.5% vs 25.9%). Likely cause:
  fields usable in bright time run out, i.e. the programme pool is not oversubscribed in all conditions.
  This has not been verified yet.
- The learned model is linear (9 weights). It is interpretable but has limited capacity.

## Next steps

1. Real WST inputs: ETC (pyetc_wst), programme list from the WST science cases, real site weather statistics
2. Deep reinforcement learning (PPO/DQN) and multi-night planning (lookahead)
3. Weather forecasting module (ML seeing/cloud nowcasting)
4. Joint optimisation of the scheduler with fibre assignment (WST-specific, shared focal plane)
5. Programme fairness constraints: no programme should be starved
