"""
Programme definitions for the WST scheduling toy model (v1).

Every number here is a HYPOTHETICAL placeholder distilled from WST documents:
  [C]  Payerne et al. (2026), "Targeting the future of spectroscopic surveys with a
       WST case cosmological study" (WP2.2.1), Table 1, Sec. 2.1, 5.2, 6.1
  [D]  illustrative assumptions for Galactic, stellar and time-domain programmes
       (literature-inspired placeholders)
Treat every value as a tunable parameter.

Field meanings
  pool       fibre pool: "LR" (MOS-LR, 30,000 fibres), "HR" (MOS-HR, 2,000 fibres), "IFS"
  kind       "survey": many targets per tile; fibres are assigned with Poisson collisions and
                       each exposure completes new targets (multi-pass completeness, [C] Sec. 5.2)
             "visit":  a fixed set of targets per tile that is re-observed in epochs (cadence)
  density    targets / deg^2 (survey) or targets per tile (visit)
  footprint  key into Survey.footprints (see wstsim.py)
  t_req      effective open-shutter time [s] needed per target (or per epoch) at the
             reference sky brightness `sky_ref` and nominal seeing/transparency
  beta       sky sensitivity of the exposure time: t ~ sky^beta
             (1 = faint, background-limited target; ~0.2 = bright, source-limited star)
  sky_ref    relative sky brightness at which t_req is defined (1 = dark, 2.5 = grey, 6 = bright)
  time_cat   ESO time category the programme is assigned to in the rule-based baseline
             ("bright": FLI>0.7, "grey": 0.4-0.7, "dark": FLI<0.4 or Moon down)
  utility    ("fisher", nP): per-tile Fisher-like information (c nP / (1 + c nP))^2,
                             nP = n_bar * P at full completeness (cosmology tracers)
             ("concave", k): diminishing returns (1 - exp(-k c)) / (1 - exp(-k))
             ("linear",):    proportional to completed epochs
  n_epochs, min_gap  cadence: number of epochs and minimum separation in nights
"""

FOV_DEG2 = 3.1          # [C] Sec. 2.1
FIBRES = {"LR": 30000, "HR": 2000, "IFS": 1}   # [C] Sec. 2.1

PROGRAMMES = [
    # ------------------------------------------------------------------ cosmology [C]
    dict(name="COSMO-BRIGHT", label="Cosmo bright (BG r<20.74)", group="cosmo",
         pool="LR", kind="survey", prio=0.6, footprint="wfd", density=2900,
         t_req=180, beta=0.6, sky_ref=6.0, time_cat="bright",
         max_am=1.8, max_seeing=2.0, min_msep=10, utility=("fisher", 3.0),
         source="[C] Table 1: BG Bright 2,900 deg^-2, 180 s bright-time exposures"),
    dict(name="COSMO-GREY", label="Cosmo grey (BG-Grey+LRG+ELG)", group="cosmo",
         pool="LR", kind="survey", prio=0.8, footprint="wfd", density=29000,
         t_req=1000, beta=1.0, sky_ref=2.5, time_cat="grey",
         max_am=1.6, max_seeing=1.6, min_msep=20, utility=("fisher", 2.0),
         source="[C] Table 1: 4,300 + 2,500 + 22,200 deg^-2, 1000 s in ESO grey time"),
    dict(name="COSMO-DARK", label="Cosmo dark (QSO+LBG)", group="cosmo",
         pool="LR", kind="survey", prio=1.0, footprint="wfd", density=17300,
         t_req=2000, beta=1.0, sky_ref=1.0, time_cat="dark",
         max_am=1.5, max_seeing=1.4, min_msep=30, utility=("fisher", 1.0),
         source="[C] Table 1: QSO 300 + LBG u/g/r 8,000/8,000/1,000 deg^-2; faint LBGs "
                "need 1-3 passes (Sec. 5.1), here 2 x 1000 s on average"),
    # ------------------------------------------------------------------ Galactic, LR
    dict(name="GAL-HALO-LR", label="Halo chemo-kinematics (LR)", group="galactic",
         pool="LR", kind="survey", prio=0.7, footprint="halo", density=1500,
         t_req=1000, beta=0.8, sky_ref=2.5, time_cat="grey",
         max_am=1.7, max_seeing=1.6, min_msep=20, utility=("concave", 1.0),
         source="[D] Gaia/LSST counts: ~500 deg^-2 at G~20 to ~2,000 deg^-2 at G~24"),
    dict(name="GAL-DISC-LR", label="Disc/bulge MSTO (LR)", group="galactic",
         pool="LR", kind="survey", prio=0.7, footprint="disc", density=20000,
         t_req=1000, beta=0.7, sky_ref=2.5, time_cat="grey",
         max_am=1.7, max_seeing=1.6, min_msep=20, utility=("concave", 1.0),
         source="[D] inner Galaxy LR 18.5<G<20.5: 5e3-7.5e4 deg^-2"),
    dict(name="WD-LR", label="White dwarfs (LR)", group="galactic",
         pool="LR", kind="survey", prio=0.5, footprint="south", density=10,
         t_req=1000, beta=0.8, sky_ref=2.5, time_cat="grey",
         max_am=1.7, max_seeing=1.6, min_msep=20, utility=("concave", 1.0),
         source="[D] ~300,000 WDs, V=16-23, over ~30,000 deg^2"),
    # ------------------------------------------------------------------ Galactic, HR
    dict(name="INSITU-HR", label="In-situ disc/bulge (HR)", group="galactic",
         pool="HR", kind="survey", prio=0.8, footprint="insitu", density=None,
         t_req=2000, beta=0.5, sky_ref=2.5, time_cat="grey",
         max_am=1.7, max_seeing=1.5, min_msep=20, utility=("concave", 1.0),
         source="[D] 1e7 targets G<21 over 30,000 deg^2, >=2 exposures per field; "
                "here 1,000 deg^-2 at |b|<15 and 150 deg^-2 elsewhere"),
    dict(name="HALO-HR", label="Halo chemical tagging (HR)", group="galactic",
         pool="HR", kind="survey", prio=0.8, footprint="halo", density=60,
         t_req=3000, beta=0.7, sky_ref=2.5, time_cat="dark",
         max_am=1.6, max_seeing=1.3, min_msep=25, utility=("concave", 1.0),
         source="[D] HR survey of the outer halo, G<18.5"),
    dict(name="HR-BRIGHT", label="Exo hosts + n-capture (HR)", group="galactic",
         pool="HR", kind="survey", prio=0.6, footprint="south", density=8,
         t_req=600, beta=0.2, sky_ref=6.0, time_cat="bright",
         max_am=1.8, max_seeing=2.0, min_msep=10, utility=("concave", 1.0),
         source="[D] ~113,000 planet hosts (most G<13); >1e5 stars G<15, S/N>100"),
    # ------------------------------------------------------------------ cadence / time domain
    dict(name="DWARFS-HR", label="Dwarf galaxies (HR, cadence)", group="cadence",
         pool="HR", kind="visit", prio=0.9, footprint="dwarfs", density=500,
         t_req=4000, beta=0.8, sky_ref=2.5, time_cat="grey",
         max_am=1.6, max_seeing=1.2, min_msep=30, utility=("linear",),
         n_epochs=3, min_gap=30,
         source="[D] >=100 stars per dSph, >=10 dSphs, G<18, S/N 25-30 per line; "
                "multi-epoch cadence for binarity"),
    dict(name="DWARFS-LR", label="Dwarf galaxies (LR RVs, cadence)", group="cadence",
         pool="LR", kind="visit", prio=0.9, footprint="dwarfs", density=3000,
         t_req=2000, beta=1.0, sky_ref=2.5, time_cat="dark",
         max_am=1.6, max_seeing=1.3, min_msep=30, utility=("linear",),
         n_epochs=3, min_gap=30,
         source="[D] LR radial velocities/metallicities of dwarf members, multi-epoch"),
    dict(name="YSO-TD", label="Young stars (time domain)", group="cadence",
         pool="LR", kind="visit", prio=0.8, footprint="sfr", density=3000,
         t_req=1000, beta=0.7, sky_ref=2.5, time_cat="grey",
         max_am=1.7, max_seeing=1.6, min_msep=20, utility=("linear",),
         n_epochs=4, min_gap=7,
         source="[D] multi-epoch YSO spectroscopy: daily, weekly-monthly and annual snapshots"),
    # ------------------------------------------------------------------ IFS drives the pointing
    dict(name="IFS-DEEP", label="IFS deep fields", group="ifs",
         pool="IFS", kind="visit", prio=1.0, footprint="ifs", density=1,
         t_req=6000, beta=1.0, sky_ref=1.0, time_cat="dark",
         max_am=1.4, max_seeing=0.8, min_msep=50, utility=("linear",),
         n_epochs=1, min_gap=0,
         source="IFS deep fields: the IFS sets the pointing, the MOS rides along"),
]

# Targets-of-opportunity (LSST-like alerts), observed with the long mode on the
# tile that contains them; the MOS collects targets of that tile at the same time.
TOO = dict(weight=1.0, rate_per_night=0.4, window_h=6.0, max_am=1.8, max_seeing=1.5)

COSMO_ONLY = ["COSMO-BRIGHT", "COSMO-GREY", "COSMO-DARK"]
