"""
Toy simulator for WST observation scheduling.

Components:
  - Ephemeris: Sun/Moon positions, field altitudes, airmass, Moon separation
  - Weather model: clouds (Markov chain) + seeing (log-normal AR(1)) + closed nights
  - Programme pool: HYPOTHETICAL programmes inspired by WST science cases
  - Shared focal plane: when a field is observed, ALL programmes whose
    constraints are met get credit from that pointing (MOS-LR + MOS-HR + IFS simultaneously)
  - ToO: transient alerts arriving at random during the night
  - Policies: random, greedy-priority, hand-tuned merit, learned (CEM)

Note: the site coordinates are a placeholder (Paranal); the WST site is not yet final.
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from astropy.utils import iers  # noqa: E402

iers.conf.auto_download = False
iers.conf.auto_max_age = None
from astropy.time import Time  # noqa: E402
from astropy.coordinates import get_body, SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402


def checkpoint(msg):
    print(f"[CHECKPOINT] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SITE_LAT = -24.627   # deg (placeholder: Paranal)
SITE_LON = -70.404   # deg, east positive
SLOT_MIN = 20        # slot length (min); WST uses ~1000 s exposures
SLOT_S = SLOT_MIN * 60
SLOTS_PER_NIGHT = 42  # 21:00 UTC -> 11:00 UTC
YEAR_START = "2041-01-01"
N_NIGHTS_YEAR = 365
CONFIG_OVERHEAD_S = 120.0  # fibre positioning + setup
SLEW_S_PER_DEG = 1.0
FULL_EXP_S = 1050.0        # open-shutter time required for full credit

CLOUD_NAMES = ["photometric", "thin", "thick"]
CLOUD_TOL = {"clear": 0, "thin": 1}

FEATURE_NAMES = [
    "priority (credited)",
    "priority x scarcity",
    "airmass - 1",
    "airmass - best tonight",
    "set-urgency",
    "program lag",
    "slew / 100 deg",
    "seeing margin",
    "ToO flag",
]
N_FEAT = len(FEATURE_NAMES)


# ----------------------------------------------------------------------------
# Spherical helpers
# ----------------------------------------------------------------------------
def altitude(ra, dec, lst):
    """ra, dec, lst in degrees; broadcastable. Returns altitude (deg)."""
    ha = np.radians(lst - ra)
    lat = np.radians(SITE_LAT)
    d = np.radians(dec)
    s = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def angsep(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def airmass_from_alt(alt):
    alt = np.asarray(alt, float)
    am = np.full(alt.shape, np.inf)
    m = alt > 5
    am[m] = 1.0 / np.sin(np.radians(alt[m]))
    return am


# ----------------------------------------------------------------------------
# Ephemeris (precomputed for one year)
# ----------------------------------------------------------------------------
class Ephemeris:
    def __init__(self):
        checkpoint("Computing ephemeris (Sun/Moon, 1 year)...")
        t0 = Time(YEAR_START + "T21:00:00", scale="utc")
        night = np.arange(N_NIGHTS_YEAR)[:, None]
        slot = np.arange(SLOTS_PER_NIGHT)[None, :]
        dt_day = night + slot * SLOT_MIN / 1440.0
        times = t0 + dt_day.ravel() * u.day
        jd = times.jd
        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
        self.lst = ((gmst + SITE_LON) % 360).reshape(dt_day.shape)

        sun = get_body("sun", times)
        moon = get_body("moon", times)
        sun_ra, sun_dec = sun.ra.deg, sun.dec.deg
        moon_ra, moon_dec = moon.ra.deg, moon.dec.deg
        lst = self.lst.ravel()
        self.sun_alt = altitude(sun_ra, sun_dec, lst).reshape(dt_day.shape)
        self.moon_alt = altitude(moon_ra, moon_dec, lst).reshape(dt_day.shape)
        self.moon_ra = moon_ra.reshape(dt_day.shape)
        self.moon_dec = moon_dec.reshape(dt_day.shape)
        elong = angsep(sun_ra, sun_dec, moon_ra, moon_dec)
        self.fli = ((1 - np.cos(np.radians(elong))) / 2).reshape(dt_day.shape)
        # Astronomical darkness: Sun < -18 deg
        self.dark = self.sun_alt < -18
        self.hours_utc = (21 + slot * SLOT_MIN / 60.0).ravel()
        checkpoint(f"Ephemeris ready: {self.dark.sum()} dark slots / year")


# ----------------------------------------------------------------------------
# Weather model
# ----------------------------------------------------------------------------
def sample_weather(rng, n_nights):
    """Generates cloud[night, slot] and seeing[night, slot]; closed nights are fully thick."""
    P = np.array([[0.95, 0.045, 0.005],
                  [0.15, 0.80, 0.05],
                  [0.05, 0.15, 0.80]])
    cloud = np.zeros((n_nights, SLOTS_PER_NIGHT), int)
    seeing = np.zeros((n_nights, SLOTS_PER_NIGHT))
    for n in range(n_nights):
        closed = rng.random() < 0.08
        state = rng.choice(3, p=[0.75, 0.20, 0.05])
        ls = np.log(0.75) + 0.35 * rng.standard_normal()
        night_med = ls
        for s in range(SLOTS_PER_NIGHT):
            state = rng.choice(3, p=P[state])
            cloud[n, s] = 2 if closed else state
            ls = night_med + 0.92 * (ls - night_med) + 0.12 * rng.standard_normal()
            seeing[n, s] = np.exp(ls)
    return cloud, seeing


# ----------------------------------------------------------------------------
# Programme pool (HYPOTHETICAL, inspired by WST science cases)
# ----------------------------------------------------------------------------
PROGRAM_SPECS = [
    dict(name="COSMO-LBG (MOS-LR, dark)", prio=1.0, max_am=1.5, max_fli=0.4,
         min_msep=40, max_seeing=1.2, cloud="clear", pool="highlat", n=140, visits=3),
    dict(name="COSMO-BG (MOS-LR, bright)", prio=0.6, max_am=1.8, max_fli=1.0,
         min_msep=20, max_seeing=1.6, cloud="thin", pool="highlat", n=180, visits=1),
    dict(name="GAL-HALO (MOS-HR)", prio=0.8, max_am=1.6, max_fli=0.7,
         min_msep=30, max_seeing=1.3, cloud="thin", pool="highlat", n=100, visits=2),
    dict(name="GAL-DISC/BULGE (MOS-HR)", prio=0.7, max_am=1.6, max_fli=1.0,
         min_msep=25, max_seeing=1.4, cloud="thin", pool="lowlat", n=70, visits=2),
    dict(name="DWARFS & MCs (MOS)", prio=0.9, max_am=1.7, max_fli=0.6,
         min_msep=35, max_seeing=1.1, cloud="clear", pool="special", n=7, visits=4),
    dict(name="IFS DEEP FIELDS (IFS)", prio=1.0, max_am=1.3, max_fli=0.2,
         min_msep=50, max_seeing=0.8, cloud="clear", pool="ifs", n=8, visits=6),
]
TOO_SPEC = dict(name="ToO TRANSIENTS", prio=1.5, max_am=2.0, max_seeing=1.5,
                cloud="thin", rate_per_night=0.4, window_slots=18)

SPECIAL_FIELDS = {  # (ra, dec)
    "LMC": (80.9, -69.75), "SMC": (13.2, -72.8), "Sculptor": (15.0, -33.7),
    "Fornax": (40.0, -34.4), "Carina": (100.4, -50.97), "Sextans": (153.3, -1.6),
    "Sgr core": (283.8, -30.5),
}


class Survey:
    """Fields + programmes + required-visits matrix R0[p, f]."""

    def __init__(self, seed=42, n_fields=500):
        rng = np.random.default_rng(seed)
        # Uniform on the sphere, dec < +25
        ra = rng.uniform(0, 360, 4 * n_fields)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 4 * n_fields)))
        m = (dec < 25) & (dec > -88)
        ra, dec = ra[m][:n_fields], dec[m][:n_fields]
        sp = np.array(list(SPECIAL_FIELDS.values()))
        self.ra = np.concatenate([ra, sp[:, 0]])
        self.dec = np.concatenate([dec, sp[:, 1]])
        nF = len(self.ra)
        gb = SkyCoord(self.ra * u.deg, self.dec * u.deg).galactic.b.deg
        idx_special = np.arange(n_fields, nF)
        highlat = np.where((np.abs(gb) > 25) & (np.arange(nF) < n_fields))[0]
        lowlat = np.where((np.abs(gb) < 15) & (np.arange(nF) < n_fields))[0]

        self.progs = PROGRAM_SPECS
        nP = len(self.progs)
        self.R0 = np.zeros((nP, nF))
        for p, spec in enumerate(self.progs):
            if spec["pool"] == "highlat":
                pool = highlat
            elif spec["pool"] == "lowlat":
                pool = lowlat
            elif spec["pool"] == "special":
                pool = idx_special
            else:  # ifs: a few deep high-latitude fields (simultaneous with MOS)
                pool = highlat
            chosen = rng.choice(pool, size=min(spec["n"], len(pool)), replace=False)
            self.R0[p, chosen] = spec["visits"]

        self.prio = np.array([s["prio"] for s in self.progs])
        self.max_am = np.array([s["max_am"] for s in self.progs])
        self.max_fli = np.array([s["max_fli"] for s in self.progs])
        self.min_msep = np.array([s["min_msep"] for s in self.progs])
        self.max_see = np.array([s["max_seeing"] for s in self.progs])
        self.cloud_tol = np.array([CLOUD_TOL[s["cloud"]] for s in self.progs])
        self.names = [s["name"] for s in self.progs]
        self.scarcity = np.ones(nP)

    def calibrate_scarcity(self, eph, rng):
        """Measures how rarely each programme's constraints are met (1/P)."""
        cloud, seeing = sample_weather(rng, 60)
        nights = rng.choice(N_NIGHTS_YEAR, 60, replace=False)
        P = np.zeros(len(self.progs))
        for k, n in enumerate(nights):
            d = eph.dark[n]
            fli = np.where(eph.moon_alt[n] > 0, eph.fli[n], 0.0)[d]
            see = seeing[k][d] * 1.2 ** 0.6
            cl = cloud[k][d]
            ok = ((fli[None, :] <= self.max_fli[:, None]) &
                  (see[None, :] <= self.max_see[:, None]) &
                  (cl[None, :] <= self.cloud_tol[:, None]))
            P += ok.mean(axis=1)
        P /= len(nights)
        s = 1.0 / np.clip(P, 0.02, 1)
        self.scarcity = s / s.mean()
        return P


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
HANDTUNED_W = np.array([1.0, 0.5, -1.0, -2.0, 0.5, 0.5, -0.5, 0.3, 2.0])


class Policy:
    def __init__(self, kind, w=None, rng=None):
        self.kind = kind
        self.w = w
        self.rng = rng or np.random.default_rng(0)

    def choose(self, F, valid):
        """F: [n_cand, N_FEAT], valid: bool mask. Returns an index or -1 (idle)."""
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return -1
        if self.kind == "random":
            return self.rng.choice(idx)
        if self.kind == "greedy":
            sc = F[idx, 0] + 1e-6 * self.rng.random(len(idx))
            return idx[np.argmax(sc)]
        w = HANDTUNED_W if self.kind == "handtuned" else self.w
        sc = F[idx] @ w
        return idx[np.argmax(sc)]


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------
def run_block(survey, eph, policy, start_night, n_nights, seed, log=False):
    """Simulates an observing block of n_nights; returns a dict of metrics."""
    rng = np.random.default_rng(seed)
    cloud, seeing = sample_weather(rng, n_nights)
    sv = survey
    R = sv.R0.copy()
    R0_tot = sv.R0.sum(axis=1)
    nP, nF = R.shape
    ra_f, dec_f = sv.ra, sv.dec

    score = 0.0
    credited_visits = np.zeros(nP)
    n_obs = n_wasted = n_idle = n_dark = 0
    am_list = []
    overhead_tot = 0.0
    too_total = too_done = 0
    too_latency = []
    timeline = []
    last_ra, last_dec = None, None

    for k in range(n_nights):
        n = (start_night + k) % N_NIGHTS_YEAR
        dark = eph.dark[n]
        lst = eph.lst[n]
        alt = altitude(ra_f[None, :], dec_f[None, :], lst[:, None])  # [S, F]
        am = airmass_from_alt(alt)
        am_dark = np.where(dark[:, None], am, np.inf)
        best_am = am_dark.min(axis=0)
        # Number of slots left tonight with X<=1.5 (reverse cumulative)
        vis15 = (am_dark <= 1.5).astype(int)
        rem_vis = np.cumsum(vis15[::-1], axis=0)[::-1]

        # ToO alerts
        n_too = rng.poisson(TOO_SPEC["rate_per_night"])
        dark_slots = np.where(dark)[0]
        toos = []
        for _ in range(n_too):
            if len(dark_slots) == 0:
                break
            t_on = rng.choice(dark_slots)
            tra = rng.uniform(0, 360)
            tdec = np.degrees(np.arcsin(rng.uniform(-1, np.sin(np.radians(20)))))
            toos.append(dict(ra=tra, dec=tdec, t_on=t_on,
                             t_off=t_on + TOO_SPEC["window_slots"], done=False))
        too_total += len(toos)

        for s in range(SLOTS_PER_NIGHT):
            if not dark[s]:
                continue
            n_dark += 1
            # The scheduler sees the measured conditions of the previous slot (persistence)
            s_obs = max(s - 1, 0)
            cl_obs, see_obs = cloud[k, s_obs], seeing[k, s_obs]
            cl_true, see_true = cloud[k, s], seeing[k, s]

            moon_up = eph.moon_alt[n, s] > 0
            fli = eph.fli[n, s] if moon_up else 0.0
            msep = angsep(ra_f, dec_f, eph.moon_ra[n, s], eph.moon_dec[n, s])
            am_s = am[s]

            def ok_matrix(cl, see):
                see_eff = see * np.clip(am_s, 1, 5) ** 0.6
                moon_ok = (fli <= sv.max_fli[:, None]) & \
                          ((~moon_up) | (msep[None, :] >= sv.min_msep[:, None]))
                return ((R > 1e-9) & (am_s[None, :] <= sv.max_am[:, None]) &
                        moon_ok & (see_eff[None, :] <= sv.max_see[:, None]) &
                        (cl <= sv.cloud_tol[:, None])), see_eff

            ok, see_eff = ok_matrix(cl_obs, see_obs)

            # --- Features (fields) ---
            F = np.zeros((nF, N_FEAT))
            pr = sv.prio[:, None] * ok
            F[:, 0] = pr.sum(axis=0)
            F[:, 1] = (pr * sv.scarcity[:, None]).sum(axis=0)
            F[:, 2] = np.where(np.isfinite(am_s), am_s - 1, 5)
            F[:, 3] = np.where(np.isfinite(am_s), am_s - best_am, 5)
            F[:, 4] = 1.0 / np.maximum(rem_vis[s], 1)
            lag = (R.sum(axis=1) / np.maximum(R0_tot, 1))[:, None]
            F[:, 5] = (pr * lag).sum(axis=0)
            if last_ra is not None:
                F[:, 6] = angsep(ra_f, dec_f, last_ra, last_dec) / 100.0
            margin = np.where(ok, sv.max_see[:, None] - see_eff[None, :], np.inf)
            mm = margin.min(axis=0)
            F[:, 7] = np.where(np.isfinite(mm), mm, 0)
            valid = F[:, 0] > 0

            # --- ToO candidates ---
            active = [t for t in toos if (not t["done"]) and t["t_on"] <= s < t["t_off"]]
            if active:
                Ft = np.zeros((len(active), N_FEAT))
                vt = np.zeros(len(active), bool)
                for j, t in enumerate(active):
                    a = altitude(t["ra"], t["dec"], lst[s])
                    x = float(airmass_from_alt(np.array([a]))[0])
                    se = see_obs * min(x, 5) ** 0.6
                    good = (x <= TOO_SPEC["max_am"]) and (se <= TOO_SPEC["max_seeing"]) \
                        and (cl_obs <= CLOUD_TOL[TOO_SPEC["cloud"]])
                    vt[j] = good
                    Ft[j, 0] = TOO_SPEC["prio"] if good else 0
                    Ft[j, 1] = Ft[j, 0]
                    Ft[j, 2] = x - 1 if np.isfinite(x) else 5
                    Ft[j, 4] = 1.0 / max(t["t_off"] - s, 1)
                    if last_ra is not None:
                        Ft[j, 6] = angsep(t["ra"], t["dec"], last_ra, last_dec) / 100
                    Ft[j, 7] = TOO_SPEC["max_seeing"] - se
                    Ft[j, 8] = 1.0
                F = np.vstack([F, Ft])
                valid = np.concatenate([valid, vt])

            c = policy.choose(F, valid)
            if c < 0:
                n_idle += 1
                if log:
                    timeline.append(dict(night=k, slot=s, kind="idle", cloud=cl_true,
                                         seeing=see_true, fli=fli, am=np.nan, progs=[]))
                continue

            # --- Execution ---
            if c < nF:
                tra, tdec = ra_f[c], dec_f[c]
            else:
                t = active[c - nF]
                tra, tdec = t["ra"], t["dec"]
            slew = 0.0 if last_ra is None else float(angsep(tra, tdec, last_ra, last_dec))
            ovh = CONFIG_OVERHEAD_S + SLEW_S_PER_DEG * slew
            overhead_tot += ovh
            frac = min(1.0, (SLOT_S - ovh) / FULL_EXP_S)
            last_ra, last_dec = tra, tdec
            n_obs += 1
            got = []
            if c < nF:
                ok_true, _ = ok_matrix(cl_true, see_true)
                ps = np.where(ok_true[:, c])[0]
                for p in ps:
                    dv = min(frac, R[p, c])
                    R[p, c] -= dv
                    credited_visits[p] += dv
                    score += sv.prio[p] * dv
                    got.append(p)
                x_obs = am_s[c]
            else:
                a = altitude(t["ra"], t["dec"], lst[s])
                x_obs = float(airmass_from_alt(np.array([a]))[0])
                se = see_true * min(x_obs, 5) ** 0.6
                if (x_obs <= TOO_SPEC["max_am"]) and (se <= TOO_SPEC["max_seeing"]) and \
                        (cl_true <= CLOUD_TOL[TOO_SPEC["cloud"]]):
                    t["done"] = True
                    too_done += 1
                    too_latency.append((s - t["t_on"]) * SLOT_MIN)
                    score += TOO_SPEC["prio"] * frac
                    got.append(nP)  # ToO code
            if got:
                am_list.append(x_obs)
            else:
                n_wasted += 1
            if log:
                timeline.append(dict(night=k, slot=s, kind="obs", cloud=cl_true,
                                     seeing=see_true, fli=fli, am=x_obs, progs=got))

    comp = credited_visits / np.maximum(R0_tot, 1)
    return dict(
        score=score,
        completion=comp,
        n_obs=n_obs, n_wasted=n_wasted, n_idle=n_idle, n_dark=n_dark,
        mean_airmass=float(np.mean(am_list)) if am_list else np.nan,
        overhead_frac=overhead_tot / max(n_obs * SLOT_S, 1),
        too_total=too_total, too_done=too_done,
        too_latency=float(np.median(too_latency)) if too_latency else np.nan,
        timeline=timeline,
    )


# ----------------------------------------------------------------------------
# Learning: optimise the merit weights with the Cross-Entropy Method
# ----------------------------------------------------------------------------
def train_cem(survey, eph, n_iter=15, pop=24, elite_frac=0.25, n_eval=2,
              block_nights=28, seed=0):
    rng = np.random.default_rng(seed)
    mu = HANDTUNED_W.copy()
    sigma = np.ones(N_FEAT) * 1.0
    n_elite = max(2, int(pop * elite_frac))
    history = []
    for it in range(n_iter):
        # Common random numbers: everyone in an iteration gets the same weather/dates
        starts = rng.integers(0, N_NIGHTS_YEAR, n_eval)
        seeds = rng.integers(0, 10_000, n_eval)
        W = mu + sigma * rng.standard_normal((pop, N_FEAT))
        W[0] = mu
        fit = np.zeros(pop)
        for i in range(pop):
            pol = Policy("learned", w=W[i])
            fit[i] = np.mean([run_block(survey, eph, pol, st, block_nights, sd)["score"]
                              for st, sd in zip(starts, seeds)])
        # Greedy reference under the same conditions (for a normalised learning curve)
        ref = np.mean([run_block(survey, eph, Policy("greedy"), st, block_nights, sd)["score"]
                       for st, sd in zip(starts, seeds)])
        elite = W[np.argsort(fit)[-n_elite:]]
        mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
        sigma = np.maximum(0.7 * elite.std(axis=0) + 0.3 * sigma, 0.05)
        history.append(dict(it=it, best=fit.max() / ref, mean=fit.mean() / ref,
                            mu_fit=fit[0] / ref))
        checkpoint(f"CEM iter {it + 1}/{n_iter}: best={fit.max() / ref:.3f}x greedy, "
                   f"mean={fit.mean() / ref:.3f}x")
    return mu, history
