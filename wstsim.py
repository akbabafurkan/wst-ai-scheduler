"""
Toy simulator for WST observation scheduling (v1).

What changed with respect to v0 (see README for the full description):
  - Event-driven nights: exposures are 180 s ("short", bright-time mode) or 1000 s ("long"),
    plus fibre-positioning and slew overheads; time advances by the actual duration.
  - Real tiling: ~10,000 tiles of 3.1 deg^2 (Fibonacci sphere) south of dec=+30, with
    programme footprints (LSST-WFD-like cosmology area, halo, disc, dwarfs, star-forming
    regions, IFS deep fields).
  - Two fibre pools shared on the same focal plane: MOS-LR (30,000) and MOS-HR (2,000), plus
    the IFS. Fibres are split between eligible programmes (priority-weighted share, then
    leftovers by priority) and assigned to targets with Poisson collisions, so successive
    passes on a tile return fewer new targets (Payerne et al. 2026, Sec. 5.2).
  - Continuous exposure quality: effective time t_eff = t_exp * speed, where speed depends on
    transparency, seeing (fibre aperture losses), extinction and sky brightness (Moon), with a
    programme-dependent sky sensitivity. Credit is proportional to t_eff / t_req.
  - Cadence programmes (epochs with a minimum separation) and an IFS-driven programme.
  - Concave (diminishing-return) science utility per programme, Fisher-like for cosmology.
  - Multi-year horizon (default 5 years) with survey-state snapshots for training.

Placeholder site: Paranal (the WST site is not final).
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process

import numpy as np  # noqa: E402

from programmes import PROGRAMMES, TOO, FOV_DEG2, FIBRES  # noqa: E402

warnings.filterwarnings("ignore")


def checkpoint(msg):
    print(f"[CHECKPOINT] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SITE_LAT = -24.627          # deg (placeholder: Paranal)
SITE_LON = -70.404          # deg, east positive
YEAR_START = "2041-01-01"
STEP_MIN = 10               # ephemeris / weather grid (min)
STEP_S = STEP_MIN * 60
STEPS_PER_NIGHT = 84        # 21:00 UTC -> 11:00 UTC
N_NIGHTS_EPH = 7 * 365 + 2  # 7 years of ephemeris (cosmology validation runs 7 years)
SURVEY_NIGHTS = 5 * 365 + 1  # nominal survey length: 5 years

MODE_T = np.array([180.0, 1000.0])   # short (bright-time) and long exposures [s]
MODE_NAMES = ["short", "long"]
CONFIG_OVERHEAD_S = 120.0   # fibre positioning + setup
SLEW_S_PER_DEG = 1.0
SPEED_MIN = 0.3             # below this relative speed an exposure gives no usable spectra
SPEED_MAX = 3.0
KAPPA = 2.0                 # max useful fibres per remaining target on a tile (collisions)
MAX_AM_CAND = 1.8
PRESELECT_MIN = 600      # above this many visible tiles, pre-select a short list
PRESELECT_K = 200

FIBRE_R_ARCSEC = 0.75       # 1.5" fibre diameter (Payerne et al. 2026, footnote 7)
SEEING_REF = 0.8            # seeing at which t_req is defined
EXT_K = 0.15                # extinction (mag / airmass)
CLOUD_NAMES = ["photometric", "thin", "thick"]
CLOUD_T = np.array([1.0, 0.7, 0.1])   # transparency
CLOSED_NIGHT_P = 0.12

CAT_NAMES = ["dark", "grey", "bright"]   # ESO: FLI<0.4 (or Moon down), 0.4-0.7, >0.7
TIME_CAT = {"dark": 0, "grey": 1, "bright": 2}
UTIL_KIND = {"fisher": 0, "concave": 1, "linear": 2}

FEATURE_NAMES = [
    "science gain rate",
    "scarcity x gain rate",
    "lag x gain rate",
    "airmass - 1",
    "airmass - best tonight",
    "set urgency",
    "slew / 100 deg",
    "cadence gain rate",
    "ToO flag",
    "short-exposure flag",
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


def fibre_fraction(seeing):
    sig = np.asarray(seeing) / 2.355
    return 1.0 - np.exp(-FIBRE_R_ARCSEC ** 2 / (2 * sig ** 2))


FF_REF = float(fibre_fraction(SEEING_REF))


def moon_sky(fli, moon_alt, msep):
    """Extra sky brightness from the Moon, relative to the dark zenith sky."""
    if moon_alt <= 0 or fli <= 0:
        return np.zeros_like(msep)
    return 8.0 * fli ** 2 * np.sin(np.radians(moon_alt)) ** 0.5 * (1 + 3 * np.exp(-msep / 20.0))


# ----------------------------------------------------------------------------
# Ephemeris (cached to cache/ephemeris.npz)
# ----------------------------------------------------------------------------
class Ephemeris:
    def __init__(self, n_nights=N_NIGHTS_EPH, cache="cache/ephemeris.npz"):
        if os.path.exists(cache):
            d = np.load(cache)
            if d["lst"].shape[0] >= n_nights:
                for k in ["lst", "sun_alt", "moon_alt", "moon_ra", "moon_dec", "fli"]:
                    setattr(self, k, d[k][:n_nights])
                self._finish()
                checkpoint(f"Ephemeris loaded from {cache} ({n_nights} nights)")
                return
        checkpoint(f"Computing ephemeris (Sun/Moon, {n_nights} nights, {STEP_MIN}-min grid)...")
        from astropy.utils import iers
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        from astropy.time import Time
        from astropy.coordinates import get_body
        import astropy.units as u
        t0 = Time(YEAR_START + "T21:00:00", scale="utc")
        night = np.arange(n_nights)[:, None]
        step = np.arange(STEPS_PER_NIGHT)[None, :]
        dt_day = (night + step * STEP_MIN / 1440.0).ravel()
        times = t0 + dt_day * u.day
        jd = times.jd
        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
        lst = (gmst + SITE_LON) % 360
        sun = get_body("sun", times)
        moon = get_body("moon", times)
        shape = (n_nights, STEPS_PER_NIGHT)
        self.lst = lst.reshape(shape)
        self.sun_alt = altitude(sun.ra.deg, sun.dec.deg, lst).reshape(shape)
        self.moon_alt = altitude(moon.ra.deg, moon.dec.deg, lst).reshape(shape)
        self.moon_ra = moon.ra.deg.reshape(shape)
        self.moon_dec = moon.dec.deg.reshape(shape)
        elong = angsep(sun.ra.deg, sun.dec.deg, moon.ra.deg, moon.dec.deg)
        self.fli = ((1 - np.cos(np.radians(elong))) / 2).reshape(shape)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, lst=self.lst, sun_alt=self.sun_alt, moon_alt=self.moon_alt,
                            moon_ra=self.moon_ra, moon_dec=self.moon_dec, fli=self.fli)
        self._finish()
        checkpoint(f"Ephemeris ready and cached: {self.dark.sum()} dark steps")

    def _finish(self):
        self.n_nights = self.lst.shape[0]
        self.dark = self.sun_alt < -18
        fli_eff = np.where(self.moon_alt > 0, self.fli, 0.0)
        self.cat = np.where(fli_eff < 0.4, 0, np.where(fli_eff <= 0.7, 1, 2))


def time_category_fractions(eph, n_nights=SURVEY_NIGHTS, moon_altitude=True):
    """Fraction of astronomically dark time in ESO dark / grey / bright. With
    moon_altitude=False the category follows the Moon phase (FLI) only."""
    d = eph.dark[:n_nights]
    if moon_altitude:
        c = eph.cat[:n_nights][d]
    else:
        f = eph.fli[:n_nights][d]
        c = np.where(f < 0.4, 0, np.where(f <= 0.7, 1, 2))
    return np.array([(c == k).mean() for k in range(3)]), d.sum() * STEP_S / 3600 / (n_nights / 365.25)


# ----------------------------------------------------------------------------
# Weather: deterministic per (seed, night), so every policy sees the same sky
# ----------------------------------------------------------------------------
P_CLOUD = np.array([[0.975, 0.022, 0.003],
                    [0.08, 0.895, 0.025],
                    [0.025, 0.08, 0.895]])
P_CLOUD_CUM = np.cumsum(P_CLOUD, axis=1)


def night_weather(seed, night):
    rng = np.random.default_rng([int(seed), int(night), 7])
    closed = rng.random() < CLOSED_NIGHT_P
    u = rng.random(STEPS_PER_NIGHT + 1)
    z = rng.standard_normal(STEPS_PER_NIGHT + 1)
    state = int(np.searchsorted([0.75, 0.95, 1.0], u[0]))
    med = np.log(0.75) + 0.3 * z[0]
    ls = med
    cloud = np.empty(STEPS_PER_NIGHT, int)
    seeing = np.empty(STEPS_PER_NIGHT)
    for s in range(STEPS_PER_NIGHT):
        state = int(np.searchsorted(P_CLOUD_CUM[state], u[s + 1]))
        cloud[s] = min(state, 2)
        ls = med + 0.96 * (ls - med) + 0.08 * z[s + 1]
        seeing[s] = np.exp(ls)
    return closed, cloud, seeing


# ----------------------------------------------------------------------------
# Survey: tiles, footprints, programmes
# ----------------------------------------------------------------------------
DWARFS = {  # (ra, dec) of southern / equatorial dwarf galaxies and the Magellanic Clouds
    "LMC": (80.9, -69.75), "SMC": (13.2, -72.8), "Sculptor": (15.04, -33.71),
    "Fornax": (39.99, -34.45), "Carina": (100.40, -50.97), "Sextans": (153.26, -1.61),
    "Sgr core": (283.83, -30.55), "Leo I": (152.12, 12.31), "Leo II": (168.37, 22.15),
    "Phoenix": (27.78, -44.44), "Tucana": (340.46, -64.42), "Crater II": (177.31, -18.41),
    "Antlia 2": (143.89, -36.77), "Reticulum II": (53.93, -54.05), "Bootes I": (210.03, 14.50),
    "Hercules": (247.77, 12.79),
}


class State:
    """Mutable survey state (copyable, picklable)."""

    def __init__(self, R, n_prog, n_tiles):
        self.R = R
        self.acc = np.zeros((n_prog, n_tiles))
        self.ep = np.zeros((n_prog, n_tiles))
        self.last = np.full((n_prog, n_tiles), -1e9)
        self.has_work = np.ones(n_tiles, bool)
        self.too_done = 0
        self.too_total = 0

    def copy(self):
        s = State.__new__(State)
        for k, v in self.__dict__.items():
            s.__dict__[k] = v.copy() if isinstance(v, np.ndarray) else v
        return s


class Survey:
    def __init__(self, programmes=PROGRAMMES, seed=42):
        rng = np.random.default_rng(seed)
        n = int(round(41253 / FOV_DEG2))
        i = np.arange(n) + 0.5
        dec = np.degrees(np.arcsin(1 - 2 * i / n))
        ra = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
        m = dec < 30
        self.ra, self.dec = ra[m], dec[m]
        nF = len(self.ra)
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        gal = SkyCoord(self.ra * u.deg, self.dec * u.deg).galactic
        self.l, self.b = gal.l.deg, gal.b.deg
        absb = np.abs(self.b)
        south = self.dec < 20

        wfd = (self.dec > -72) & (self.dec < 12.5) & (absb > 15)
        fp = dict(
            south=south, wfd=wfd, halo=south & (absb > 20), disc=south & (absb < 15),
            insitu=south,
        )
        dw = np.unique([np.argmin(angsep(self.ra, self.dec, r, d)) for r, d in DWARFS.values()])
        fp["dwarfs"] = np.isin(np.arange(nF), dw)
        sfr_pool = np.where(south & (absb < 5))[0]
        fp["sfr"] = np.isin(np.arange(nF), rng.choice(sfr_pool, 40, replace=False))
        ifs_pool = np.where(wfd & (absb > 40))[0]
        fp["ifs"] = np.isin(np.arange(nF), rng.choice(ifs_pool, 20, replace=False))
        self.footprints = fp

        self.progs = programmes
        P = len(programmes)
        self.names = [p["name"] for p in programmes]
        self.labels = [p["label"] for p in programmes]
        self.groups = [p["group"] for p in programmes]
        g = lambda k: np.array([p[k] for p in programmes], float)  # noqa: E731
        self.prio, self.t_req, self.beta = g("prio"), g("t_req"), g("beta")
        self.sky_ref, self.max_am = g("sky_ref"), g("max_am")
        self.max_see, self.min_msep = g("max_seeing"), g("min_msep")
        self.visit = np.array([p["kind"] == "visit" for p in programmes])
        self.n_ep = np.array([p.get("n_epochs", 1) for p in programmes], float)
        self.min_gap = np.array([p.get("min_gap", 0) for p in programmes], float)
        self.time_cat = np.array([TIME_CAT[p["time_cat"]] for p in programmes])
        self.util_kind = np.array([UTIL_KIND[p["utility"][0]] for p in programmes])
        self.util_par = np.array([p["utility"][1] if len(p["utility"]) > 1 else 0
                                  for p in programmes], float)
        self.pool_idx = {}
        for pool in FIBRES:
            idx = [k for k, p in enumerate(programmes) if p["pool"] == pool]
            if idx:
                self.pool_idx[pool] = np.array(sorted(idx, key=lambda k: -self.prio[k]))

        self.N0 = np.zeros((P, nF))
        for k, p in enumerate(programmes):
            mask = fp[p["footprint"]]
            if p["kind"] == "visit":
                self.N0[k, mask] = p["density"]
            elif p["footprint"] == "insitu":
                self.N0[k, mask] = np.where(absb[mask] < 15, 1000, 150) * FOV_DEG2
            else:
                self.N0[k, mask] = p["density"] * FOV_DEG2
        self.Ntot = self.N0.sum(axis=1)
        self.nfields = (self.N0 > 0).sum(axis=1)
        self.fp_idx = [np.where(self.N0[k] > 0)[0] for k in range(P)]
        self.scarcity = np.ones(P)
        self.n_tiles = nF
        self.area = {k: v.sum() * FOV_DEG2 for k, v in fp.items()}

    def new_state(self):
        st = State(self.N0.copy(), len(self.progs), self.n_tiles)
        st.has_work = self.N0.sum(axis=0) > 0
        return st

    # ----- science utility --------------------------------------------------
    def g(self, p, c):
        k, a = self.util_kind[p], self.util_par[p]
        if k == 0:
            x = c * a
            return (x / (1 + x)) ** 2 / (a / (1 + a)) ** 2
        if k == 1:
            return (1 - np.exp(-a * c)) / (1 - np.exp(-a))
        return c

    def utilities(self, st):
        U = np.zeros(len(self.progs))
        for p, m in enumerate(self.fp_idx):
            if self.visit[p]:
                U[p] = np.mean((st.ep[p, m] + np.minimum(st.acc[p, m], 1)) / self.n_ep[p])
            else:
                c = 1 - st.R[p, m] / self.N0[p, m]
                U[p] = (self.N0[p, m] * self.g(p, c)).sum() / self.Ntot[p]
        return U

    def completeness(self, st):
        C = np.zeros(len(self.progs))
        for p, m in enumerate(self.fp_idx):
            if self.visit[p]:
                C[p] = np.mean((st.ep[p, m] + np.minimum(st.acc[p, m], 1)) / self.n_ep[p])
            else:
                C[p] = 1 - st.R[p, m].sum() / self.N0[p, m].sum()
        return C

    def score(self, st, U=None):
        U = self.utilities(st) if U is None else U
        too = TOO["weight"] * st.too_done / (TOO["rate_per_night"] * SURVEY_NIGHTS)
        return float((self.prio * U).sum() + too)

    def max_score(self):
        return float(self.prio.sum() + TOO["weight"])

    # ----- how often are each programme's conditions met? ------------------
    def calibrate_scarcity(self, eph, n_samples=300, seed=1):
        rng = np.random.default_rng(seed)
        ok = np.zeros(len(self.progs))
        cnt = 0
        while cnt < n_samples:
            n = rng.integers(0, 365)
            dsteps = np.where(eph.dark[n])[0]
            if len(dsteps) == 0:
                continue
            g = rng.choice(dsteps)
            closed, cloud, seeing = night_weather(10_000 + cnt, n)
            if closed:
                cnt += 1
                continue
            alt = altitude(self.ra, self.dec, eph.lst[n, g])
            X = airmass_from_alt(alt)
            vis = X <= 1.5
            if not vis.any():
                continue
            speed, hard = conditions(self, eph, n, g, cloud[g], seeing[g], np.where(vis)[0], X[vis])[:2]
            good = hard & (speed >= 0.6)
            for p in range(len(self.progs)):
                inside = self.N0[p, vis] > 0
                if inside.any():
                    ok[p] += good[p, inside].mean()
            cnt += 1
        P_ok = ok / n_samples
        s = 1.0 / np.clip(P_ok, 0.02, 1)
        self.scarcity = s / s.mean()
        return P_ok


def conditions(sv, eph, n, g, cloud, seeing, cand, X):
    """Speed [P, C] and hard-constraint mask [P, C] for candidate tiles."""
    moon_alt = eph.moon_alt[n, g]
    moon_up = moon_alt > 0
    fli = eph.fli[n, g] if moon_up else 0.0
    msep = angsep(sv.ra[cand], sv.dec[cand], eph.moon_ra[n, g], eph.moon_dec[n, g])
    see_eff = seeing * np.clip(X, 1, 5) ** 0.6
    base = CLOUD_T[cloud] ** 2 * (fibre_fraction(see_eff) / FF_REF) ** 2 * \
        10 ** (-0.8 * EXT_K * (np.clip(X, 1, 5) - 1))
    sky = 0.6 + 0.4 * np.clip(X, 1, 5) + moon_sky(fli, moon_alt, msep)
    speed = np.minimum(base[None, :] * (sv.sky_ref[:, None] / sky[None, :]) ** sv.beta[:, None],
                       SPEED_MAX)
    hard = (X[None, :] <= sv.max_am[:, None]) & (see_eff[None, :] <= sv.max_see[:, None])
    if moon_up:
        hard &= msep[None, :] >= sv.min_msep[:, None]
    return speed, hard, base / sky, see_eff


def allocate_fibres(sv, ok, cap):
    """Split each fibre pool between eligible programmes: priority-weighted share
    (capped by what a programme can use), then leftovers by priority order."""
    F = np.zeros_like(cap)
    for pool, idx in sv.pool_idx.items():
        o = ok[idx]
        if not o.any():
            continue
        w = sv.prio[idx, None] * o
        W = w.sum(axis=0)
        share = np.divide(w, W, out=np.zeros_like(w), where=W > 0)
        c = cap[idx] * o
        Fp = np.minimum(c, share * FIBRES[pool])
        left = FIBRES[pool] - Fp.sum(axis=0)
        for j in range(len(idx)):
            add = np.clip(np.minimum(left, c[j] - Fp[j]), 0, None)
            Fp[j] += add
            left -= add
        F[idx] = Fp
    return F


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
HANDTUNED_W = np.array([1.0, 0.5, 0.5, -0.5, -1.0, 0.3, -0.3, 0.5, 1.0, 0.0])


class Policy:
    """kind: random | eso (ESO time-category plan + gain-rate tile choice) | greedy |
    handtuned | learned"""

    def __init__(self, kind, w=None):
        self.kind = kind
        self.w = None if w is None else np.asarray(w, float)

    def choose(self, Fe, valid, aux, rng):
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return -1
        if self.kind == "random":
            return int(rng.choice(idx))
        if self.kind in ("eso", "greedy"):
            # "eso" differs from "greedy" only upstream: programmes and exposure mode are
            # restricted to the ESO time category of the moment (see run_night)
            return int(idx[np.argmax(Fe[idx, 0])])
        w = HANDTUNED_W if self.kind == "handtuned" else self.w
        return int(idx[np.argmax(Fe[idx] @ w)])


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------
def _new_acc():
    return dict(dark_s=0.0, closed_s=0.0, idle_s=0.0, open_s=0.0, overhead_s=0.0,
                wasted_s=0.0, n_exp=0, n_short=0, am_sum=0.0, am_n=0,
                cat_mode_s=np.zeros((3, 2)), prog_open_s=np.zeros(len(PROGRAMMES)))


def _evaluate(sv, st, eph, n, g, gobs, cloud, seeing, cand, X, U, best_am, rem_vis,
              last_pos, toos, prog_mask, modes):
    """Score every (tile, exposure-mode) candidate under the *observed* conditions."""
    P = len(sv.progs)
    C = len(cand)
    speed, hard, too_speed, see_eff = conditions(sv, eph, n, g, cloud[gobs], seeing[gobs], cand, X)
    N0c, Rc = sv.N0[:, cand], st.R[:, cand]
    accc, epc, lastc = st.acc[:, cand], st.ep[:, cand], st.last[:, cand]
    work_v = (N0c > 0) & (epc < sv.n_ep[:, None]) & \
        ((n - lastc >= sv.min_gap[:, None]) | (accc > 0))
    work = np.where(sv.visit[:, None], work_v, Rc > 0.5)
    ok0 = work & hard & (speed >= SPEED_MIN) & prog_mask[:, None]
    slew = np.zeros(C) if last_pos is None else \
        angsep(sv.ra[cand], sv.dec[cand], sv.ra[last_pos], sv.dec[last_pos])
    if C > PRESELECT_MIN:
        # Stage 1: cheap proxies over all visible tiles; keep a short list (union of the
        # best tiles by approximate gain, by priority x remaining work, by slew and by
        # airmass relative to tonight's best), plus tiles holding an active ToO.
        frac_l = np.minimum(1.0, MODE_T[1] * speed / sv.t_req[:, None]) * ok0
        rem = np.where(sv.visit[:, None], 1.0, Rc / np.maximum(N0c, 1e-9))
        proxy = np.zeros(C)
        for p in np.where(ok0.any(axis=1))[0]:
            if sv.visit[p]:
                proxy += sv.prio[p] * frac_l[p] / (sv.n_ep[p] * sv.nfields[p])
            else:
                c0 = 1 - rem[p]
                slope = (sv.g(p, np.minimum(c0 + 0.01, 1)) - sv.g(p, c0)) / 0.01
                proxy += sv.prio[p] * frac_l[p] * slope * \
                    np.minimum(FIBRES[sv.progs[p]["pool"]], Rc[p]) / sv.Ntot[p]
        eso_proxy = (sv.prio[:, None] * (ok0 * rem)).sum(axis=0)
        keep = np.zeros(C, bool)
        for score, k in [(proxy, PRESELECT_K), (eso_proxy, PRESELECT_K),
                         (-slew, PRESELECT_K // 2), (-(X - best_am[cand]), PRESELECT_K // 2)]:
            keep[np.argpartition(-score, k)[:k]] = True
        too_tiles = {t["tile"] for t in toos if not t["done"] and t["on"] <= g < t["off"]}
        keep |= np.isin(cand, list(too_tiles))
        keep &= ok0.any(axis=0) | np.isin(cand, list(too_tiles))
        if not keep.any():
            keep[:] = True
        cand, X, slew = cand[keep], X[keep], slew[keep]
        speed, hard, too_speed, see_eff = speed[:, keep], hard[:, keep], too_speed[keep], see_eff[keep]
        N0c, Rc, accc, epc = N0c[:, keep], Rc[:, keep], accc[:, keep], epc[:, keep]
        work, ok0 = work[:, keep], ok0[:, keep]
        C = len(cand)
    cap = np.where(sv.visit[:, None], N0c, KAPPA * Rc)
    # ToO bonus (long mode only)
    too_val = TOO["weight"] / (TOO["rate_per_night"] * SURVEY_NIGHTS)
    too_bonus = np.zeros(C)
    pos = {f: c for c, f in enumerate(cand)}
    for t in toos:
        if t["done"] or not (t["on"] <= g < t["off"]) or t["tile"] not in pos:
            continue
        c = pos[t["tile"]]
        if X[c] <= TOO["max_am"] and see_eff[c] <= TOO["max_seeing"] and too_speed[c] >= SPEED_MIN:
            too_bonus[c] += too_val

    out = []
    for m in modes:
        tm = MODE_T[m]
        frac = np.minimum(1.0, tm * speed / sv.t_req[:, None]) * ok0
        F = allocate_fibres(sv, frac > 0, cap)
        dU = np.zeros((P, C))
        for p in np.where((F > 0).any(axis=1))[0]:
            if sv.visit[p]:
                add = frac[p] * np.minimum(1, F[p] / np.maximum(N0c[p], 1e-9))
                dU[p] = (np.minimum(accc[p] + add, 1) - np.minimum(accc[p], 1)) / \
                    sv.n_ep[p] / sv.nfields[p]
            else:
                R = Rc[p]
                n0 = np.maximum(N0c[p], 1e-9)
                assigned = R * -np.expm1(-F[p] * frac[p] / np.maximum(R, 1e-9))
                c0 = 1 - R / n0
                dU[p] = N0c[p] * (sv.g(p, np.minimum(c0 + assigned / n0, 1)) - sv.g(p, c0)) / sv.Ntot[p]
        pdU = sv.prio[:, None] * dU
        too_b = too_bonus if m == 1 else np.zeros(C)
        gain = pdU.sum(axis=0) + too_b
        ttot = tm + CONFIG_OVERHEAD_S + SLEW_S_PER_DEG * slew
        Fe = np.zeros((C, N_FEAT))
        Fe[:, 0] = gain / ttot
        Fe[:, 1] = ((sv.scarcity[:, None] * pdU).sum(axis=0) + too_b) / ttot
        Fe[:, 2] = (((1 - U)[:, None] * pdU).sum(axis=0) + too_b) / ttot
        Fe[:, 3] = X - 1
        Fe[:, 4] = X - best_am[cand]
        Fe[:, 5] = 1.0 / np.maximum(rem_vis[cand], 1)
        Fe[:, 6] = slew / 100.0
        Fe[:, 7] = pdU[sv.visit].sum(axis=0) / ttot
        Fe[:, 8] = too_b > 0
        Fe[:, 9] = 1.0 if m == 0 else 0.0
        rem = np.where(sv.visit[:, None], 1.0, Rc / np.maximum(N0c, 1e-9))
        eso = (sv.prio[:, None] * (frac > 0) * rem).sum(axis=0) + 10 * too_b / max(too_val, 1e-12)
        out.append(dict(m=m, F=F, Fe=Fe, gain=gain, eso=eso, ttot=ttot, dU=dU))
    Fe = np.vstack([o["Fe"] for o in out])
    for k in (0, 1, 2, 7):
        mx = Fe[:, k].max()
        if mx > 0:
            Fe[:, k] /= mx
    valid = np.concatenate([o["gain"] for o in out]) > 0
    aux = dict(eso=np.concatenate([o["eso"] for o in out]))
    return out, Fe, valid, aux, work, slew, cand


def _execute(sv, st, eph, n, g, cloud, seeing, f, m, F_col, toos, work_col, acc):
    """Apply one exposure under the *true* conditions; returns (credited programmes, ToO done)."""
    X = float(airmass_from_alt(np.array([altitude(sv.ra[f], sv.dec[f], eph.lst[n, g])]))[0])
    speed, hard, too_speed, see_eff = conditions(sv, eph, n, g, cloud[g], seeing[g],
                                                 np.array([f]), np.array([X]))
    speed, hard = speed[:, 0], hard[:, 0]
    tm = MODE_T[m]
    ok = work_col & hard & (speed >= SPEED_MIN) & (F_col > 0)
    frac = np.minimum(1.0, tm * speed / sv.t_req) * ok
    got = []
    for p in np.where(frac > 0)[0]:
        if sv.visit[p]:
            st.acc[p, f] += frac[p] * min(1.0, F_col[p] / sv.N0[p, f])
            if st.acc[p, f] >= 1 - 1e-9:
                st.ep[p, f] += 1
                st.acc[p, f] = 0.0
                st.last[p, f] = n
        else:
            R = st.R[p, f]
            st.R[p, f] = R - R * -np.expm1(-F_col[p] * frac[p] / max(R, 1e-9))
        acc["prog_open_s"][p] += tm * F_col[p] / FIBRES[sv.progs[p]["pool"]]
        got.append(p)
    too_hit = False
    if m == 1:
        for t in toos:
            if (not t["done"]) and t["on"] <= g < t["off"] and t["tile"] == f and \
                    X <= TOO["max_am"] and see_eff[0] <= TOO["max_seeing"] and \
                    too_speed[0] >= SPEED_MIN:
                t["done"] = True
                st.too_done += 1
                too_hit = True
    vis_work = (sv.N0[:, f] > 0) & np.where(sv.visit, st.ep[:, f] < sv.n_ep, st.R[:, f] > 0.5)
    st.has_work[f] = vis_work.any()
    return got, too_hit, X


def run_night(sv, eph, policy, st, n, seed, acc, rng, log=None, prog_subset=None):
    dark = eph.dark[n]
    dsteps = np.where(dark)[0]
    if len(dsteps) == 0:
        return
    acc["dark_s"] += len(dsteps) * STEP_S
    closed, cloud, seeing = night_weather(seed, n)
    # ToO alerts (deterministic per seed/night)
    trng = np.random.default_rng([int(seed), int(n), 3])
    toos = []
    for _ in range(trng.poisson(TOO["rate_per_night"])):
        ra = trng.uniform(0, 360)
        dec = np.degrees(np.arcsin(trng.uniform(-1, np.sin(np.radians(20)))))
        on = int(trng.choice(dsteps))
        toos.append(dict(tile=int(np.argmin(angsep(sv.ra, sv.dec, ra, dec))), on=on,
                         off=on + int(TOO["window_h"] * 60 / STEP_MIN), done=False))
    st.too_total += len(toos)
    if log is not None:
        log.update(night=n, cloud=cloud, seeing=seeing, closed=closed,
                   fli=np.where(eph.moon_alt[n] > 0, eph.fli[n], 0.0), dark=dark, exps=[])
    if closed:
        acc["closed_s"] += len(dsteps) * STEP_S
        return
    alt = altitude(sv.ra[None, :], sv.dec[None, :], eph.lst[n][:, None])
    am = airmass_from_alt(alt)
    amd = np.where(dark[:, None], am, np.inf)
    best_am = amd.min(axis=0)
    rem_vis_all = np.cumsum((amd <= 1.5)[::-1], axis=0)[::-1]
    U = sv.utilities(st)
    all_progs = np.ones(len(sv.progs), bool) if prog_subset is None else prog_subset
    t = dsteps[0] * STEP_S
    t_end = (dsteps[-1] + 1) * STEP_S
    last_pos = None
    while t < t_end:
        g = int(t // STEP_S)
        if not dark[g]:
            t = (g + 1) * STEP_S
            continue
        cand = np.where((am[g] <= MAX_AM_CAND) & st.has_work)[0]
        if len(cand) == 0:
            acc["idle_s"] += STEP_S
            t = (g + 1) * STEP_S
            continue
        X = am[g, cand]
        gobs = max(g - 1, 0)
        if policy.kind == "eso":
            cat = eph.cat[n, g]
            tries = [(all_progs & (sv.time_cat == cat), [0] if cat == 2 else [1]),
                     (all_progs, [0, 1])]
        else:
            tries = [(all_progs, [0, 1])]
        for pmask, modes in tries:
            out, Fe, valid, aux, work, slew, cand_k = _evaluate(
                sv, st, eph, n, g, gobs, cloud, seeing, cand, X, U, best_am, rem_vis_all[g],
                last_pos, toos, pmask, modes)
            if valid.any():
                break
        i = policy.choose(Fe, valid, aux, rng)
        if i < 0:
            acc["idle_s"] += STEP_S
            t = (g + 1) * STEP_S
            continue
        C = len(cand_k)
        o = out[i // C]
        c = i % C
        f = int(cand_k[c])
        m = o["m"]
        got, too_hit, Xo = _execute(sv, st, eph, n, g, cloud, seeing, f, m, o["F"][:, c],
                                    toos, work[:, c], acc)
        tm = MODE_T[m]
        ovh = CONFIG_OVERHEAD_S + SLEW_S_PER_DEG * slew[c]
        acc["open_s"] += tm
        acc["overhead_s"] += ovh
        acc["n_exp"] += 1
        acc["n_short"] += int(m == 0)
        acc["cat_mode_s"][eph.cat[n, g], m] += tm
        if not got and not too_hit:
            acc["wasted_s"] += tm
        else:
            acc["am_sum"] += Xo
            acc["am_n"] += 1
        if log is not None:
            dom = int(np.argmax(sv.prio * o["dU"][:, c])) if got else -1
            log["exps"].append(dict(t=t / 3600, dur=(tm + ovh) / 3600, mode=m, prog=dom,
                                    too=too_hit, am=Xo, tile=f))
        last_pos = f
        t += tm + ovh


def run(sv, eph, policy, start_night, n_nights, seed, state=None, log_nights=(),
        snapshot_every=None, prog_subset=None, record_every=1):
    st = sv.new_state() if state is None else state.copy()
    rng = np.random.default_rng([int(seed), int(start_night), 99])
    acc = _new_acc()
    score0 = sv.score(st)
    hist_n, hist_score, hist_U, hist_C = [], [], [], []
    snaps, logs = [], {}
    for k in range(n_nights):
        n = start_night + k
        log = {} if k in log_nights else None
        run_night(sv, eph, policy, st, n, seed, acc, rng, log=log, prog_subset=prog_subset)
        if log is not None:
            logs[k] = log
        if (k + 1) % record_every == 0 or k == n_nights - 1:
            U = sv.utilities(st)
            hist_n.append(k + 1)
            hist_U.append(U)
            hist_C.append(sv.completeness(st))
            hist_score.append(sv.score(st, U))
        if snapshot_every and (k + 1) % snapshot_every == 0:
            snaps.append((n + 1, st.copy()))
    return dict(
        policy=policy.kind, seed=seed, start=start_night, n_nights=n_nights,
        score0=score0, score=hist_score[-1], gain=hist_score[-1] - score0,
        hist_n=np.array(hist_n), hist_score=np.array(hist_score),
        hist_U=np.array(hist_U), hist_C=np.array(hist_C),
        acc=acc, too_done=st.too_done, too_total=st.too_total,
        state=st, snaps=snaps, logs=logs,
    )


# ----------------------------------------------------------------------------
# Analytic Poisson pass model (Payerne et al. 2026, Sec. 5.2 / 6.1)
# ----------------------------------------------------------------------------
def poisson_pass_model(density, t_req, n_tiles, f_cat, n_fibres=FIBRES["LR"],
                       f_obs=0.8, hours_per_night=12.0, n_pass=40):
    """Completeness after p passes and the calendar years they take."""
    N = density * FOV_DEG2
    R = N
    comp = [0.0]
    for _ in range(n_pass):
        R = R * np.exp(-n_fibres / R)
        comp.append(1 - R / N)
    n_exp = np.ceil(t_req / MODE_T[1]) if t_req > MODE_T[0] else 1
    t_exp = MODE_T[1] if t_req > MODE_T[0] else MODE_T[0]
    t_pass_h = n_tiles * n_exp * (t_exp + CONFIG_OVERHEAD_S) / 3600
    hours_per_year = 365.25 * hours_per_night * f_obs * f_cat
    years = np.arange(n_pass + 1) * t_pass_h / hours_per_year
    return years, np.array(comp)


# ----------------------------------------------------------------------------
# Parallel execution
# ----------------------------------------------------------------------------
_G = {}


def _init_worker(sv, eph):
    _G["sv"], _G["eph"] = sv, eph


def _job(args):
    policy, start, n_nights, seed, state, kw = args
    kw = dict(kw)
    keep = kw.pop("keep_state", False)
    r = run(_G["sv"], _G["eph"], policy, start, n_nights, seed, state=state, **kw)
    if not keep:
        r.pop("state", None)
        r["snaps"] = r["snaps"] if kw.get("snapshot_every") else []
    return r


def make_pool(sv, eph, n_workers=None):
    n_workers = n_workers or os.cpu_count() or 2
    return ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                               initargs=(sv, eph))


def run_many(pool, jobs):
    """jobs: list of (policy, start, n_nights, seed, state, kwargs-for-run).
    kwargs may include keep_state=True to return the final survey state."""
    return list(pool.map(_job, jobs))


# ----------------------------------------------------------------------------
# Learning: Cross-Entropy Method over the merit weights, on survey-state snapshots
# ----------------------------------------------------------------------------
def train_cem(pool, snapshots, n_iter=15, pop=16, elite_frac=0.25, n_eval=3,
              block_nights=28, seed=0):
    """snapshots: list of (night, State) taken from a reference multi-year run, so the
    learner sees early, middle and late survey states."""
    rng = np.random.default_rng(seed)
    mu = HANDTUNED_W.copy()
    sigma = np.ones(N_FEAT) * 0.7
    n_elite = max(2, int(pop * elite_frac))
    history = []
    for it in range(n_iter):
        pick = rng.choice(len(snapshots), n_eval, replace=False)
        seeds = rng.integers(1_000, 50_000, n_eval)
        W = mu + sigma * rng.standard_normal((pop, N_FEAT))
        W[0] = mu
        jobs = []
        for w in W:
            for j, sd in zip(pick, seeds):
                night, stt = snapshots[j]
                jobs.append((Policy("learned", w=w), night, block_nights, sd, stt, {}))
        for j, sd in zip(pick, seeds):
            night, stt = snapshots[j]
            jobs.append((Policy("greedy"), night, block_nights, sd, stt, {}))
        res = run_many(pool, jobs)
        gains = np.array([r["gain"] for r in res])
        ref = gains[-n_eval:]
        fit = (gains[:-n_eval].reshape(pop, n_eval) / np.maximum(ref, 1e-12)).mean(axis=1)
        elite = W[np.argsort(fit)[-n_elite:]]
        mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
        sigma = np.maximum(0.7 * elite.std(axis=0) + 0.3 * sigma, 0.05)
        history.append(dict(it=it, best=fit.max(), mean=fit.mean(), mu_fit=fit[0]))
        checkpoint(f"CEM iter {it + 1}/{n_iter}: best={fit.max():.3f}x greedy, "
                   f"mean={fit.mean():.3f}x, current mean policy={fit[0]:.3f}x")
    return mu, history
