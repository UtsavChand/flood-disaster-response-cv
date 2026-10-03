import numpy as np
import gymnasium as gym
from gymnasium import spaces

ZMAX = 12
KEYS = ["x", "y", "flood_frac", "population", "road_access", "hospital_km"]


def compute_need(z):
    load = z["population"] * z["flood_frac"]
    need = np.stack([
        np.ceil(load / 3000),
        np.ceil(load * (1 + z["hospital_km"] / 20) / 5000),
        np.ceil(load * (2 - z["road_access"]) / 4000),
    ], axis=1)
    sev = load * (1 + (1 - z["road_access"]) + z["hospital_km"] / 20)
    return need, sev / (sev.sum() + 1e-9)


def synth_zones(rng, n):
    ff = np.clip(rng.beta(2, 4, n), 0.02, 1)
    return {"x": rng.uniform(0, 50, n), "y": rng.uniform(0, 50, n), "flood_frac": ff,
            "population": rng.integers(3000, 40000, n).astype(float),
            "road_access": np.clip(1 - 0.7 * ff + rng.normal(0, 0.1, n), 0.05, 1),
            "hospital_km": rng.uniform(1, 25, n)}


def pad(z, zmax):
    n = len(z["x"])
    return {k: np.concatenate([z[k], np.full(zmax - n, 1e4 if k in ("x", "y") else 0.0)]) for k in KEYS}


class FloodAllocEnv(gym.Env):
    """Pass zones+units for a real scenario. Pass nothing to generate random training scenarios."""

    def __init__(self, zones=None, units=None, randomize=True, dscale=25.0, seed=0, zmax=ZMAX):
        self.zmax, self.Z = zmax, zmax
        self.randomize, self.dscale = randomize, dscale
        self.rng = np.random.default_rng(seed)
        self.units0 = units
        self.z0, self.n = None, 0
        if zones is not None:
            if len(zones) > zmax:
                raise ValueError(f"At most {zmax} zones supported")
            self.z0 = {k: zones[k].to_numpy(float) for k in KEYS}
            self.n = len(zones)
        self.action_space = spaces.Discrete(zmax)
        self.observation_space = spaces.Box(-1, 100, shape=(5 * zmax + 4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.z0 is None:
            self.n = int(self.rng.integers(3, self.zmax + 1))
            z = synth_zones(self.rng, self.n)
        else:
            z = {k: v.copy() for k, v in self.z0.items()}
            if self.randomize:
                z["flood_frac"] = np.clip(z["flood_frac"] * self.rng.lognormal(0, 0.3, self.n), 0.01, 1)
                z["population"] = z["population"] * self.rng.lognormal(0, 0.2, self.n)
        z = pad(z, self.zmax)
        self.xy = np.stack([z["x"], z["y"]], 1)
        self.need, self.sev = compute_need(z)
        self.rem = self.need.copy()
        if self.units0 is None:
            units = []
            for t in range(3):
                dx, dy = self.rng.uniform(0, 50, 2)
                units += [(t, float(dx), float(dy))] * int(self.rng.integers(4, 17))
        else:
            units = list(self.units0)
        self.units = [units[i] for i in self.rng.permutation(len(units))]
        self.i, self.log = 0, []
        return self._obs(), {}

    def _cur(self):
        return self.units[min(self.i, len(self.units) - 1)]

    @property
    def cur_type(self):
        return self._cur()[0]

    def dists(self):
        _, ux, uy = self._cur()
        return np.hypot(self.xy[:, 0] - ux, self.xy[:, 1] - uy)

    def _obs(self):
        rem = (self.rem / max(self.need.max(), 1)).ravel()
        d = np.minimum(self.dists() / 50.0, 5.0)
        return np.concatenate([self.sev * self.n, rem, d, np.eye(3)[self.cur_type],
                               [1 - self.i / len(self.units)]]).astype(np.float32)

    def step(self, a):
        t, d = self.cur_type, self.dists()[a]
        useful = self.rem[a, t] > 0
        if useful:
            self.rem[a, t] -= 1
            r = self.sev[a] * self.n * np.exp(-d / self.dscale)
        else:
            r = -0.1
        self.log.append((int(a), int(t), float(d), bool(useful)))
        self.i += 1
        return self._obs(), float(r), self.i >= len(self.units), False, {}

    def metrics(self):
        n = self.n
        need, rem, sev = self.need[:n], self.rem[:n], self.sev[:n]
        tot = max(need.sum(), 1)
        zc = 1 - rem.sum(1) / np.maximum(need.sum(1), 1)
        used = [l[2] for l in self.log if l[3]]
        return {
            "coverage": float((tot - rem.sum()) / tot),
            "severity_weighted_coverage": float((sev * zc).sum()),
            "worst_zone_coverage": float(zc.min()),
            "mean_travel_km": float(np.mean(used)) if used else 0.0,
            "wasted_units": len(self.log) - len(used),
        }
