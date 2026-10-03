import numpy as np
import pandas as pd
from src import heuristics as H
from src.env import FloodAllocEnv, ZMAX
from src.resources import load_zones, load_units


def build_policies(n_zones=None):
    pols = {k: (lambda e, o, r, f=f: f(e, r)) for k, f in H.POLICIES.items()}
    try:
        from stable_baselines3 import PPO
        m = PPO.load("models/ppo_alloc")
        if m.observation_space.shape[0] != 5 * ZMAX + 4:
            raise ValueError("Old PPO model, retrain with python -m src.train_rl")
        if n_zones is not None and n_zones > ZMAX:
            raise ValueError(f"PPO supports at most {ZMAX} zones")
        pols["ppo"] = lambda e, o, r: int(m.predict(o, deterministic=True)[0])
    except Exception as ex:
        print("PPO not loaded:", ex)
    return pols


def run(env, pol, seed):
    obs, _ = env.reset(seed=seed)
    rng, done = np.random.default_rng(seed), False
    while not done:
        obs, _, done, _, _ = env.step(pol(env, obs, rng))
    return env.metrics()


if __name__ == "__main__":
    zones = load_zones()
    env = FloodAllocEnv(zones, load_units(), randomize=True)
    rows = [{"policy": n, **run(env, p, s)}
            for n, p in build_policies(len(zones)).items() for s in range(1000, 1200)]
    res = pd.DataFrame(rows).groupby("policy").mean().round(3)
    res.to_csv("results/rl_vs_heuristics.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(res)
