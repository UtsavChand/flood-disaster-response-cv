import numpy as np


def _mask(env):
    return env.rem[:, env.cur_type] > 0


def severity_first(env, rng):
    m = _mask(env)
    return int(np.argmax(np.where(m, env.sev, -1))) if m.any() else int(np.argmax(env.sev))


def nearest_first(env, rng):
    m, d = _mask(env), env.dists()
    return int(np.argmin(np.where(m, d, np.inf))) if m.any() else int(np.argmin(d))


def largest_need(env, rng):
    return int(np.argmax(env.rem[:, env.cur_type]))


def random_valid(env, rng):
    m = _mask(env)
    return int(rng.choice(np.flatnonzero(m) if m.any() else np.arange(env.Z)))


POLICIES = {"severity_first": severity_first, "nearest_first": nearest_first,
            "largest_need": largest_need, "random": random_valid}