"""Fixed reference policies (no learning), 30 seeds x 100 episodes, cap 2,000: zero force, uniform random force,
P and PD control on the scaled angle, and, through the organoid interface, no motor spikes and random spikes at the
organoid median rate (3.91 per decision, split over the two units).
Usage: python -I scripts/reference_policies.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from organoids.env import THETA_LIMIT, CartPole  # noqa: E402
from organoids.interface import Decoder  # noqa: E402
from organoids.metrics import m_gain, m_top  # noqa: E402

POLICIES = {"zero force": lambda x, xp, r: 0.0, "uniform random": lambda x, xp, r: r.uniform(-1, 1),
            "P, angle only": lambda x, xp, r: 0.5 * x, "PD, two frames": lambda x, xp, r: x + 10.0 * (x - xp)}


def run(name, seed):
    env, rng, dec, out = CartPole(seed=seed), np.random.default_rng(seed + 10_000), Decoder(), []
    if name.startswith("interface"):
        rng = np.random.default_rng(seed + 20_000)
    for _ in range(100):
        xp = x = env.reset()[2] / THETA_LIMIT
        t = 0
        while True:
            if name == "interface: no motor spikes":
                a = dec(np.zeros(2))
            elif name == "interface: random spikes":
                a = dec(rng.poisson(3.91 / 2, 2).astype(float))
            else:
                a = POLICIES[name](x, xp, rng)
            obs, _, done = env.step(a)
            t += 1
            if done or t >= 2000:
                break
            xp, x = x, obs[2] / THETA_LIMIT
        out.append(t)
    return np.array(out)


for name in list(POLICIES) + ["interface: no motor spikes", "interface: random spikes"]:
    L = [run(name, s) for s in range(30)]
    print(f"{name:28s} M_top {np.median([m_top(x) for x in L]):7.1f} | M_gain {np.median([m_gain(x) for x in L]):6.1f}")
