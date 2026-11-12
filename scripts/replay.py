"""Open-loop replay: each organoid run's own logged actions (first 100 episodes) are played without feedback from new
initial states (5 environment seeds); replay episode i starts with the actions of logged episode i and continues into
the next ones. A run whose actions depend on feedback should beat its replay.
Usage: python -I scripts/replay.py PATH/aggregated_experiment_data.pkl   (needs out/organoid_runs.csv)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from organoids.data import episode_actions, load  # noqa: E402
from organoids.env import CartPole  # noqa: E402
from organoids.metrics import m_top  # noqa: E402
from organoids.stats import cluster_weights, wmedian  # noqa: E402


def replay(eps, seed):
    stream, starts = np.concatenate(eps), np.cumsum([0] + [len(a) for a in eps[:-1]])
    env, out = CartPole(seed=seed), []
    for s in starts:
        env.reset()
        t = 0
        while True:
            _, _, done = env.step(float(stream[(s + t) % len(stream)]))
            t += 1
            if done or t >= 2000:
                break
        out.append(t)
    return np.array(out, float)


data = load(sys.argv[1])
runs = {(e["exp_id"], i): r for e in data.values() for i, r in enumerate(e["cartpole"]["runs"])}
R = pd.read_csv(ROOT / "out" / "organoid_runs.csv")
R = R[~R.excluded].reset_index(drop=True)
R["replay_M_top"] = [np.median([m_top(replay(episode_actions(runs[(r.exp_id, r.run)])[: r.n], 1000 * j + k))
                                for j in range(5)]) for k, r in enumerate(R.itertuples())]
R["diff"] = R.M_top - R.replay_M_top
R.to_csv(ROOT / "out" / "replay.csv", index=False)
rng = np.random.default_rng(0)
for (p, c), X in R.groupby(["paradigm", "condition"]):
    bd = wmedian(X["diff"].values, cluster_weights(X.organoid.values, rng))
    print(f"{p:10s} {c:8s} own - replay: median {X['diff'].median():5.1f} "
          f"[{np.percentile(bd, 2.5):.1f}, {np.percentile(bd, 97.5):.1f}]")
