"""Train agents (PPO) and save per-episode lengths.
Usage: python -I scripts/train.py AGENT --seeds 0-9 --lr 3e-3 --epu 4 [--episodes 100] [--min-steps 0] [--workers 8]
       python -I scripts/train.py AGENT --tune [--seeds 9000-9004]   (grid: lr {3e-4,1e-3,3e-3} x epu {1,4,16})
AGENT: mlp gru snn snn5 snn10 snn5x2 res5x2, *_out0 (small output init), if_mlp if_gru if_snn5 (organoid interface).
Output: out/runs/<agent>_lr<lr>_epu<epu>_e<episodes>_s<seed>.npz"""
import argparse
import os
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

RUNS = ROOT / "out" / "runs"


def job(a):
    agent, lr, epu, ep, min_steps, seed = a
    p = RUNS / f"{agent}_lr{lr:g}_epu{epu}_e{ep}_s{seed}.npz"
    if not p.exists():
        import torch
        torch.set_num_threads(1)
        from organoids import interface, ppo
        r = (interface.run(agent, seed, ep, lr, epu) if agent.startswith("if_")
             else ppo.run(agent, seed, ep, lr, epu, min_steps))
        np.savez(p, **r)
    return p


def seeds(s):
    a, _, b = s.partition("-")
    return range(int(a), int(b or a) + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epu", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--min-steps", type=int, default=0)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    grid = product([3e-4, 1e-3, 3e-3], [1, 4, 16]) if a.tune else [(a.lr, a.epu)]
    jobs = [(a.agent, lr, epu, a.episodes, a.min_steps, s) for lr, epu in grid for s in seeds(a.seeds)]
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(a.workers) as pool:
        paths = pool.map(job, jobs)
    if a.tune:
        obj = {}
        for (_, lr, epu, *_), p in zip(jobs, paths):
            obj.setdefault((lr, epu), []).append(np.load(p)["lengths"].mean())
        best = max(obj, key=lambda k: (np.mean(obj[k]), -k[0], k[1]))
        for k, v in sorted(obj.items()):
            print(f"lr {k[0]:g} epu {k[1]:2d}: mean episode length {np.mean(v):.1f}")
        print(f"selected: lr {best[0]:g}, epu {best[1]}")
    else:
        print(f"saved {len(paths)} runs to {RUNS}")
