"""Organoid runs from the UCSC pickle: condition, exclusion and metrics over the first 100 episodes.
Usage: python -I scripts/organoid_runs.py PATH/aggregated_experiment_data.pkl  -> out/organoid_runs.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from organoids.data import load, runs_table  # noqa: E402
from organoids.metrics import m_gain, m_top  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
R, lengths = runs_table(load(sys.argv[1]))
R["n"] = [min(len(lengths[k]), 100) for k in zip(R.exp_id, R.run)]
R["M_top"] = [m_top(lengths[k][:100]) for k in zip(R.exp_id, R.run)]
R["M_gain"] = [m_gain(lengths[k][:100]) for k in zip(R.exp_id, R.run)]
R.to_csv(OUT / "organoid_runs.csv", index=False)
K = R[~R.excluded]
print(f"{len(R)} runs, {int(R.excluded.sum())} excluded")
print(K.groupby(["paradigm", "condition"]).agg(runs=("run", "size"), organoids=("organoid", "nunique"),
                                              M_top=("M_top", "median"), M_gain=("M_gain", "median")))
