
import ast
import pickle

import numpy as np
import pandas as pd

ALLOWED = {("numpy", "dtype"), ("numpy._core.multiarray", "scalar"), ("numpy._core.numeric", "_frombuffer")}
COND = {"none": "nostim", "random": "random", "rl": "adaptive"}   # "nostim" = null (pandas reads "null" as NaN)
PHARMA = {"23138_RL_pharma", "p001237b_RL_pharma"}                    # drug experiments, excluded
BAD_ELECTRODES = {("23133_exp1", None), ("22194_baseline", 0), ("22194_baseline", 1), ("22194_baseline", 2)}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("blocked persistent id")


def load(path):
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def electrodes(v):
    """'[18422, 20846]' or '[[17998], [17549]]' -> flat list of ints (literal parsing only)."""
    try:
        v = ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        return None
    if isinstance(v, (int, float)):
        return [int(v)]
    return [int(y) for x in v for y in (x if isinstance(x, (list, tuple)) else [x])]


def runs_table(data):
    """One row per run (condition, paradigm, organoid, exclusion) and the episode lengths of every run."""
    rows, lengths = [], {}
    for e in data.values():
        exp = e["exp_id"]
        for i, r in enumerate(e["cartpole"]["runs"]):
            sizes = [len(x) for x, tag in ((electrodes(el), tg) for el, tg in
                                           zip(r["stim_log"]["stim_electrodes"], r["stim_log"]["tag"]))
                     if tag == "train" and x is not None]
            cond = COND[r["exp_type"]]
            label_ok = (not sizes if cond == "nostim" else
                        bool(sizes) and all(s == 2 for s in sizes) if cond == "adaptive" else
                        bool(sizes) and all(s in (4, 5) for s in sizes))
            bad = exp in PHARMA or (exp, None) in BAD_ELECTRODES or (exp, i) in BAD_ELECTRODES or not label_ok
            lengths[(exp, i)] = np.asarray(r["reward_log"]["reward"], float)
            rows.append(dict(exp_id=exp, run=i, organoid=e["metadata"]["organoid"], paradigm=e["metadata"]["paradigm"],
                             condition=cond, excluded=bad))
    return pd.DataFrame(rows), lengths


def episode_actions(run):
    """Logged actions of one run, split into episodes (non-finite actions -> 0)."""
    gl, rl = run["game_log"], run["reward_log"]
    t = np.asarray(gl["time"], float)
    a = np.asarray([x if not isinstance(x, str) else np.nan for x in gl["action"]], float)
    idx = np.searchsorted(np.asarray(rl["time"], float), t, side="left")
    return [np.nan_to_num(a[idx == k], nan=0.0, posinf=0.0, neginf=0.0) for k in range(len(rl["time"]))]
