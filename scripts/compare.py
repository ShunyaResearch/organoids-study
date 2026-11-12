import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from organoids.metrics import m_gain, m_top  # noqa: E402
from organoids.stats import compare, holm  # noqa: E402

R = pd.read_csv(ROOT / "out" / "organoid_runs.csv")
O = R[~R.excluded & (R.paradigm == "cycled") & (R.condition == "adaptive")]
rng, rows = np.random.default_rng(0), []
for pattern in sys.argv[1:]:
    L = [np.load(p)["lengths"][:100] for p in sorted((ROOT / "out" / "runs").glob(pattern + ".npz"))]
    for name, f in (("M_top", m_top), ("M_gain", m_gain)):
        rows.append(dict(agent=pattern, metric=name, n=len(L), median=np.median([f(x) for x in L]),
                         **compare([f(x) for x in L], O[name].values, O.organoid.values, rng)))
S = pd.DataFrame(rows)
S["p_holm"] = holm(S.p)
S["claimed"] = (S.p_holm < 0.05) & ((S.lo > 0) | (S.hi < 0))
print(f"organoids: {len(O)} runs, {O.organoid.nunique()} organoids, M_top {O.M_top.median():.1f}, M_gain {O.M_gain.median():.1f}")
print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
