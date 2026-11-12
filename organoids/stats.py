"""Agent-vs-organoid comparison: difference in medians with a 95% two-stage bootstrap interval (organoids, then runs
within organoids; seeds for agents), Mann-Whitney test, Holm correction. A difference is claimed only if the interval
excludes 0 and the corrected test is below 0.05."""
import numpy as np
from scipy.stats import mannwhitneyu

NB = 10_000


def cluster_weights(org_ids, rng, nb=NB):
    org_ids = np.asarray(org_ids)
    uniq = np.unique(org_ids)
    idx = {o: np.flatnonzero(org_ids == o) for o in uniq}
    W = np.zeros((nb, len(org_ids)))
    for b in range(nb):
        for o in rng.choice(uniq, len(uniq)):
            np.add.at(W[b], rng.choice(idx[o], len(idx[o])), 1.0)
    return W


def seed_weights(n, rng, nb=NB):
    return rng.multinomial(n, np.ones(n) / n, size=nb).astype(float)


def wmedian(x, W):
    ok = ~np.isnan(x)
    x, W = x[ok], W[:, ok]
    o = np.argsort(x)
    c = np.cumsum(W[:, o], axis=1)
    return x[o][np.argmax(c >= c[:, -1:] / 2.0, axis=1)]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate([min(1.0, (len(p) - i) * p[j]) for i, j in enumerate(o)])
    out = np.empty(len(p))
    out[o] = adj
    return out


def compare(agent, organoid, organoid_ids, rng):
    """Returns median difference, 95% interval and Mann-Whitney p (agent minus organoids)."""
    agent, organoid = np.asarray(agent, float), np.asarray(organoid, float)
    d = wmedian(agent, seed_weights(len(agent), rng)) - wmedian(organoid, cluster_weights(organoid_ids, rng))
    d = d[~np.isnan(d)]
    a, o = agent[~np.isnan(agent)], organoid[~np.isnan(organoid)]
    return dict(diff=float(np.median(a) - np.median(o)), lo=float(np.percentile(d, 2.5, method="lower")),
                hi=float(np.percentile(d, 97.5, method="higher")),
                p=float(mannwhitneyu(a, o, alternative="two-sided").pvalue))
