"""Per-run metrics from episode lengths (identical for organoids and agents)."""
import numpy as np


def m_top(L):
    """Peak performance: 90th percentile of episode length."""
    return float(np.percentile(L, 90)) if len(L) else np.nan


def m_gain(L, win=20):
    """Within-run improvement: mean of the last 20 episodes minus the first 20."""
    return float(np.mean(L[-win:]) - np.mean(L[:win])) if len(L) >= 2 * win else np.nan
