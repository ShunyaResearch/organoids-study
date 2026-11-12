"""Actor-critic policies. step(x, state) -> (mu, value, state) for one decision; sequence(X, mask) -> (mu, value) over
padded episodes (state reset at t = 0). x is the angle scaled by the 16 deg limit. All weights are trained by PPO.

  mlp    MLP 2-63-63 tanh on [theta_t, theta_{t-1}]          gru    GRU, 36 units, on theta_t
  snn    64 recurrent LIF neurons, 1 update per decision, leaky nonspiking readout (decay 0.8)
  snn5 / snn10   5 / 10 LIF updates per decision, learnable per-neuron and readout decays
  snn5x2 SNN-K5 on [theta_t, theta_{t-1}]                    res5x2 snn5x2 with input/recurrent weights frozen
  <name>_out0    the same spiking network with output weights x0.01 and output bias 0 at initialisation
"""
import math

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

LOG_STD0 = math.log(0.5)


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.tensor(LOG_STD0))


class MLP(_Base):
    def __init__(self, hidden=63):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 2)

    def init_state(self, b):
        return None

    def step(self, x, state):
        prev = x if state is None else state
        out = self.head(self.net(torch.stack([x, prev], -1)))
        return out[..., 0], out[..., 1], x

    def sequence(self, X, mask):
        out = self.head(self.net(torch.stack([X, torch.cat([X[:, :1], X[:, :-1]], 1)], -1)))
        return out[..., 0], out[..., 1]


class GRU(_Base):
    def __init__(self, hidden=36):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.hidden = hidden

    def init_state(self, b):
        return torch.zeros(1, b, self.hidden)

    def step(self, x, state):
        y, h = self.gru(x[:, None, None], state)
        out = self.head(y[:, 0])
        return out[:, 0], out[:, 1], h

    def sequence(self, X, mask):
        out = self.head(self.gru(X[:, :, None])[0])
        return out[..., 0], out[..., 1]


class _Seq(_Base):
    """Shared loop: subclasses define init_state and _one(x, state) -> (out, state)."""
    def step(self, x, state):
        m, state = self._one(x, state)
        return m[:, 0], m[:, 1], state

    def sequence(self, X, mask):
        state, outs = self.init_state(X.shape[0]), []
        for t in range(X.shape[1]):
            m, state = self._one(X[:, t], state)
            outs.append(m)
        out = torch.stack(outs, 1)
        return out[..., 0], out[..., 1]


class SNN(_Seq):
    BETA_OUT = 0.8

    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.fc_in = nn.Linear(1, n)
        self.lif = snn.RLeaky(beta=0.9, linear_features=n, all_to_all=True, threshold=1.0,
                              spike_grad=surrogate.fast_sigmoid(slope=25.0), reset_mechanism="subtract",
                              learn_beta=False, learn_threshold=False)
        self.readout = nn.Linear(n, 2)

    def init_state(self, b):
        return (torch.zeros(b, self.n), torch.zeros(b, self.n), torch.zeros(b, 2))

    def _one(self, x, state):
        spk, mem, m = state
        spk, mem = self.lif(self.fc_in(x[:, None]), spk, mem)
        m = self.BETA_OUT * m + self.readout(spk)
        return m, (spk, mem, m)


class SNNK(_Seq):
    K = 5

    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.fc_in = nn.Linear(1, n)
        self.lif = snn.RLeaky(beta=torch.full((n,), 0.9), linear_features=n, all_to_all=True, threshold=1.0,
                              spike_grad=surrogate.fast_sigmoid(slope=25.0), reset_mechanism="subtract",
                              learn_beta=True, learn_threshold=False)
        self.readout = nn.Linear(n, 2)
        self.beta_out = nn.Parameter(torch.full((2,), 0.8))

    def init_state(self, b):
        return (torch.zeros(b, self.n), torch.zeros(b, self.n), torch.zeros(b, 2))

    def _run(self, cur, state):
        spk, mem, m = state
        bo = self.beta_out.clamp(0.0, 1.0)
        for _ in range(self.K):
            spk, mem = self.lif(cur, spk, mem)
            m = bo * m + self.readout(spk)
        return m, (spk, mem, m)

    def _one(self, x, state):
        return self._run(self.fc_in(x[:, None]), state)


class SNNK10(SNNK):
    K = 10


class SNNK2f(SNNK):
    """SNN-K5 on [theta_t, theta_{t-1}] (theta_{t-1} = theta_t at the first step)."""
    def __init__(self, n=64):
        super().__init__(n)
        self.fc_in = nn.Linear(2, n)

    def init_state(self, b):
        return super().init_state(b) + (None,)

    def step(self, x, state):
        prev = x if state[3] is None else state[3]
        m, st = self._run(self.fc_in(torch.stack([x, prev], -1)), state[:3])
        return m[:, 0], m[:, 1], st + (x,)

    def sequence(self, X, mask):
        P = torch.cat([X[:, :1], X[:, :-1]], 1)
        state, outs = SNNK.init_state(self, X.shape[0]), []
        for t in range(X.shape[1]):
            m, state = self._run(self.fc_in(torch.stack([X[:, t], P[:, t]], -1)), state)
            outs.append(m)
        out = torch.stack(outs, 1)
        return out[..., 0], out[..., 1]


class Reservoir(SNNK2f):
    """Input weights, recurrent weights and membrane decay fixed at their random initial values; readout trained."""
    def __init__(self, n=64):
        super().__init__(n)
        for p in list(self.fc_in.parameters()) + list(self.lif.parameters()):
            p.requires_grad_(False)


def small_output(cls):
    """Same network with output (mu, V) weights x0.01 and bias 0 at initialisation."""
    def __init__(self, n=64):
        cls.__init__(self, n)
        with torch.no_grad():
            self.readout.weight.mul_(0.01)
            self.readout.bias.zero_()
    return type(cls.__name__ + "Out0", (cls,), {"__init__": __init__})


AGENTS = {"mlp": MLP, "gru": GRU, "snn": SNN, "snn5": SNNK, "snn10": SNNK10, "snn5x2": SNNK2f, "res5x2": Reservoir}
AGENTS.update({k + "_out0": small_output(v) for k, v in list(AGENTS.items()) if k.startswith(("snn", "res"))})


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
