"""The organoids' stimulation and recording interface, reimplemented from the UCSC experiment code, and agents that
act through it.

Encoder: two sensory electrodes stimulated periodically at ((-/+ sin(theta) + 0.15) * 7)^2 Hz, counted per 200 ms
decision. Decoder: motor spike counts c_k, r_k <- 0.2 r_k + 0.8 c_k, force = clip(-(r_0 - r_1), -1, 1); r persists
across the episodes of a run (reproduces every logged action from the logged spike counts).
Interface agents see only the two pulse counts and emit two motor firing rates (softplus); spike counts are sampled
Poisson and decoded. PPO as organoids/ppo.py with the Poisson log-probability of the sampled counts.
"""
import time

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from . import ppo
from .env import CartPole


class Encoder:
    def __init__(self):
        self.ph = np.zeros(2)

    def reset(self):
        self.ph = np.zeros(2)

    def __call__(self, theta):
        s = np.sin(theta)
        self.ph = self.ph + np.array([((-s + 0.15) * 7.0) ** 2, ((s + 0.15) * 7.0) ** 2]) * 0.2
        n = np.floor(self.ph)
        self.ph = self.ph - n
        return n


class Decoder:
    def __init__(self):
        self.r = np.zeros(2)

    def __call__(self, counts):
        self.r = 0.2 * self.r + 0.8 * np.asarray(counts, float)
        return float(np.clip(-(self.r[0] - self.r[1]), -1.0, 1.0))


class IfMLP(nn.Module):
    def __init__(self, hidden=63):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 3)

    def init_state(self, b):
        return None

    def step(self, x, state):
        return self.head(self.net(torch.cat([x, x if state is None else state], -1))), x

    def sequence(self, X):
        return self.head(self.net(torch.cat([X, torch.cat([X[:, :1], X[:, :-1]], 1)], -1)))


class IfGRU(nn.Module):
    def __init__(self, hidden=36):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        self.hidden = hidden

    def init_state(self, b):
        return torch.zeros(1, b, self.hidden)

    def step(self, x, state):
        y, h = self.gru(x[:, None, :], state)
        return self.head(y[:, 0]), h

    def sequence(self, X):
        return self.head(self.gru(X)[0])


class IfSNN5(nn.Module):
    """SNN-K5 on the two pulse counts, small output initialisation."""
    K = 5

    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.fc_in = nn.Linear(2, n)
        self.lif = snn.RLeaky(beta=torch.full((n,), 0.9), linear_features=n, all_to_all=True, threshold=1.0,
                              spike_grad=surrogate.fast_sigmoid(slope=25.0), reset_mechanism="subtract",
                              learn_beta=True, learn_threshold=False)
        self.readout = nn.Linear(n, 3)
        self.beta_out = nn.Parameter(torch.full((3,), 0.8))
        with torch.no_grad():
            self.readout.weight.mul_(0.01)
            self.readout.bias.zero_()

    def init_state(self, b):
        return (torch.zeros(b, self.n), torch.zeros(b, self.n), torch.zeros(b, 3))

    def step(self, x, state):
        spk, mem, m = state
        cur, bo = self.fc_in(x), self.beta_out.clamp(0.0, 1.0)
        for _ in range(self.K):
            spk, mem = self.lif(cur, spk, mem)
            m = bo * m + self.readout(spk)
        return m, (spk, mem, m)

    def sequence(self, X):
        state, outs = self.init_state(X.shape[0]), []
        for t in range(X.shape[1]):
            m, state = self.step(X[:, t], state)
            outs.append(m)
        return torch.stack(outs, 1)


AGENTS = {"if_mlp": IfMLP, "if_gru": IfGRU, "if_snn5": IfSNN5}


def _rates(out):
    return torch.nn.functional.softplus(out[..., :2]) + 1e-4


def run(agent, seed, n_episodes, lr, epu):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env, gen = CartPole(seed=seed), torch.Generator().manual_seed(seed + 1)
    model = AGENTS[agent]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    enc, dec, lengths, buf, t0 = Encoder(), Decoder(), [], [], time.time()
    for _ in range(n_episodes):
        obs, state = env.reset(), model.init_state(1)
        enc.reset()
        X, C, LP, V, R = [], [], [], [], []
        with torch.no_grad():
            while True:
                x = torch.tensor(enc(obs[2]), dtype=torch.float32)[None]
                out, state = model.step(x, state)
                lam = _rates(out)[0]
                c = torch.poisson(lam, generator=gen)
                lp = torch.distributions.Poisson(lam).log_prob(c).sum()
                obs, r, done = env.step(dec(c.numpy()))
                X.append(x[0].numpy()); C.append(c.numpy()); LP.append(float(lp)); V.append(float(out[0, 2])); R.append(r)
                if done:
                    last_val = 0.0
                    break
                if len(X) >= ppo.CAP:
                    ph = enc.ph.copy()
                    xv = torch.tensor(enc(obs[2]), dtype=torch.float32)[None]
                    enc.ph = ph
                    last_val = float(model.step(xv, state)[0][0, 2])
                    break
        lengths.append(len(X))
        buf.append((np.array(X), np.array(C), np.array(LP)) + ppo.gae(np.array(R) * ppo.RSCALE, np.array(V), last_val))
        if len(buf) >= epu:
            _update(model, opt, buf)
            buf = []
        if len(lengths) >= ppo.SOLVED_WIN and all(L >= ppo.CAP for L in lengths[-ppo.SOLVED_WIN:]):
            break
    L = np.r_[lengths, np.full(max(0, n_episodes - len(lengths)), ppo.CAP)].astype(int)
    return dict(lengths=L, steps=int(sum(lengths)), seconds=time.time() - t0)


def _pad2(seqs, T):
    out = torch.zeros(len(seqs), T, 2)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.float32)
    return out


def _update(model, opt, buf):
    T = max(len(b[0]) for b in buf)
    X, C = _pad2([b[0] for b in buf], T), _pad2([b[1] for b in buf], T)
    LP, ADV, RET = (ppo.pad([b[i] for b in buf], T) for i in (2, 3, 4))
    M = ppo.pad([np.ones(len(b[0])) for b in buf], T)
    n = M.sum()
    if n > 1:
        mean = (ADV * M).sum() / n
        ADV = (ADV - mean) / torch.sqrt((((ADV - mean) * M) ** 2).sum() / (n - 1)).clamp_min(1e-8) * M
    for _ in range(ppo.EPOCHS):
        out = model.sequence(X)
        ratio = torch.exp(torch.distributions.Poisson(_rates(out)).log_prob(C).sum(-1) - LP)
        pg = -torch.min(ratio * ADV, torch.clamp(ratio, 1 - ppo.CLIP, 1 + ppo.CLIP) * ADV)
        loss = ((pg + ppo.VCOEF * (out[..., 2] - RET) ** 2) * M).sum() / n
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.MAXGRAD)
        opt.step()
