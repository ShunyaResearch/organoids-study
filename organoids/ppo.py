
import time

import numpy as np
import torch

from .agents import AGENTS
from .env import THETA_LIMIT, CartPole

CAP, SOLVED_WIN = 2000, 20
GAMMA, LAM, CLIP, VCOEF, MAXGRAD, EPOCHS, RSCALE = 0.99, 0.95, 0.2, 0.5, 0.5, 4, 0.01


def gae(rew, val, last_val):
    adv, g = np.zeros(len(rew)), 0.0
    for t in reversed(range(len(rew))):
        nv = val[t + 1] if t + 1 < len(rew) else last_val
        g = rew[t] + GAMMA * nv - val[t] + GAMMA * LAM * g
        adv[t] = g
    return adv, adv + val


def pad(seqs, T):
    out = torch.zeros(len(seqs), T)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.float32)
    return out


def run(agent, seed, n_episodes, lr, epu, min_steps=0):
    """Train one agent from scratch; returns per-episode lengths. min_steps > 0 continues past n_episodes until that
    many environment steps have been used (step-matched budget)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env, gen = CartPole(seed=seed), torch.Generator().manual_seed(seed + 1)
    model = AGENTS[agent]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lengths, buf, t0 = [], [], time.time()
    while len(lengths) < n_episodes or sum(lengths) < min_steps:
        obs, state = env.reset(), model.init_state(1)
        X, A, LP, V, R = [], [], [], [], []
        with torch.no_grad():
            while True:
                x = torch.tensor([obs[2] / THETA_LIMIT], dtype=torch.float32)
                mu, v, state = model.step(x, state)
                std = model.log_std.exp()
                a = mu + std * torch.randn(1, generator=gen)
                lp = torch.distributions.Normal(mu, std).log_prob(a)
                obs, r, done = env.step(float(a))
                X.append(float(x)); A.append(float(a)); LP.append(float(lp)); V.append(float(v)); R.append(r)
                if done:
                    last_val = 0.0
                    break
                if len(X) >= CAP:
                    last_val = float(model.step(torch.tensor([obs[2] / THETA_LIMIT], dtype=torch.float32), state)[1])
                    break
        lengths.append(len(X))
        buf.append((X, A, LP) + gae(np.array(R) * RSCALE, np.array(V), last_val))
        if len(buf) >= epu:
            update(model, opt, buf)
            buf = []
        if len(lengths) >= SOLVED_WIN and all(L >= CAP for L in lengths[-SOLVED_WIN:]):
            break
    L = np.r_[lengths, np.full(max(0, n_episodes - len(lengths)), CAP)].astype(int)
    return dict(lengths=L, steps=int(sum(lengths)), seconds=time.time() - t0)


def update(model, opt, buf):
    T = max(len(b[0]) for b in buf)
    X, A, LP, ADV, RET = (pad([b[i] for b in buf], T) for i in range(5))
    M = pad([np.ones(len(b[0])) for b in buf], T)
    n = M.sum()
    if n > 1:
        mean = (ADV * M).sum() / n
        ADV = (ADV - mean) / torch.sqrt((((ADV - mean) * M) ** 2).sum() / (n - 1)).clamp_min(1e-8) * M
    for _ in range(EPOCHS):
        mu, v = model.sequence(X, M)
        dist = torch.distributions.Normal(mu, model.log_std.exp())
        ratio = torch.exp(dist.log_prob(A) - LP)
        pg = -torch.min(ratio * ADV, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV)
        loss = ((pg + VCOEF * (v - RET) ** 2) * M).sum() / n
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAXGRAD)
        opt.step()
