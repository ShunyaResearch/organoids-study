"""Continuous cart-pole identified from the organoid logs: frictionless equations (Florian 2005), explicit Euler,
20 ms step, force 10 N x clip(a, -1, 1), episode ends at |theta| > 16 deg, no cart bounds, state reset U(+-0.05)."""
import math

import numpy as np

G, M_CART, M_POLE, HALF_LEN, TAU, GAIN = 9.8, 1.0, 0.1, 0.5, 0.02, 10.0
THETA_LIMIT = math.radians(16.0)


class CartPole:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, th, th_dot = self.state
        f = GAIN * float(np.clip(action, -1.0, 1.0))
        total = M_CART + M_POLE
        tmp = (f + M_POLE * HALF_LEN * th_dot ** 2 * math.sin(th)) / total
        th_acc = (G * math.sin(th) - math.cos(th) * tmp) / (HALF_LEN * (4 / 3 - M_POLE * math.cos(th) ** 2 / total))
        x_acc = tmp - M_POLE * HALF_LEN * th_acc * math.cos(th) / total
        self.state = np.array([x + TAU * x_dot, x_dot + TAU * x_acc, th + TAU * th_dot, th_dot + TAU * th_acc])
        return self.state.copy(), 1.0, abs(self.state[2]) > THETA_LIMIT
