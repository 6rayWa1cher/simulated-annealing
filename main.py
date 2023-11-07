import math
import random

import numpy as np
from tqdm import trange


class SimulatedAnnealing:
    def __init__(self, f_func, k, t0, t_func, a_func, x_0, h_func, seed, bound_left, bound_right,
                 skip_unsuccessful=False):
        self.a_func = a_func
        self.t_func = t_func
        self.f_func = f_func
        self.x_0 = x_0
        self.trace = list()
        self.k = k
        self.t0 = t0
        self.h_func = h_func
        self.seed = seed
        self.bounds = bound_left, bound_right
        self.skip_unsuccessful = skip_unsuccessful

    def optimize(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        e = self.f_func(self.x_0)
        x = self.x_0

        t = self.t0

        min_x0 = self.x_0
        min_e = e

        self.trace = np.zeros((self.k, 4))
        for i in range(1, self.k + 1):
            self.trace[i - 1] = x, e, min_x0, min_e
            if e < min_e:
                min_x0 = x
                min_e = e

            t = self.t_func(self.t0, t, i)

            first = True
            while first or not self.skip_unsuccessful:
                x_new = self.a_func(x, t, self.bounds)
                e_new = self.f_func(x_new)

                alpha = random.random()

                if alpha < self.h_func(e_new - e, t):
                    x = x_new
                    e = e_new
                    break
                first = False

        return min_x0


def exp_h_func(delta_e, t):
    if delta_e < 0:
        return 1
    if -delta_e / t > 0:
        return 1
    return math.exp(-delta_e / t)


def boltzman_t_func(t0, t, k):
    return t0 / math.log(1 + k)


def boltzman_a_func(x, t, bounds):
    return random.normalvariate(x, t)


def cauchy_t_func(t0, t, k):
    return t0 / math.sqrt(k)


def cauchy_a_func(x, t, bounds):
    alpha = random.random()
    return t * math.tan(math.pi * alpha - math.pi / 2)


def make_superfast_t_func(m, n):
    c = max(m * math.exp(-n), 1e-300)
    d = math.exp(-c)

    def func(t0, t, k):
        return max(t * d, 1e-300)

    return func


def sgn(x):
    return 1 if x >= 0 else -1


def superfast_a_func(x, t, bounds):
    a, b = bounds
    x_new = b + b
    while not a <= x_new <= b:
        alpha = random.random()
        z = sgn(alpha - 0.5) * t * ((1 + 1 / t) ** abs(2 * alpha - 1) - 1)
        x_new = x + z * (b - a)
    return x_new


def boltzman_t_exp(t0, t, k):
    return t * 0.99


def superfast_t_exting(m, n, q):
    c = m * np.exp(-n)

    def func(t0, t, k):
        return max(t0 * np.exp(-c * (k ** q)), 1e-300)

    return func


def test_func_1(x):
    return x ** 4 + (x - 1.4) ** 3 + (x - 3) ** 2 + x - 1


def test_func_2(x):
    return x * math.sin(x) * math.cos(x) + 0.3 * x ** 2


def test_func_3(x):
    return 6 * math.sin(x) * math.cos((x - 4) / 2) if 0 <= x <= 4 * math.pi else 0


def test_func_4(x):
    if not (-3 * math.pi <= x <= 3 * math.pi):
        return 0
    return 6 * math.sin(x) * math.cos((x - 4) / 2) * math.sin((x - 6) / 5) * math.cos((x ** 2 - 3) / 6)


def test_func_5(x):
    if not (-3 * math.pi <= x <= 3 * math.pi):
        return 0
    return 6 * math.sin(x) * math.cos((x - 4) / 2) * math.sin((x - 6) / 5) * math.cos((x ** 2 - 3) / 6) * x


def score_one(optimizer_lambda, func, expected, x0, a, b):
    optimizer = optimizer_lambda(func, x0, a, b)
    x = optimizer.optimize()
    # print(x, func(x))
    return abs(x - expected)


def score(optimizer_lambda, func_expected_x0_a_b_tpls):
    scores = []
    for func, expected, x0, a, b in func_expected_x0_a_b_tpls:
        scores.append(score_one(optimizer_lambda, func, expected, x0, a, b))
    return scores, sum(scores)


if __name__ == '__main__':
    test_cases = [
        (test_func_1, -1.74, 0, -5, 5),
        (test_func_2, 0, 20, -100, 100),
        (test_func_3, 4.568, 10, 0, 4 * math.pi),
        (test_func_4, -1.722, -8, -3 * math.pi, 3 * math.pi),
        (test_func_5, 8.798, -9, -3 * math.pi, 3 * math.pi)
    ]
    methods = [
        ('boltzman', boltzman_t_func, boltzman_a_func, 25),
        ('cauchy', cauchy_t_func, cauchy_a_func, 25),
        ('superfast_1_-160', make_superfast_t_func(1, -160), superfast_a_func, 1e-7)
    ]
    for method_name, t_func, a_func, t0 in methods:
        for k in (500, 2000, 10000, 50000):
            print('method', method_name, 'k', k)
            s = score(
                lambda func, x0, a, b: SimulatedAnnealing(func, k, t0, t_func, a_func, x0, exp_h_func, 0, a, b),
                test_cases)
            print('scoring', s[1])
