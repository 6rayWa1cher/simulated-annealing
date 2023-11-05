import math
import random


class SimulatedAnnealing:
    def __init__(self, f_func, k, t0, t_func, a_func, x_0, h_func, seed):
        self.a_func = a_func
        self.t_func = t_func
        self.f_func = f_func
        self.x_0 = x_0
        self.trace = list()
        self.k = k
        self.t0 = t0
        self.h_func = h_func
        self.seed = seed

    def optimize(self):
        random.seed(self.seed)
        e = self.f_func(self.x_0)
        x = self.x_0

        t = self.t0

        min_x0 = self.x_0
        min_e = e
        for i in range(1, self.k):
            if e < min_e:
                min_x0 = x
                min_e = e

            t = self.t_func(self.t0, t, i)

            while True:
                x_new = self.a_func(x, t)
                e_new = self.f_func(x_new)

                alpha = random.random()

                if alpha < self.h_func(e_new - e, t):
                    x = x_new
                    e = e_new
                    break

        return min_x0


def exp_h_func(delta_e, t):
    return math.exp(-delta_e / t)


def boltzman_t_func(t0, t, k):
    return t0 / math.log(1 + k)


def boltzman_a_func(x, t):
    return random.normalvariate(x, t)


def test_func_1(x):
    return x ** 4 + (x - 1.4) ** 3 + (x - 3) ** 2 + x - 1


def test_func_2(x):
    return x * math.sin(x) * math.cos(x) + 0.3 * x ** 2


def test_func_3(x):
    return 6 * math.sin(x) * math.cos((x - 4) / 2) if 0 <= x <= 4 * math.pi else 0


def score_one(optimizer_lambda, func, expected, x0):
    optimizer = optimizer_lambda(func, x0)
    x = optimizer.optimize()
    return abs(x - expected)


def score(optimizer_lambda, func_expected_x0_tpls):
    sum_scores = 0
    for func, expected, x0 in func_expected_x0_tpls:
        sum_scores += score_one(optimizer_lambda, func, expected, x0)
    return sum_scores


if __name__ == '__main__':
    test_cases = [(test_func_1, -1.74, 0), (test_func_2, 0, 20), (test_func_3, 4.568, 10)]
    print(score(
        lambda func, x0: SimulatedAnnealing(
            func, 10000, 100, boltzman_t_func, boltzman_a_func, x0, exp_h_func, 0
        ),
        test_cases
    ))
    print(score(
        lambda func, x0: SimulatedAnnealing(
            func, 500, 100, boltzman_t_func, boltzman_a_func, x0, exp_h_func, 0
        ),
        test_cases
    ))
