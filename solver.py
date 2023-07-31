import numpy as np
import math
from functools import lru_cache
from itertools import combinations


class Solution:

    def __init__(self, value, alpha):
        self.value = value
        self.alpha = alpha

    def extend(self, alice_gets_coin, coin, idx):
        copy = Solution(self.value, self.alpha.copy())
        if alice_gets_coin:
            copy.value += coin
            copy.alpha.insert(idx, 1)
        else:
            copy.alpha.insert(idx, 0)
        return copy

    def __lt__(self, other):
        return (self.value, self.alpha) < (other.value, other.alpha)

    def __str__(self):
        return f"{self.value}: {self.alpha}"

    @staticmethod
    def max(x, y):
        if x is None:
            return y
        if y is None:
            return x

        return y if x < y else x

    @staticmethod
    def min(x, y):
        if x is None:
            return y
        if y is None:
            return x

        return x if x < y else y


@lru_cache(maxsize=None)
def __g(a, b, C, strategy_stealing, fixed_decisions):
    """
    Returns the optimal solution of a given game if both players
    play optimally.

    Args:
        a: a capacity of Alice's buffer
        b: a capacity of Bob's buffer
        C: a sorted tuple of all the remaining coins on the table
    """

    # termination check
    if a == 0 or b == 0:
        if a != 0:
            return Solution(value=sum(C), alpha=[1] * len(C))
        else:
            return Solution(value=0, alpha=[0] * len(C))

    # explicit solutions
    if not strategy_stealing and fixed_decisions is None:
        if a == 1:
            idx = b // 2
            alpha = [0] * len(C)
            alpha[idx] = 1
            return Solution(value=C[idx], alpha=alpha)

        if b == 1:
            idx = math.ceil(a / 2)
            alpha = [1] * len(C)
            alpha[idx] = 0
            return Solution(value=sum(C) - C[idx], alpha=alpha)

    C = list(C)
    is_alices_turn = len(C) % 2 == 0
    outer_fn, inner_fn = (Solution.max, Solution.min) if is_alices_turn else (
        Solution.min, Solution.max)
    outer_solution = None
    for idx, coin in enumerate(C):

        if strategy_stealing and not is_alices_turn and C.count(coin) % 2 == 0:
            continue

        sub_C = C.copy()
        del sub_C[idx]

        if fixed_decisions is not None:
            fixed_sub_decisions = list(fixed_decisions)
            del fixed_sub_decisions[idx]
            fixed_sub_decisions = tuple(fixed_sub_decisions)
        else:
            fixed_sub_decisions = None

        inner_solution = None
        for alice_gets_coin in (True, False):

            if fixed_decisions is not None:
                if (fixed_decisions[idx] !=  alice_gets_coin) ^ is_alices_turn:
                    continue

            sub_a, sub_b = (a - 1, b) if alice_gets_coin else (a, b - 1)
            sub_solution = __g(sub_a, sub_b, tuple(sub_C), strategy_stealing, fixed_sub_decisions if not is_alices_turn else None)
            sub_solution = sub_solution.extend(alice_gets_coin, coin, idx)
            inner_solution = inner_fn(inner_solution, sub_solution)

        outer_solution = outer_fn(outer_solution, inner_solution)

    return outer_solution


def g(a, b, C, strategy_stealing=False, bob_decisions=None):
    """
    Returns the optimal solution of a given game if both players
    play optimally.

    Args:
        a: a capacity of Alice's buffer
        b: a capacity of Bob's buffer
        C: a sorted list of all the remaining coins on the table
    """
    assert a + b == len(C), "The sum `a + b` must be equal to the size of C."
    s = __g(a, b, tuple(C), strategy_stealing, bob_decisions)
    __g.cache_clear()
    return s


def get_vectors(length, ones):

    result = []
    for subset_idx in combinations(range(length), int(ones)):
        vector = np.zeros(length, np.int32)
        vector[list(subset_idx)] = 1
        result.append(list(vector))

    return result


def vectors_to_solutions(C, vectors):

    solutions = []
    for vector in vectors:
        value = np.array(C) @ np.array(vector)
        solutions.append(Solution(value=value, alpha=vector))

    return solutions


def rank(C, solution):
    """
    Returns the rank of a given solution in a game with given coins.

    Args:
        C: a sorted list of coins on the table
        solution: an solution of a given game
    """

    vectors = get_vectors(len(C), sum(solution.alpha))
    solutions = vectors_to_solutions(C, vectors)

    rank_value = 0
    for s in solutions:
        if s < solution:
            rank_value += 1

    return rank_value


def is_mirrored(C):
    l = len(C)
    sums = [C[i] + C[l - i - 1] for i in range(l)]
    return min(sums) == max(sums)


if __name__ == "__main__":
    # solution = g(a=2, b=2, C=(1, 2, 3, 4))
    print(g(a=3, b=3, C=[0, 0, 1, 1, 2, 2], strategy_stealing=True).value)
