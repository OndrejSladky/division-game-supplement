from solver import *
import pulp as lp

from pysat.formula import CNF
from pysat.pb import PBEnc
from pysat.solvers import Minisat22
import time


def geq(x, y):
    sumx = sumy = 0
    for i in range(len(x)):
        sumx += x[i]
        sumy += y[i]
        if sumx > sumy:
            return False

    return True


def downset(x):
    return set(tuple(y) for y in get_vectors(len(x), sum(x)) if geq(x, y))


def downsetunion(generators):
    ret = set()
    for g in generators:
        ret |= downset(g)
    return ret


def complement(alpha):
    return tuple(1 - x for x in alpha)


def complement_of_set(s):
    return set(complement(alpha) for alpha in s)


def clean_generators(gens):
    result = set(gens)
    for g in gens:
        if g in result:
            result -= downset(g)
            result.add(g)

    return result


def generate_all(generators, down, rem, half):
    if not rem:
        if len(down) == half:
            return [clean_generators(generators)]
        else:
            return []

    e = next(iter(rem))
    result = []
    for x in [e, complement(e)]:
        d = downset(x)
        new_down = down | d
        if len(new_down) > half:
            continue

        new_rem = (rem - d) - complement_of_set(d)
        new_gens = generators | {x}
        new_result = generate_all(new_gens, new_down, new_rem, half)
        result += new_result

    return result


def generate(n):
    vecs = set(tuple(x) for x in get_vectors(2 * n, n))
    return generate_all(set(), set(), vecs, len(vecs) // 2)


def cut_to_coins(cut, anticut):

    assert len(cut) > 0, "The cut must be non-empty."
    coins_n = len(cut[0])

    problem = lp.LpProblem("The_Cut_Problem")
    variables = [lp.LpVariable(f"coin_{coin_idx}")
                 for coin_idx in range(coins_n)]

    for prev_var, next_var in zip(variables, variables[1:]):
        problem += prev_var <= next_var, f"CoinsOrdering{prev_var},{next_var}"

    for generator in cut:
        for antigenerator in anticut:
            problem += lp.lpDot(variables, generator) + \
                1 <= lp.lpDot(variables, antigenerator)

    problem.solve(lp.PULP_CBC_CMD(msg=0))
    if problem.status == lp.LpStatusInfeasible:
        return None

    return [var.varValue for var in variables]


def clean_game(game):
    game = [int(round(x)) for x in game]
    return [x - min(game) for x in game]


def different_games(n):
    cuts = generate(n)
    games = [cut_to_coins(list(cut), list(complement_of_set(cut)))
             for cut in cuts]
    for c, g in zip(cuts, games):
        if g is None:
            print(f"for cut {c} there is no game")
        else:
            print(len(c), clean_game(g))
    return [g for g in games if g is not None]


def is_complementary(x, y):
    return x == [1 - v for v in y]




def generate_sat(n):

    solutions = get_vectors(2 * n, n)
    lits_number = len(solutions)

    cnf = CNF()
    cnf.extend(PBEnc.equals(list(range(1, lits_number + 1)),
               bound=lits_number // 2, top_id=2 * lits_number))

    for idx1, sol1 in enumerate(solutions):
        cnf.append([-(idx1 + lits_number + 1), (idx1 + 1)])

        ancestors_idc = []
        for idx2, sol2 in enumerate(solutions):
            if idx1 == idx2:
                continue

            if geq(sol2, sol1):
                cnf.append([-(idx2 + 1), idx1 + 1])
                cnf.append([-(idx2 + lits_number + 1), -
                           (idx1 + lits_number + 1)])
                ancestors_idc.append(idx2)

            if is_complementary(sol1, sol2):
                cnf.append([-(idx1 + 1), -(idx2 + 1)])

        cnf.append([-(idx1 + 1), (idx1 + lits_number + 1)] +
                   [(ancestor_idx + lits_number + 1) for ancestor_idx in ancestors_idc])

    solver = Minisat22()
    solver.append_formula(cnf)

    cuts = []
    for model in solver.enum_models():
        cut = []
        for idx, solution in enumerate(solutions):
            if model[idx + lits_number] > 0:
                cut.append(tuple(solution))

        cuts.append(cut)

    return cuts


def compare_generators_performance(n):
    start = time.time()
    print(f"Cuts (classic): {len(generate(n))}")
    print(f"Time elapsed: {time.time() - start:.2f} s\n")

    start = time.time()
    print(f"Cuts (SAT): {len(generate_sat(n))}")
    print(f"Time elapsed: {(time.time() - start):.2f} s")


if __name__ == "__main__":

    print("Generating cuts...")
    start = time.time()
    cuts = generate(4)
    print(f"Time: {time.time() - start:0.2f} s")
    
    print("Transforming cuts...")
    start = time.time()    
    valid_cuts = 0
    for cut in cuts:
        game = cut_to_coins(list(cut), list(complement_of_set(cut)))
        if game is not None:
            valid_cuts += 1
            
        print(str(cut) + " " + str(game if game is not None else "None"))
        
    print(f"Time: {time.time() - start:0.2f} s\n")
    print(f"Cuts: {len(cuts)}")
    print(f"Valid cuts: {valid_cuts}")
    print(f"Ratio: {valid_cuts / len(cuts):0.2f}")

    # print(generate_sat(3))
    # compare_generators_performance(5)
    # print(len(different_games(4)))
