from itertools import permutations, combinations, product
from functools import lru_cache
from multiprocessing import Pool

class PartialMove:
    def __init__(self, above_threshold: bool, shift: int):
        self.above_threshold = above_threshold
        self.shift = shift
    def __eq__(self, other):
        return self.above_threshold == other.above_threshold and self.shift == other.shift


class BobMove:
    def __init__(self, first_game_move: PartialMove, second_game_move: PartialMove):
        self.first_game_move = first_game_move
        self.second_game_move = second_game_move


class AliceResponse:
    def __init__(self, first_above: bool, second_above: bool):
        self.first_above = first_above
        self.second_above = second_above


def get_base(shifts1, shifts2):
    return [
        BobMove(PartialMove(t1, s1), PartialMove(t2, s2))
        for s1 in shifts1
        for s2 in shifts2
        for t1 in (True, False)
        for t2 in (True, False)]


def get_responses():
    return [
        AliceResponse(t1, t2)
        for t1 in (True, False)
        for t2 in (True, False)
    ]


def get_strategies(base):
    responses = get_responses()
    n = len(base)
    strategies = []

    for remainder in range(len(responses) ** n):
        strategy = []
        for _ in range(n):
            strategy.append(responses[remainder % 4])
            remainder //= 4
        strategies.append(strategy)

    return strategies


class Game:
    def __init__(self, elected: PartialMove):
        self.elected = elected

def game_from_basemove(basemove: PartialMove):
    g = Game(basemove)
    g.a = []
    g.b = []
    if basemove.above_threshold:
        g.a.append(int(basemove.above_threshold))
    else:
        g.b.append(int(basemove.above_threshold))
    return g

def games_from_base(base: BobMove):
    return (game_from_basemove(base.first_game_move), game_from_basemove(base.second_game_move))


def response_to(game: Game, basal_move: PartialMove, response: bool):
    if game.elected.above_threshold:
        accept_responses = [-1,0,2,3]
        decline_responses = [-2,-1,0,2]
    else:
        accept_responses = [-1,1,2,3]
        decline_responses = [-2,-1,1,2]
    if response == 0:
        game.a.append(decline_responses[basal_move.shift])
    else:
        game.b.append(accept_responses[basal_move.shift])
    game.a = sorted(game.a)
    game.b = sorted(game.b)
    return game


def responsed_games(games: (Game, Game), basemove: BobMove, responses: AliceResponse):
    return (response_to(games[0], basemove.first_game_move, responses.first_above), response_to(games[1], basemove.second_game_move, responses.second_above))


def run_games(base, response):
    return responsed_games(games_from_base(base), base, response)


def is_destroying(first, second):
    if len(first.a) != len(second.b):
        return False
    for x, y in zip(first.b, second.a):
        if y > x:
            return False
    for x, y in zip(first.a, second.b):
        if y < x:
            return False
    return True


#def add(base, reaction):
#    return ((base[0], reaction[0]), (base[1], reaction[1]))


#def negate_move(move):
#    return (-move[0], -move[1])


#def is_doomed(outcomes):
#    ab, ba = list(zip(*(outcomes * 2)))
#    return all(is_destroying(x, y) for x, y in zip(ab[:-1], ba[1:]))


#def permutation_is_doomed(permutation):
#    return any(is_doomed(permutation[:i]) for i in range(1, len(permutation) + 1))


def get_outcome(reaction, base):
    return [run_games(b, r) for b, r in zip(base, reaction)]


#def any_permutation_is_doomed(perms):
#    return any(permutation_is_doomed(x) for x in perms)


def strategy_is_doomed(outcomes: (Game, Game), first_second = None, last_first = None):
    for i in range(len(outcomes)):
        if last_first is None or is_destroying(last_first, outcomes[i][1]):
            new_first_second = first_second if first_second is not None else outcomes[i][1]
            if is_destroying(outcomes[i][0], new_first_second):
                return True
            if strategy_is_doomed(outcomes[:i]+outcomes[i+1:], new_first_second, outcomes[i][0]):
                return True
    return False



#def is_valid(s, base, workers=1):

#    perms = list(permutations(get_outcome(s, base)))
#    chunk_size = len(perms) // workers
#    jobs = [perms[i * chunk_size: (i + 1) * chunk_size] for i in range(workers - 1)]
#    jobs.append(perms[(workers - 1) * chunk_size:])
#
#    with Pool() as pool:
#        results = pool.map(any_permutation_is_doomed, jobs)
#
#    return not any(results)

#steroid_constant = 5
def is_valid(s, base):
    outcome = get_outcome(s, base)
    return not strategy_is_doomed(outcome)
#    for n in range(1, steroid_constant):
#        for c in combinations(outcome, n):
#            for p in permutations(c):
#                if is_doomed(p):
#                    return False
#    return True


def is_dependent(strategy, base):
    for i in range(len(strategy)):
        for j in range(i + 1, len(strategy)):
            if base[i].first_game_move == base[j].first_game_move:
                if strategy[i].first_above != strategy[j].first_above:
                    return True
            if base[i].second_game_move == base[j].second_game_move:
                if strategy[i].second_above != strategy[j].second_above:
                    return True
    return False


def get_substrategies(shifts1, shifts2):
    base = get_base(shifts1, shifts2)
    strategies = [s for s in get_strategies(base) if not is_dependent(s, base)]
    strategies = [s for s in strategies if is_valid(s, base)]
    print(len(strategies))
    return strategies


def print_strategies(strategies):
    for strategy in strategies:
        for x in strategy:
            print(f"{int(x.first_above)}-{int(x.second_above)};",end="")
        print("")
    if not strategies:
        print("No strategies left")


def print_base(base):
    for b in base:
        print(f"{int(b.first_game_move.above_threshold)}-{int(b.second_game_move.above_threshold)};", end="")
    print("")
    for b in base:
        print(f"{int(b.first_game_move.shift)}-{int(b.second_game_move.shift)};", end="")
    print("\n=================================================")


def strategies_product(members, base, partial_result = [[]]):
    if not members:
        return partial_result
    next_partial_result = [a + b for a in partial_result for b in members[0] if not is_dependent(a + b, base)]
    return strategies_product(members[1:], base, next_partial_result)


def get_massacred_strategies(shifts):
    substrategies = [
        get_substrategies((s1,), (s2,))
        for s1 in shifts
        for s2 in shifts
    ]
    base = get_base(shifts, shifts)
    strategies = strategies_product(substrategies, base)
    print(len(strategies))
    strategies = [s for s in strategies if not is_dependent(s, base) and is_valid(s, base)]
    print(len(strategies))
    return strategies

shifts = (0,1,2,3)
print_base(get_base(shifts, shifts))
print_strategies(get_massacred_strategies(shifts))

