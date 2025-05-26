import random
from typing import List

import pandas as pd
from ortools.linear_solver import pywraplp
from deap import base, creator, tools, algorithms
from config.constants import salary_constraints
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD


# ---------------------------------------------------------------- GA
def solve_ga(df_sl: pd.DataFrame, platform: str, pred_flag: bool) -> List[int]:
    cap = salary_constraints[platform]["salary_cap"]
    pos_req = salary_constraints[platform][
        "positions"
    ]  # e.g. {'PG':2,'SG':2,'G':1,'F':1,'C':1}
    roster_sz = sum(pos_req.values())

    if len(df_sl) < roster_sz:  # not enough players in the pool
        return []

    # ------------------------------------------------------------------
    # 1.  DEAP primitive registration (created once per interpreter)   -
    # ------------------------------------------------------------------
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("indices", random.sample, range(len(df_sl)), roster_sz)
    tb.register("individual", tools.initIterate, creator.Individual, tb.indices)
    tb.register("population", tools.initRepeat, list, tb.individual)

    score_col = f"fp_{platform}_pred" if pred_flag else f"fp_{platform}"

    # ------------------------------------------------------------------
    # 2.  Helpers                                                       -
    # ------------------------------------------------------------------
    def salary_ok(ind):
        return sum(df_sl.iloc[i][f"{platform}_salary"] for i in ind) <= cap

    def pos_ok(ind):
        cnt = {k: 0 for k in pos_req}  # fresh counter
        for i in ind:
            slots = str(df_sl.iloc[i][f"{platform}_position"]).split("/")
            for s in slots:
                if s in cnt:
                    cnt[s] += 1
            # derive the flex slots once per player
            if ("PG" in slots or "SG" in slots) and "G" in cnt:
                cnt["G"] += 1
            if ("SF" in slots or "PF" in slots) and "F" in cnt:
                cnt["F"] += 1
        # **≥** not **==**  → roster may over-satisfy a slot (UTIL absorbs it)
        return all(cnt[p] >= need for p, need in pos_req.items() if p != "UTIL")

    def fitness(ind):
        if (len(set(ind)) != roster_sz) or (not salary_ok(ind)) or (not pos_ok(ind)):
            return (-1e6,)  # make it undesirable
        return (sum(df_sl.iloc[i][score_col] for i in ind),)

    tb.register("evaluate", fitness)
    tb.register("mate", tools.cxTwoPoint)

    def mutate(ind):
        # simple “replace one player” mutation
        idx = random.randrange(roster_sz)
        new = random.randrange(len(df_sl))
        while new in ind:
            new = random.randrange(len(df_sl))
        ind[idx] = new
        return (ind,)

    tb.register("mutate", mutate)
    tb.register("select", tools.selTournament, tournsize=3)

    # ------------------------------------------------------------------
    # 3.  Run GA                                                        -
    # ------------------------------------------------------------------
    pop = tb.population(n=min(200, len(df_sl)))  # larger search space
    hof = tools.HallOfFame(1)  # keep best ever
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", max)

    algorithms.eaSimple(
        pop,
        tb,
        cxpb=0.7,
        mutpb=0.2,
        ngen=40,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best = hof[0] if hof else []
    # final safety check (could all be infeasible)
    return best if best and fitness(best)[0] > -1e5 else []


# ---------------------------------------------------------------- ILP
def solve_ilp(df_sl: pd.DataFrame, platform: str, pred_flag: bool) -> List[int]:
    cap = salary_constraints[platform]["salary_cap"]
    pos_req = salary_constraints[platform]["positions"]
    roster = sum(pos_req.values())

    score_col = f"fp_{platform}_pred" if pred_flag else f"fp_{platform}"

    # ---------- sanity: quick feasibility check --------------------------
    for pos, k in pos_req.items():
        if pos == "UTIL":
            continue
        if (
            df_sl[f"{platform}_position"]
            .dropna()
            .apply(lambda s: pos in str(s).split("/"))
            .sum()
        ) < k:
            raise RuntimeError(f"ILP infeasible: not enough {pos}s in pool")

    # ---------- build ILP -------------------------------------------------
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Could not create CBC solver")

    x = {i: solver.BoolVar(f"x_{i}") for i in df_sl.index}

    # objective
    solver.Maximize(solver.Sum(df_sl.at[i, score_col] * x[i] for i in df_sl.index))

    # salary + roster size
    solver.Add(
        solver.Sum(df_sl.at[i, f"{platform}_salary"] * x[i] for i in df_sl.index) <= cap
    )
    solver.Add(solver.Sum(x.values()) == roster)

    # positional ≥ (not ==) to use UTIL more flexibly
    for pos, k in pos_req.items():
        if pos == "UTIL":
            continue
        solver.Add(
            solver.Sum(
                x[i]
                for i in df_sl.index
                if pos in str(df_sl.at[i, f"{platform}_position"]).split("/")
            )
            >= k
        )

    # give CBC a gentle time limit (seconds)
    solver.SetTimeLimit(1_000_000)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"ILP infeasible or time‑out (status={status})")

    return [i for i in df_sl.index if x[i].solution_value() > 0.5]


def solve_pulp(df, plat="fanduel", pred=True):
    prob = LpProblem("dfs", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in df.index}
    fp_col = f"fp_{plat}_pred" if pred else f"fp_{plat}"

    # objective
    prob += lpSum(df.at[i, fp_col] * x[i] for i in df.index)

    # salary, roster, positions (same as ILP snippet)
    cap = salary_constraints[plat]["salary_cap"]
    pos_req = salary_constraints[plat]["positions"]
    prob += lpSum(df.at[i, f"{plat}_salary"] * x[i] for i in df.index) <= cap
    prob += lpSum(x.values()) == sum(pos_req.values())
    for p, k in pos_req.items():
        if p == "UTIL":
            continue
        prob += (
            lpSum(
                x[i]
                for i in df.index
                if p in str(df.at[i, f"{plat}_position"]).split("/")
            )
            == k
        )

    prob.solve(PULP_CBC_CMD(msg=False))  # or PULP_HIGHSLP
    return [i for i, var in x.items() if var.value()]


#
# def solve_mip(df, plat="fanduel", pred=True):
#     try:
#         # Check if CBC library is available
#         from mip.cbc import has_cbc
#
#         if not has_cbc:
#             raise ImportError("CBC library not available")
#
#         m = Model(sense=MAX, solver_name="CBC")
#         x = [m.add_var(var_type=BINARY) for _ in df.index]
#         fp_col = f"fp_{plat}_pred" if pred else f"fp_{plat}"
#         m.objective = xsum(df.at[i, fp_col] * x[i] for i in df.index)
#
#         # Add salary constraint
#         cap = salary_constraints[plat]["salary_cap"]
#         m += xsum(df.at[i, f"{plat}_salary"] * x[i] for i in range(len(df))) <= cap
#
#         # Add roster size constraint
#         pos_req = salary_constraints[plat]["positions"]
#         roster_size = sum(pos_req.values())
#         m += xsum(x) == roster_size
#
#         # Add position constraints
#         for pos, k in pos_req.items():
#             if pos == "UTIL":
#                 continue
#             m += (
#                 xsum(
#                     x[i]
#                     for i in range(len(df))
#                     if pos in str(df.at[i, f"{plat}_position"]).split("/")
#                 )
#                 >= k
#             )
#
#         m.optimize(max_seconds=30)
#         return [i for i, v in enumerate(x) if v.x >= 0.99]
#     except ImportError as e:
#         raise RuntimeError(f"MIP solver failed: {str(e)}")
#     except Exception as e:
#         raise RuntimeError(f"MIP solver failed: {str(e)}")
