import random
from typing import List

import pandas as pd
from ortools.linear_solver import pywraplp
from deap import base, creator, tools, algorithms
from config.constants import salary_constraints
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD


# ---------------------------------------------------------------- GA (unchanged)
def solve_ga(df_sl: pd.DataFrame, platform: str, pred_flag: bool) -> List[int]:
    cap = salary_constraints[platform]["salary_cap"]
    pos_req = salary_constraints[platform]["positions"]
    roster = sum(pos_req.values())
    # Check if we have enough players for a valid roster
    if len(df_sl) < roster:
        # Not enough players to form a roster
        return []
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("indices", random.sample, range(len(df_sl)), roster)
    tb.register("individual", tools.initIterate, creator.Individual, tb.indices)
    tb.register("population", tools.initRepeat, list, tb.individual)

    pred_col = f"fp_{platform}_pred" if pred_flag else f"fp_{platform}"

    def feasible(idx_list):
        sal_ok = sum(df_sl.iloc[i][f"{platform}_salary"] for i in idx_list) <= cap
        pos_cnt = {k: 0 for k in pos_req}
        for i in idx_list:
            tokens = str(df_sl.iloc[i][f"{platform}_position"]).split("/")
            for t in tokens:
                if t in pos_cnt:
                    pos_cnt[t] += 1
                if t in ("PG", "SG") and "G" in pos_cnt:
                    pos_cnt["G"] += 1
                if t in ("SF", "PF") and "F" in pos_cnt:
                    pos_cnt["F"] += 1
        pos_ok = all(pos_cnt[p] == v for p, v in pos_req.items() if p != "UTIL")
        return sal_ok and pos_ok and len(set(idx_list)) == roster

    def fitness(ind):
        return (
            (sum(df_sl.iloc[i][pred_col] for i in ind),) if feasible(ind) else (-1e6,)
        )

    tb.register("evaluate", fitness)
    tb.register("mate", tools.cxTwoPoint)

    def mutate(ind):
        if random.random() < 0.2:
            replace = random.choice(ind)
            new_idx = random.randrange(len(df_sl))
            while new_idx in ind:
                new_idx = random.randrange(len(df_sl))
            ind[ind.index(replace)] = new_idx
        return (ind,)

    tb.register("mutate", mutate)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(n=min(50, len(df_sl)))
    algorithms.eaSimple(pop, tb, 0.5, 0.2, 10, verbose=False)
    return tools.selBest(pop, 1)[0]


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
    solver.SetTimeLimit(10_000)

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
