import logging
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import salary_constraints
from src.lineup_solvers import solve_ga, solve_ilp  # , solve_mip

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SOLVER_MAP = {
    "GA": solve_ga,
    "ILP": solve_ilp,
    # "PULP": solve_pulp,
    # "MIP": solve_mip,
}


def _as_sequence(solver) -> Sequence[str]:
    """Allow solver to be 'GA' or ['ILP','GA'] etc."""
    if isinstance(solver, (list, tuple)):
        return [str(s).upper() for s in solver]
    return [str(solver).upper()]


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #


def get_best_lineup(
    date,
    df,
    platform,
    pred_flag,
    solver,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Try the requested solver(s) in order, return (df_day, index_list).
    If none succeed → index_list = [] (caller decides what to do).
    """
    df_day = df[df["game_date"] == date].reset_index(drop=True)
    if df_day.empty:
        log.warning("No rows for date %s; returning empty lineup.", date)
        return df_day, []

    need = [
        f"{platform}_salary",
        f"fp_{platform}",
        f"fp_{platform}_pred",
        f"{platform}_position",
    ]
    
    # Debug logging
    log.info("Columns in df_day before dropna: %s", df_day.columns.tolist())
    log.info("Required columns: %s", need)
    log.info("Sample data before dropna:\n%s", df_day[need].head() if not df_day.empty else "Empty DataFrame")
    
    df_day = df_day.dropna(subset=need)
    df_day = df_day.replace([np.inf, -np.inf], np.nan).dropna(subset=need)

    # More debug logging
    log.info("Rows after dropna: %d", len(df_day))
    log.info("Sample data after dropna:\n%s", df_day[need].head() if not df_day.empty else "Empty DataFrame")

    # Check if we have enough players for a valid roster
    roster_size = sum(salary_constraints[platform]["positions"].values())
    if len(df_day) < roster_size:
        log.warning(
            "Not enough players (%d) for a full roster (%d) on %s",
            len(df_day),
            roster_size,
            date,
        )
        return df_day, []

    for name in _as_sequence(solver):
        if name not in _SOLVER_MAP:
            log.warning("Unknown solver '%s' – skipping.", name)
            continue

        try:
            idx_list = _SOLVER_MAP[name](df_day, platform, pred_flag)
            if idx_list:
                log.info("Solver %s succeeded (|roster|=%d).", name, len(idx_list))
                return df_day, idx_list
            else:
                log.warning("Solver %s returned empty roster – trying next.", name)
        except Exception as e:
            log.warning("Solver %s failed: %s – trying next.", name, e)

    log.error("All solvers failed for date %s / %s.", date, platform)
    return df_day, []
