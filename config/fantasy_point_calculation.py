import pandas as pd


def calculate_fp_fanduel(row, pred_mode=False):
    pred = "_pred" if pred_mode else ""
    if pred_mode:
        for stat in ["pts", "reb", "ast", "stl", "blk", "tov"]:
            assert f"{stat}_pred" in row, f"Expected predicted stat '{stat}_pred' missing in row"
    return (
        row[f"pts{pred}"]
        + row[f"reb{pred}"] * 1.2
        + row[f"ast{pred}"] * 1.5
        + row[f"stl{pred}"] * 3
        + row[f"blk{pred}"] * 3
        - row[f"tov{pred}"] * 1
    )


def calculate_fp_yahoo(row, pred_mode=False):
    pred = "_pred" if pred_mode else ""
    return (
        row[f"pts{pred}"]
        + row[f"reb{pred}"] * 1.2
        + row[f"ast{pred}"] * 1.5
        + row[f"stl{pred}"] * 3
        + row[f"blk{pred}"] * 3
        - row[f"tov{pred}"] * 1
    )


def calculate_fp_draftkings(row, pred_mode=False):
    pred = "_pred" if pred_mode else ""
    fp = (
        row[f"pts{pred}"]
        + row[f"reb{pred}"] * 1.25
        + row[f"ast{pred}"] * 1.5
        + row[f"stl{pred}"] * 2
        + row[f"blk{pred}"] * 2
        - row[f"tov{pred}"] * 0.5
    )

    # Calculate Double-Double and Triple-Double bonuses
    stats = [
        row[f"pts{pred}"],
        row[f"reb{pred}"],
        row[f"ast{pred}"],
        row[f"stl{pred}"],
        row[f"blk{pred}"],
    ]
    double_double = sum([1 for stat in stats if stat >= 10]) >= 2
    triple_double = sum([1 for stat in stats if stat >= 10]) >= 3

    if double_double:
        fp += 1.5
    if triple_double:
        fp += 3

    return fp


def calculate_exceptional_games_and_doubles(group, thresholds):
    """Calculate exceptional games, double-doubles, and triple-doubles for a group of games."""
    results = {}

    # Exceptional games counts based on thresholds
    for stat, threshold in thresholds.items():
        exceptional_col = f"{stat}_exceptional_games"
        results[exceptional_col] = (group[stat] >= threshold).sum()

    # Double-doubles and triple-doubles
    double_double = (
        (group[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 2
    ).sum()
    triple_double = (
        (group[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 3
    ).sum()
    results["double_doubles"] = double_double
    results["triple_doubles"] = triple_double

    return pd.Series(results)
