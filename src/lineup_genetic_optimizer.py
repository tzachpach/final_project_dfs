import logging
import random
import warnings

from config.constants import salary_constraints

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from fuzzywuzzy import fuzz, process


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def get_best_match(name, name_list, threshold=80):
    best_match, best_score = process.extractOne(name, name_list, scorer=fuzz.token_set_ratio)
    if best_score >= threshold:
        return best_match
    else:
        return None

def get_lineup(df):

    df = df.reset_index(drop=True)
    df = df.sort_values('game_date', ascending=True)

    # Get the list of unique dates
    unique_dates = df['game_date'].unique()
    date_res = []

    # For each unique date, run the GA and get the best lineup
    for date in unique_dates:
        spec_date = {"date": date}
        logging.info(f"Running for {date}")
        for platform in ['fanduel']:  # , 'yahoo', 'draftkings'
            df_filtered, best_individual = get_best_lineup(date, df, platform)
            df_filtered_pred, best_individual_pred = get_best_lineup(date, df, platform, pred_flag=True)

            spec_date.update({
                # 1) Count of players in the data for this date/platform
                f"{platform}_player_pool_count": len(df_filtered),

                # 2) Best lineup chosen by maximizing GT (actual) points
                f"{platform}_GT_players": [
                    df_filtered.iloc[i]["player_name"] for i in best_individual
                ],

                # 3) Best lineup chosen by maximizing predicted points
                f"{platform}_predicted_players": [
                    df_filtered_pred.iloc[i]["player_name"] for i in best_individual_pred
                ],

                # 4) Sum of actual (GT) points for the best GT lineup
                f"{platform}_GT_points": sum(
                    df_filtered.iloc[i][f'fp_{platform}'] for i in best_individual
                ),

                # 5) Sum of predicted points for the best predicted lineup
                f"{platform}_predicted_points": sum(
                    df_filtered_pred.iloc[i][f'fp_{platform}_pred'] for i in best_individual_pred
                ),

                # 6) How well the best predicted lineup performed in reality
                f"{platform}_predicted_lineup_GT_points": sum(
                    df_filtered.iloc[i][f'fp_{platform}'] for i in best_individual_pred
                ),

                # 7) How well the best GT lineup performed in prediction
                f"{platform}_GT_lineup_predicted_points": sum(
                    df_filtered.iloc[i][f'fp_{platform}_pred'] for i in best_individual
                ),

                # 8) Total salary used by the best GT lineup
                f"{platform}_GT_salary": sum(
                    df_filtered.iloc[i][f'{platform}_salary'] for i in best_individual
                ),

                # 9) Total salary used by the best predicted lineup
                f"{platform}_predicted_salary": sum(
                    df_filtered_pred.iloc[i][f'{platform}_salary'] for i in best_individual_pred
                ),

                # 10) Duplicated players (if any) in the best GT lineup
                f"{platform}_GT_duplicates": (
                        len(best_individual) - len(np.unique(best_individual))
                ),

                # 11) Duplicated players (if any) in the best predicted lineup
                f"{platform}_predicted_duplicates": (
                        len(best_individual_pred) - len(np.unique(best_individual_pred))
                ),
            })

            # ----- Overlap information -----
            overlap_indices = set(best_individual).intersection(set(best_individual_pred))
            overlap_player_names = [df_filtered.iloc[i]["player_name"] for i in overlap_indices]

            spec_date.update({
                # List of players who appear in both lineups
                f"{platform}_overlap_players": overlap_player_names,

                # Count of overlapping players
                f"{platform}_overlap_count": len(overlap_indices),

                # Sum of actual points for overlapping players
                f"{platform}_overlap_GT_points": sum(
                    df_filtered.iloc[i][f'fp_{platform}'] for i in overlap_indices
                ),

                # Sum of predicted points for overlapping players
                f"{platform}_overlap_predicted_points": sum(
                    df_filtered.iloc[i][f'fp_{platform}_pred'] for i in overlap_indices
                )
            })

        date_res.append(spec_date)

    return pd.DataFrame(date_res)


def get_best_lineup(date, df, platform="fanduel", pred_flag=False):
    df_filtered = df[df['game_date'] == date].reset_index(drop=True)
    salary_cap = salary_constraints[platform]['salary_cap']
    position_constraints = salary_constraints[platform]['positions']
    num_players_selected = sum(position_constraints.values())

    # Check if classes already exist
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_unique", random.sample, range(len(df_filtered)), num_players_selected)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_unique)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        total_cost = sum(df_filtered.iloc[i][f'{platform.lower()}_salary'] for i in individual)
        penalty = max(0, total_cost - salary_cap) * 10
        pred_col = f'fp_{platform.lower()}_pred' if pred_flag else f'fp_{platform.lower()}'
        total_score = sum(df_filtered.iloc[i][pred_col] for i in individual)

        position_count = {pos: 0 for pos in position_constraints.keys()}
        for i in individual:
            if  pd.isna(df_filtered.iloc[i][f'{platform.lower()}_position']):
                continue
            pos_split = df_filtered.iloc[i][f'{platform.lower()}_position'].split("/")
            for p in pos_split:
                if p in position_count:
                    position_count[p] += 1
                if p in ["PF", "SF"] and "F" in position_count:
                    position_count["F"] += 1
                if p in ["PG", "SG"] and "G" in position_count:
                    position_count["G"] += 1
            if "UTIL" in position_count:
                position_count["UTIL"] += 1

        positional_penalty = 0
        for pos, required_count  in position_constraints.items():
            if pos != "UTIL":
                positional_penalty += max(0, required_count - position_count[pos]) * 10

        # Penalize duplicate players
        unique_individuals = set(individual)
        duplicate_penalty = (len(individual) - len(unique_individuals)) * 10

        fitness = total_score - penalty - positional_penalty - duplicate_penalty
        return fitness,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)

    def mutate(individual):
        if random.random() < 0.2:
            player_to_replace = random.choice(individual)
            new_player = random.choice(range(len(df_filtered)))
            while new_player in individual:
                new_player = random.choice(range(len(df_filtered)))
            individual[individual.index(player_to_replace)] = new_player
        return individual,

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    try:
        population = toolbox.population(n=min(len(df_filtered), 50))
    except:
        logging.error(f"Population size is too large for the given date, {date}.")
        return df_filtered, []
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    return df_filtered, best_individual