import logging
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# from IDC.final_project.df_loader import df_loader
# from IDC.final_project.fetch_y_pred import predict_dkfp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

salary_constraints = {
    "yahoo": {
        "salary_cap": 200,
        "positions": {
            "PG": 1,
            "SG": 1,
            "SF": 1,
            "PF": 1,
            "C": 1,
            "G": 1,  # Guard (PG or SG)
            "F": 1,  # Forward (SF or PF)
            "UTIL": 1  # Any position
        }
    },
    "fanduel": {
        "salary_cap": 60000,
        "positions": {
            "PG": 2,
            "SG": 2,
            "SF": 2,
            "PF": 2,
            "C": 1
        }
    },
    "drartkings": {
        "salary_cap": 50000,
        "positions": {
            "PG": 1,
            "SG": 1,
            "SF": 1,
            "PF": 1,
            "C": 1,
            "G": 1,  # Guard (PG or SG)
            "F": 1,  # Forward (SF or PF)
            "UTIL": 1  # Any position
        }
    }
}

def get_best_match(name, name_list, threshold=80):
    best_match, best_score = process.extractOne(name, name_list, scorer=fuzz.token_set_ratio)
    if best_score >= threshold:
        return best_match
    else:
        return None

def merge_fp_pred_and_salaries(fp_pred_df, salaries_df):
    name_mapping = {}
    prices_names = set(salaries_df['player_name'].tolist())
    for name in set(fp_pred_df['player_name']):
        if name == 'Juan Hernangomez': # Comfusion with Willy Hernangomez
            continue
        best_match = get_best_match(name, prices_names)
        if best_match:
            name_mapping[name] = best_match
        if best_match and name != best_match:
            print(name, best_match)
    fp_pred_df['player_name'] = fp_pred_df['player_name'].map(name_mapping).fillna(fp_pred_df['player_name'])
    fp_pred_df['team'] = fp_pred_df['team'].apply(lambda x: x.lower())
    # Merge dfFrames on 'player_name' and 'game_date' (date in df2)
    merged_df = pd.merge(fp_pred_df, salaries_df, left_on=['player_name', 'game_date'], right_on=['player_name', 'date'], how='left')
    return merged_df

def get_lineup(df):

    df = df.reset_index()
    df = df.drop('index', axis=1)

    # Get the list of unique dates
    unique_dates = df['game_date'].unique()
    date_res = []

    # For each unique date, run the GA and get the best lineup
    for date in unique_dates:
        for platform in ['yahoo', 'fanduel', 'draftkings']:
            df_filtered, best_individual = get_best_lineup(date, df, platform)
            df_filtered_pred, best_individual_pred = get_best_lineup(date, df, platform, pred_flag=True)


            date_res.append({
                "date": date,
                f"max_num_of_players_{platform}": len(df_filtered),
                f"best_actual_lineup_{platform}": [df_filtered.iloc[i]["player_name"] for i in best_individual],
                f"best_pred_lineup_{platform}": [df_filtered_pred.iloc[i]["player_name"] for i in best_individual_pred],
                f"best_actual_score_{platform}": sum(df_filtered.iloc[i][f'fp_{platform}'] for i in best_individual),
                f"best_pred_score_{platform}": sum(df_filtered.iloc[i][f'fp_{platform}_pred'] for i in best_individual),
                f"best_actual_cost_{platform}": sum(df_filtered.iloc[i][f'{platform}_salary'] for i in best_individual),
                f"best_pred_cost_{platform}": sum(df_filtered_pred.iloc[i][f'{platform}_salary'] for i in best_individual_pred),
                "is_repeat": len(best_individual) - len(np.unique(best_individual)),
                "is_repeat_pred": len(best_individual_pred) - len(np.unique(best_individual_pred)),
            })

    return pd.dfFrame(date_res)

def get_best_lineup(date, df, platform="yahoo", pred_flag=False):
    df_filtered = df[df['game_date'] == date].reset_index(drop=True)
    salary_cap = salary_constraints[platform]['salary_cap']
    position_constraints = salary_constraints[platform]['positions']
    num_players_selected = sum(position_constraints.values())

    # Create a fitness class for the GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the genetic operators
    toolbox = base.Toolbox()

    # Generate a random permutation of the players and then select the first num_players_selected players
    toolbox.register("attr_unique", random.sample, range(len(df_filtered)), num_players_selected)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_unique)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom fitness function that penalizes salary cap violations
    def evaluate(individual):
        total_cost = sum(df_filtered.iloc[i][f'{platform.lower()}_salary'] for i in individual)
        penalty = max(0, total_cost - salary_cap) * 10
        pred_col = f'fp_{platform.lower()}_pred' if pred_flag else f'fp_{platform.lower()}'
        total_score = sum(df_filtered.iloc[i][pred_col] for i in individual)

        # Get the positions of the selected players
        positions = [df_filtered.iloc[i][f'{platform.lower()}_position'] for i in individual]
        position_count = {pos: 0 for pos in position_constraints.keys()}
        for pos in positions:
            pos_split = pos.split("/")
            for p in pos_split:
                if p in position_count:
                    position_count[p] += 1
                if p in ["PF", "SF"]:
                    position_count["F"] += 1
                if p in ["PG", "SG"]:
                    position_count["G"] += 1
            position_count["UTIL"] += 1  # Any position can count for UTIL

        # Add a penalty for not meeting the positional constraints
        positional_penalty = 0
        for pos, required_count in position_constraints.items():
            positional_penalty += abs(required_count - position_count[pos]) * 10

        # TODO: rework that last part, it doesn't really work with G/F and UTIL
        # Add a penalty for not meeting the positional constraints
        # Currently we demand 2 bigs, 2 forwards and 2 points and the rest is whatever

        fitness = total_score - penalty - positional_penalty
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

    # Create the population
    population = toolbox.population(n=min(len(df_filtered), 50))

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10)

    # Get the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]
    return df_filtered, best_individual



# df = df_loader()
# df = predict_dkfp(df,should_train=True, should_plot=True)
fp_pred_df = pd.read_csv('fp_pred.csv')
salaries_df = pd.read_csv('output_csv/historic_dfs_data_v2.csv')

df = merge_fp_pred_and_salaries(fp_pred_df, salaries_df)
res = get_lineup(df)
res.to_csv('all_res.csv')
# TODO: understand the current bug with the repeating players
# TODO: start talking to mentors


# TODO: read up on deep q learning
# TODO: understand why RL and if RL is really needed to solve
# TODO: manipulate the results and find some KPIs

# TODO: get better df - ctg
# TODO: get better df - basketball reference
# TODO: get better df - fanduel
# TODO: get better df - draftkings
