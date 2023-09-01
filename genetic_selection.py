import logging
import random

import pandas as pd
from deap import base, creator, tools, algorithms

from IDC.final_project.data_loader import data_loader
from IDC.final_project.fetch_y_pred import predict_dkfp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define your data (replace this with your actual data)

def get_lineup(data):

    data = data.reset_index()
    data = data.drop('index', axis=1)

    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data = data.drop(['year', 'month', 'day'], axis=1)


    # Define your lineup constraints
    salary_cap = 1200
    num_players_selected = 8

    # Get the list of unique dates
    unique_dates = data['date'].unique()
    date_res = []

    # For each unique date, run the GA and get the best lineup
    for date in unique_dates:

        data_filtered, best_individual = get_best_lineup(date, data)
        data_filtered_pred, best_individual_pred = get_best_lineup(date, data, pred_flag=True)


        date_res.append({
            "date": date,
            "best_lineup": [data_filtered.iloc[i]["player"] for i in best_individual],
            "best_pred_score": sum(data_filtered.iloc[i]["y_pred"] for i in best_individual),
            "best_cost": sum(data_filtered.iloc[i]["cost"] for i in best_individual),
            "best_actual_score": sum(data_filtered.iloc[i]["dkfp"] for i in best_individual),
            "best_lineup_pred": [data_filtered_pred.iloc[i]["player"] for i in best_individual_pred],
            "best_pred_score_pred": sum(data_filtered_pred.iloc[i]["y_pred"] for i in best_individual_pred),
            "best_cost_pred": sum(data_filtered_pred.iloc[i]["cost"] for i in best_individual_pred),
            "best_actual_score_pred": sum(data_filtered_pred.iloc[i]["dkfp"] for i in best_individual_pred)

        })

    return pd.DataFrame(date_res)

def get_best_lineup(date, data, pred_flag=False, salary_cap=1200, num_players_selected=8):
    data_filtered = data[data['date'] == date]
    data_filtered = data_filtered.reset_index(drop=True)
    # Create a fitness class for the GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the genetic operators
    toolbox = base.Toolbox()

    # Generate a random permutation of the players and then select the first num_players_selected players
    toolbox.register("attr_unique", random.sample, range(len(data_filtered)), num_players_selected)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_unique)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom fitness function that penalizes salary cap violations
    def evaluate(individual):
        total_cost = sum(data_filtered.iloc[i]["cost"] for i in individual)
        penalty = max(0, total_cost - salary_cap) * 10
        pred_col = "y_pred" if pred_flag else "dkfp"
        total_score = sum(data_filtered.iloc[i][pred_col] for i in individual)

        # Get the positions of the selected players
        positions = [data_filtered.iloc[i]["pos"] for i in individual]

        # Check the positional constraints
        num_bigs = positions.count('Big')
        num_forwards = positions.count('Forward')
        num_points = positions.count('Point')

        # Add a penalty for not meeting the positional constraints
        # Currently we demand 2 bigs, 2 forwards and 2 points and the rest is whatever
        positional_penalty = 0
        positional_penalty += abs(2 - num_bigs) * 10
        positional_penalty += abs(2 - num_forwards) * 10
        positional_penalty += abs(2 - num_points) * 10

        fitness = total_score - penalty - positional_penalty
        # logging.info("Evaluating individual with fitness: %s", fitness)
        return fitness,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)

    # Custom mutation operator that selects a random player from the individual and replaces it with a random player from the data
    def mutate(individual):
        if random.random() < 0.2:
            player_to_replace = random.choice(individual)
            new_player = random.choice(range(len(data_filtered)))
            while new_player in individual:
                new_player = random.choice(range(len(data_filtered)))
            individual[individual.index(player_to_replace)] = new_player
        return individual,

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create the population
    population = toolbox.population(n=min(len(data_filtered), 50))

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10)

    # Get the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]
    return data_filtered, best_individual


data = data_loader(n=10)
data = predict_dkfp(data)
res = get_lineup(data)

# TODO: understand the current bug with the repeating players
# TODO: start talking to mentors


# TODO: read up on deep q learning
# TODO: understand why RL and if RL is really needed to solve
# TODO: manipulate the results and find some KPIs
# TODO: talk to Eyal? to Assaf? to Nir?

# TODO: get better data - ctg
# TODO: get better data - basketball reference
# TODO: get better data - fanduel
# TODO: get better data - draftkings
