import logging
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            "max_num_of_players": len(data_filtered),
            "lineup_ideal": [data_filtered.iloc[i]["player"] for i in best_individual],
            "predicted_score_ideal": sum(data_filtered.iloc[i]["y_pred"] for i in best_individual),
            "cost_ideal": sum(data_filtered.iloc[i]["cost"] for i in best_individual),
            "actual_score_ideal": sum(data_filtered.iloc[i]["dkfp"] for i in best_individual),
            "lineup_selected": [data_filtered_pred.iloc[i]["player"] for i in best_individual_pred],
            "predicted_score_selected": sum(data_filtered_pred.iloc[i]["y_pred"] for i in best_individual_pred),
            "cost_selected": sum(data_filtered_pred.iloc[i]["cost"] for i in best_individual_pred),
            "actual_score_selected": sum(data_filtered_pred.iloc[i]["dkfp"] for i in best_individual_pred)

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

    def generate_individual(data_filtered, num_players_selected):
        # Segment players based on some attribute, e.g., cost
        low_cost_players = data_filtered[data_filtered['cost'] < data_filtered['cost'].quantile(0.33)].index.tolist()
        mid_cost_players = data_filtered[(data_filtered['cost'] >= data_filtered['cost'].quantile(0.33)) &
                                         (data_filtered['cost'] < data_filtered['cost'].quantile(0.66))].index.tolist()
        high_cost_players = data_filtered[data_filtered['cost'] >= data_filtered['cost'].quantile(0.66)].index.tolist()

        # Randomly sample players from each segment
        lineup = random.sample(low_cost_players, num_players_selected // 3) + \
                 random.sample(mid_cost_players, num_players_selected // 3) + \
                 random.sample(high_cost_players, num_players_selected - 2 * (num_players_selected // 3))

        # Shuffle the lineup to introduce more diversity
        random.shuffle(lineup)

        return lineup

    # Generate a random permutation of the players and then select the first num_players_selected players
    toolbox.register("attr_unique", random.sample, range(len(data_filtered)), num_players_selected)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: generate_individual(data_filtered, num_players_selected))
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