import ast
import logging

import numpy as np
import pandas as pd


def post_process_results(df):
    df['lineup_ideal'] = df['lineup_ideal'].apply(lambda x: ast.literal_eval(x))
    df['lineup_selected'] = df['lineup_selected'].apply(lambda x: ast.literal_eval(x))


    df['is_repeat'] = df['lineup_ideal'].apply(lambda x: len(x) - len(np.unique(x)))
    df['is_repeat_selected'] = df['lineup_selected'].apply(lambda x: len(x) - len(np.unique(x)))

    # TODO: ensure is_repeat is 0!
    logging.info(("Added is_repeat columns"))

    df['precision'] = df.apply(lambda row: calculate_kpis(row)[0], axis=1)
    df['recall'] = df.apply(lambda row: calculate_kpis(row)[1], axis=1)
    df['f1'] = df.apply(lambda row: calculate_kpis(row)[2], axis=1)
    df['budget_efficiency_ideal'] = df.apply(lambda row: calculate_kpis(row)[3], axis=1)
    df['budget_efficiency_selected'] = df.apply(lambda row: calculate_kpis(row)[4], axis=1)


    logging.info(("Added KPIs columns"))

    avg_precision = df['precision'].mean()
    avg_recall = df['recall'].mean()
    avg_f1 = df['f1'].mean()
    avg_budget_efficiency_gain = df['budget_efficiency_selected'].mean() / df['budget_efficiency_ideal'].mean()
    # avg_top_player_accuracy = df['top_player_accuracy'].mean()
    # avg_positional_accuracy = df['positional_accuracy'].mean()

    aggregate_res = pd.DataFrame({"avg_precision": [avg_precision],
                                  "avg_recall": [avg_recall],
                                  "avg_f1": [avg_f1],
                                  "avg_budget_efficiency_gain": [avg_budget_efficiency_gain],
                                  #   "avg_top_player_accuracy": [avg_top_player_accuracy],
                                  #   "avg_positional_accuracy": [avg_positional_accuracy]
                                    })
    logging.info("Added aggregate KPIs")

    # TODO: add visualization of the results
    return df, aggregate_res


def calculate_kpis(row):
    selected_set = set(row['lineup_selected'])
    ideal_set = set(row['lineup_ideal'])

    common_players = selected_set.intersection(ideal_set)

    precision = len(common_players) / len(selected_set) if len(selected_set) > 0 else 0
    recall = len(common_players) / len(ideal_set) if len(ideal_set) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    budget_efficiency_ideal = row['actual_score_ideal'] / row['cost_ideal'] if row['cost_ideal'] > 0 else 0
    budget_efficiency_selected = row['actual_score_selected'] / row['cost_selected'] if row['cost_selected'] > 0 else 0

    # TODO: add top_player_accuracy and positional_accuracy as soon as the data catches up
    # top_player_accuracy = len(set(row['top_players']).intersection(selected_set)) / len(row['top_players']) if len(
    #     row['top_players']) > 0 else 0
    # positional_accuracy = len(set(row['positions']).intersection(selected_set)) / len(row['positions']) if len(
    #     row['positions']) > 0 else 0

    return precision, recall, f1_score, budget_efficiency_ideal, budget_efficiency_selected #top_player_accuracy, positional_accuracy

#             "lineup_ideal": [data_filtered.iloc[i]["player"] for i in best_individual],
#             "predicted_score_ideal": sum(data_filtered.iloc[i]["y_pred"] for i in best_individual),
#             "cost_ideal": sum(data_filtered.iloc[i]["cost"] for i in best_individual),
#             "actual_score_ideal": sum(data_filtered.iloc[i]["dkfp"] for i in best_individual),
#             "lineup_selected": [data_filtered_pred.iloc[i]["player"] for i in best_individual_pred],
#             "predicted_score_selected": sum(data_filtered_pred.iloc[i]["y_pred"] for i in best_individual_pred),
#             "cost_selected": sum(data_filtered_pred.iloc[i]["cost"] for i in best_individual_pred),
#             "actual_score_selected": sum(data_filtered_pred.iloc[i]["dkfp"] for i in best_individual_pred)

df = pd.read_csv('genetic_selection.csv')
df, agg = post_process_results(df)