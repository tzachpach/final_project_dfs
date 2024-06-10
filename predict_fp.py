import numpy as np

import pandas as pd

import xgboost as xgb
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA


rolling_window = 10
dfs_cats = ['reb', 'pts', 'ast', 'stl', 'blk', 'to']
same_game_cols = ['min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct',
                  'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'to', 'pf', 'pts',
                  'plus_minus', 'ast_pct', 'ast_ratio', 'ast_tov', 'def_rating',
                  'dreb_pct', 'efg_pct', 'e_def_rating', 'e_net_rating', 'e_off_rating',
                  'e_pace', 'e_usg_pct', 'net_rating', 'off_rating', 'oreb_pct', 'pace',
                  'pace_per40', 'pie', 'poss', 'reb_pct', 'tm_tov_pct', 'ts_pct',
                  'usg_pct']



def calculate_fp_fanduel(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    return (row[f'pts{pred}'] +
            row[f'reb{pred}'] * 1.2 +
            row[f'ast{pred}'] * 1.5 +
            row[f'stl{pred}'] * 3 +
            row[f'blk{pred}'] * 3 -
            row[f'to{pred}'] * 1)

def calculate_fp_yahoo(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    return (row[f'pts{pred}'] +
            row[f'reb{pred}'] * 1.2 +
            row[f'ast{pred}'] * 1.5 +
            row[f'stl{pred}'] * 3 +
            row[f'blk{pred}'] * 3 -
            row[f'to{pred}'] * 1)
def calculate_fp_draftkings(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    fp = (row[f'pts{pred}'] +
          row[f'reb{pred}'] * 1.25 +
          row[f'ast{pred}'] * 1.5 +
          row[f'stl{pred}'] * 2 +
          row[f'blk{pred}'] * 2 -
          row[f'to{pred}'] * 0.5)

    # Calculate Double-Double and Triple-Double bonuses
    stats = [row[f'pts{pred}'], row[f'reb{pred}'], row[f'ast{pred}'], row[f'stl{pred}'], row[f'blk{pred}']]
    double_double = sum([1 for stat in stats if stat >= 10]) >= 2
    triple_double = sum([1 for stat in stats if stat >= 10]) >= 3

    if double_double:
        fp += 1.5
    if triple_double:
        fp += 3

    return fp

def rolling_train_test(X, y, league_week_gb, player_name_to_id, num_weeks_for_training=16, save_model=True, model_dir='models'):
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Initialize lists to store predictions and true values
    all_predictions = []
    all_true_values = []
    all_game_ids = []
    all_game_dates = []
    all_player_ids = []

    # Get the maximum and minimum weeks
    max_week = league_week_gb['league_week'].max()
    min_week = league_week_gb['league_week'].min()

    for start_week in range(max_week, min_week + num_weeks_for_training - 1, -1):
        end_week = start_week - num_weeks_for_training + 1  # inclusive range

        if end_week < min_week:
            break  # Stop if the end_week is less than the minimum week

        # Select training data
        training_weeks = list(range(start_week, end_week - 1, -1))
        X_train = X[X['league_week'].isin(training_weeks)]
        identifying_train_data = X_train[['player_id', 'game_date', 'game_id']]
        X_train = X_train.drop(columns=['game_date', 'game_id'])
        y_train = y.loc[X_train.index]

        # Select test data (the week immediately following the training period)
        X_test = X[X['league_week'] == end_week - 1]
        identifying_test_data = X_test[['player_id', 'game_date', 'game_id']]
        X_test = X_test.drop(columns=['game_date', 'game_id'])

        y_test = y.loc[X_test.index]

        # Create DMatrix for training and testing data
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Specify the booster parameters with tree_method set to 'gpu_hist' if using GPU
        params = {
            'tree_method': 'hist',  # Use 'gpu_hist' if you have a GPU
            'enable_categorical': True
        }

        # Train the model
        model = xgb.train(params, dtrain)

        # Make predictions
        y_pred = model.predict(dtest)

        # Store the predictions and true values
        all_predictions.extend(y_pred)
        all_true_values.extend(y_test)
        all_game_ids.extend(list(identifying_test_data['game_id']))
        all_game_dates.extend(list(identifying_test_data['game_date']))
        all_player_ids.extend(list(identifying_test_data['player_id']))

        # Save the model if requested
        if save_model:
            model_filename = f'{model_dir}/model_week_{start_week}_to_{end_week}.pkl'
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Print the evaluation metrics
        print(f'Training weeks: {training_weeks}')
        print(f'Test weeks: {end_week - 1}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')
        print('')

        print(f'identifying_test_data {identifying_test_data.shape}')
        print(f'y_test')
    # Combine predictions and true values into a DataFrame
    results_df = pd.DataFrame({
        'player_id': all_player_ids,
        'game_id': all_game_ids,
        'game_date': all_game_dates,
        'y': all_true_values,
        'y_pred': all_predictions,

    })
    results_df['player_name'] = results_df['player_id'].apply(
        lambda x: player_name_to_id[player_name_to_id['player_id'] == x].index[0])
    return results_df

def handle_missing_values(df):
    na_cols = df.columns[df.isna().any()].tolist()

    df = df.dropna(subset=['player_name'])
    df['start_position'] = df['start_position'].fillna('Bench')
    df['min'] = df['min'].fillna('0:00')  # Fill missing values with 0 minutes
    df['min'] = df['min'].apply(
        lambda x: int(x.split(':')[0].split('.')[0]) + int(x.split(':')[1]) / 60)  # Convert to numeric format
    df[na_cols] = df[na_cols].fillna(0)  # Fill missing values with 0

    cols_to_drop = ['nickname', 'comment', 'player_name', 'team_city']
    return df.drop(columns=cols_to_drop)


def add_time_dependent_features(df, rolling_window):
    # Add moving average columns:
    cols_for_ma = ['min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb',
                   'reb', 'ast', 'stl', 'blk', 'to', 'pf', 'pts', 'plus_minus', 'ast_pct', 'ast_ratio', 'ast_tov',
                   'def_rating', 'dreb_pct', 'efg_pct', 'e_def_rating', 'e_net_rating', 'e_off_rating', 'e_pace',
                   'e_usg_pct', 'net_rating', 'off_rating', 'oreb_pct', 'pace', 'pace_per40', 'pie', 'poss', 'reb_pct',
                   'tm_tov_pct', 'ts_pct', 'usg_pct']

    for col in cols_for_ma:
        df[f'{col}_rolling_{rolling_window}_day_avg'] = df.groupby('player_id')[col].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        df[f'{col}_rolling_{rolling_window}_day_std'] = df.groupby('player_id')[col].transform(lambda x: x.rolling(rolling_window, min_periods=1).std())

    # # Add lag columns
    # cols_for_lag = ['min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb',
    #                'reb', 'ast', 'stl', 'blk', 'to', 'pf', 'pts', 'plus_minus', 'ast_pct', 'ast_ratio', 'ast_tov',
    #                'def_rating', 'dreb_pct', 'efg_pct', 'e_def_rating', 'e_net_rating', 'e_off_rating', 'e_pace',
    #                'e_usg_pct', 'net_rating', 'off_rating', 'oreb_pct', 'pace', 'pace_per40', 'pie', 'poss', 'reb_pct',
    #                'tm_tov_pct', 'ts_pct', 'usg_pct']
    # for col in cols_for_lag:
    #     df[f'{col}_lag_1'] = df.groupby('player_id')[col].shift(1)
    #     df[f'{col}_lag_2'] = df.groupby('player_id')[col].shift(2)
    #     df[f'{col}_lag_3'] = df.groupby('player_id')[col].shift(3)
    #
    # # Add difference columns
    # cols_for_diff = ['min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb', 'dreb',
    #                'reb', 'ast', 'stl', 'blk', 'to', 'pf', 'pts', 'plus_minus', 'ast_pct', 'ast_ratio', 'ast_tov',
    #                'def_rating', 'dreb_pct', 'efg_pct', 'e_def_rating', 'e_net_rating', 'e_off_rating', 'e_pace',
    #                'e_usg_pct', 'net_rating', 'off_rating', 'oreb_pct', 'pace', 'pace_per40', 'pie', 'poss', 'reb_pct',
    #                'tm_tov_pct', 'ts_pct', 'usg_pct']
    # for col in cols_for_diff:
    #     df[f'{col}_diff_1'] = df.groupby('player_id')[col].diff(1)
    #     df[f'{col}_diff_2'] = df.groupby('player_id')[col].diff(2)
    #     df[f'{col}_diff_3'] = df.groupby('player_id')[col].diff(3)
    return df


def predict_fp(df, rolling_window=rolling_window):
    # Create a dictionary to map player names to player ids
    player_name_to_id = df.groupby('player_name')['player_id'].first()
    player_name_to_id = player_name_to_id.to_frame()
    player_name_to_id = player_name_to_id.reset_index()

    # Handle missing values
    df = handle_missing_values(df)

    # Preprocess specific columns
    df['game_date'] = pd.to_datetime(df['game_date'])
    cat_cols = ['team_abbreviation', 'player_id', 'start_position']
    df[cat_cols] = df[cat_cols].astype('category')

    # Reverse the order of the df
    df['week'] = df['game_date'].apply(lambda x: x.weekofyear)
    league_week_dict = {week: i for i, week in enumerate(df['week'].unique(), 1)}
    df['league_week'] = df['week'].map(league_week_dict)
    df = df.sort_values('league_week', ascending=False)

    df = add_time_dependent_features(df, rolling_window=rolling_window)

    # One hot encode categorical columns - currently not in use
    # encoder = OneHotEncoder(drop='first', sparse=False)
    # encoded_categorical = encoder.fit_transform(df[cat_cols])
    # encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(cat_cols))

    # df = pd.concat([df, encoded_categorical_df], axis=1)
    # df_with_player_id = df
    # df = df.drop(columns=cat_cols)
    # df.drop(columns=['comment'], inplace=True)

    league_week_gb = df.groupby('league_week')['game_date'].unique().reset_index()
    league_week_gb = league_week_gb.sort_values('league_week', ascending=False)

    total_results = {}
    for cat in dfs_cats:
        target = cat
        target_related_cols = same_game_cols
        features = df.columns.difference(target_related_cols).tolist()

        # Split data into features and target
        X = df[features]
        y = df[target]



        # Train the model
        print(f'Training models for {cat}')
        print('---------------------------------')
        results = rolling_train_test(X, y, league_week_gb, num_weeks_for_training=16, save_model=False,
                                     player_name_to_id=player_name_to_id)
        results.to_csv(f'output_csv/{cat}_results.csv', index=False)
        total_results.update({f'{cat}_full data': results})

    dfs = []
    # Iterate over the dictionary items
    for key, df in total_results.items():
        prefix = key.split('_')[0]
        df.rename(columns={'y': prefix, 'y_pred': f'{prefix}_pred'}, inplace=True)
        dfs.append(df)

    combined_df = dfs[0]

    for df in dfs[1:]:
        combined_df = pd.merge(combined_df, df, on=['game_id', 'player_id', 'player_name', 'game_date'], how='outer')

    # Add FP calculations:
    combined_df['fp_fanduel'] = combined_df.apply(
        lambda row: row['pts'] + row['reb'] * 1.2 + row['ast'] * 1.5 + row['stl'] * 3 + row['blk'] * 3 - row['to'] * 1,
        axis=1)
    combined_df['fp_fanduel_pred'] = combined_df.apply(
        lambda row: row['pts_pred'] + row['reb_pred'] * 1.2 + row['ast_pred'] * 1.5 + row['stl_pred'] * 3 + row[
            'blk_pred'] * 3 - row['to_pred'] * 1, axis=1)

    combined_df['fp_yahoo'] = combined_df.apply(calculate_fp_yahoo, axis=1)
    combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)

    combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)
    combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True), axis=1)

    return combined_df

df = pd.read_csv('output_csv/2021-22_game_by_game.csv')
df.columns = df.columns.str.lower()
res = predict_fp(df)
res.to_csv('fp_pred.csv')
