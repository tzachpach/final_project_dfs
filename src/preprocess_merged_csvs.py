import pandas as pd
import os

def preprocess_basic(df):
    df['starter'] = df['starter(y/n)'].apply(lambda x: 1 if x == 'Y' else 0)
    df['venue'] = df['venue(r/h)'].apply(lambda x: 1 if x == 'H' else 0)
    df['is_playoff'] = df['regular/playoffs'].apply(lambda x: 0 if x == 'Regular' else 0)
    df['is_wl'] = df['wl'].apply(lambda x: 0 if x == 'W' else 0)
    df['days_rest_int'] = df['days_rest'].astype(int)
    cols_to_drop = ['player_id', 'team_id', 'days_rest', 'starter(y/n)', 'venue(r/h)', 'regular/playoffs', 'wl']
    df = df.drop(cols_to_drop, axis=1)
    df['minutes_played'] = df['minutes_played'].fillna(0)  # Fill missing values with 0 minutes
    return df


def merge_all_seasons():
    dfs_to_merge = [pd.read_csv(f'data/{f}') for f in os.listdir('data/') if 'merged_gamelogs_salaries_' in f and f.endswith('.csv')]
    df = pd.concat(dfs_to_merge)
    df = df.drop('Unnamed: 0', axis= 1)
    df['venue(r/h)'] = df['venue(r/h)'].fillna(df['venue(r/h/n)'])
    df.drop(columns=['venue(r/h/n)'], inplace=True)
    df = df.dropna(subset=['available_flag'])
    df = preprocess_basic(df)
    return df