import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
from dateutil import rrule



common_columns = ['date', 'player_name', 'team', 'opp.', 'score']
not_common_columns = ['yahoo_position', 'yahoo_points', 'yahoo_salary',
                      'fanduel_position', 'fanduel_points', 'fanduel_salary', 'draftkings_position', 'draftkings_points', 'draftkings_salary']

end_cols = common_columns + not_common_columns
games = ['yahoo', 'fanduel', 'draftkings']
existing_dfs_data = pd.DataFrame(columns=end_cols)

def merge_dfs(dfs_dict):
    # Get the common columns from the first dataframe in the dictionary
    # Merge all dataframes on common columns using an outer join
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_columns, how='outer'), dfs_dict.values())
    for col in not_common_columns:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
    return merged_df


def pull_historic_dfs_stat(start_date, end_date, existing_dfs_data, games=games):
    dates = [dt.date().strftime('%Y-%m-%d') for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date)]
    existing_dates = list(existing_dfs_data['date'].astype(str))
    new_dates = sorted(list(set(dates) - set(existing_dates)))
    # print(f'new dates = {new_dates}')
    erros = {d: [] for d in new_dates}

    for date in new_dates:
        day = date[8:10]
        month = date[5:7]
        year = date[0:4]
        games_salaries_per_date = {}

        for game in games:
            html_string = f'http://rotoguru1.com/cgi-bin/hyday.pl?game={game}&mon={month}&day={day}&year={year}'

            try:
                html_data = pd.read_html(html_string)
                salary_df = html_data[5]

                # Rename columns
                cols = [i.replace(" ", "_").lower() for i in salary_df.iloc[1]]
                cols[0] = f'{game}_position'
                cols[1] = 'player_name'
                cols[3] = f'{game}_salary'
                salary_df.columns = cols
                # Clean data
                salary_df[f'{game}_salary'] = salary_df[f'{game}_salary'].fillna('')
                salary_df = salary_df[salary_df[f'{game}_salary'].str.contains('^\$', regex=True)]
                salary_df['player_name'] = salary_df['player_name'].str.replace('^', '', regex=True)

                # Add date column
                salary_df['date'] = date

                # Process salary and player name
                salary_df[f'{game}_salary'] = salary_df[f'{game}_salary'].str.replace('$', '').str.replace(',', '').astype(float)
                salary_df['player_name'] = salary_df['player_name'].str.split(', ').str[::-1].str.join(' ')

                games_salaries_per_date.update({game: salary_df})
            except Exception as e:
                print(f"Failed to process date {date} and game {game}: {e}")
                erros[date].append(game)
                continue
        if not games_salaries_per_date:
            print(f"Failed to process date {date} - no games found")
            continue


        merged_df = merge_dfs(games_salaries_per_date)
        existing_dfs_data = pd.concat([existing_dfs_data, merged_df[end_cols]], ignore_index=True).drop_duplicates()
        existing_dfs_data = existing_dfs_data.sort_values('date').drop_duplicates(['date', 'player_name']).reset_index(drop=True)

        print(f"Processed date: {date}, Num results: {len(merged_df)}")


    existing_dfs_data = existing_dfs_data.sort_values('date', ascending=False).reset_index(drop=True)
    existing_dfs_data.to_csv(f'output_csv/salaries_data_{start_date}-{end_date}.csv', index=False)
    return existing_dfs_data


start_date = datetime.strptime('2016-10-25', '%Y-%m-%d')
end_date = datetime.strptime('2017-06-18', '%Y-%m-%d')

pull_historic_dfs_stat(start_date, end_date, existing_dfs_data)
