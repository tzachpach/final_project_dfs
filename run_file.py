from final_project_dfs.data_loader import data_loader
from final_project_dfs.fetch_y_pred import predict_dkfp
from final_project_dfs.genetic_selection import get_lineup
from final_project_dfs.post_process import post_process_results

if __name__ == '__main__':
    data = data_loader(seasons_to_reduce=8)
    perdiced_data = predict_dkfp(data)
    res = get_lineup(perdiced_data)
    per_row, gen_res = post_process_results(res)
    res.to_csv('genetic_selection.csv', index=False)


# TODO: understand the current bug with the repeating players
# TODO: start talking to mentors


# TODO: read up on deep q learning
# TODO: understand why RL and if RL is really needed to solve

# TODO: get better data - ctg
# TODO: get better data - basketball reference
# TODO: get better data - fanduel
# TODO: get better data - draftkings
