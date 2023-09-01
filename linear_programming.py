import pulp

# create the LP object, set up as a maximization problem
prob = pulp.LpProblem('FantasyBasketball', pulp.LpMaximize)

# create decision variables
player_vars = pulp.LpVariable.dicts('Player', players, cat='Binary')

# objective function: maximize the sum of predicted points of selected players
prob += pulp.lpSum([y_pred[i] * player_vars[i] for i in players])

# constraint: do not exceed budget
prob += pulp.lpSum([cost[i] * player_vars[i] for i in players]) <= budget

# position constraints
# e.g., for PG: prob += pulp.lpSum([player_vars[i] for i in PGs]) >= 1

# solve the problem
status = prob.solve()

# print the results
selected_players = [i for i in players if pulp.value(player_vars[i]) == 1]
print('Selected players:', selected_players)
print('Total predicted points:', pulp.value(prob.objective))
