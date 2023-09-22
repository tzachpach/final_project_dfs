import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def predict_dkfp(train_data, should_train=False, should_plot=True):
    label_encoders = {}
    train_data['year'] = train_data['date'].dt.year
    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.day
    train_data['day_of_week'] = train_data['date'].dt.dayofweek

    columns_to_exclude = ['dkfp', 'cost', 'date']
    X = train_data.drop(columns=columns_to_exclude)
    y = train_data[['dkfp', 'cost']]

    categorical_columns = ["player", "team", "against", "pos", 'season']
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    cost = y['cost']
    y = y.drop('cost', axis=1)

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    if should_train:
        print("Training model...")
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=tscv,
                                   verbose=1, n_jobs=-1)

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'xgboost_best_.pkl')

    else:
        print("Loading model...")
        best_model = joblib.load('xgboost_best_.pkl')

    y_pred = best_model.predict(X_test)

    print("Predicted DKFP values:", y_pred)

    # Inverse transform the label encoded columns
    test_df = X_test.copy()
    for col in categorical_columns:
        test_df[col] = label_encoders[col].inverse_transform(test_df[col])

    test_df['y_pred'] = y_pred
    test_df['cost'] = cost.iloc[test_index]
    test_df['dkfp'] = y_test['dkfp']

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)

    if should_plot:
        plot_results(y_test['dkfp'], y_pred)
    return test_df

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual DKFP')
    plt.ylabel('Predicted DKFP')
    plt.title('Actual vs Predicted DKFP')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=(y_test - y_pred))
    plt.xlabel('Actual DKFP')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual DKFP')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(y_test - y_pred, kde=True)
    plt.xlabel('Residuals')
    plt.title('Histogram of Residuals')
    plt.show()