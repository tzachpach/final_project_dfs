import torch
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

rolling_window = 10

thresholds_for_exceptional_games = {
    "pts": 20,  # High-scoring game threshold
    "reb": 10,  # High-rebounding game threshold
    "ast": 8,  # High-assist game threshold
    "stl": 5,  # High-steal game threshold
    "blk": 5,  # High-block game threshold
    "tov": 5,  # High-turnover game threshold
    "fp_draftkings": 50,  # Exceptional DFS performance for DraftKings
    "fp_fanduel": 45,  # Exceptional DFS performance for FanDuel
    "fp_yahoo": 40,  # Exceptional DFS performance for Yahoo
}

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
            "UTIL": 1,  # Any position
        },
    },
    "fanduel": {
        "salary_cap": 60000,
        "positions": {
            "PG": 2,
            "SG": 2,
            "SF": 2,
            "PF": 2,
            "C": 1,
            "UTIL": 0,  # Any position
        },
    },
    "draftkings": {
        "salary_cap": 50000,
        "positions": {
            "PG": 1,
            "SG": 1,
            "SF": 1,
            "PF": 1,
            "C": 1,
            "G": 1,  # Guard (PG or SG)
            "F": 1,  # Forward (SF or PF)
            "UTIL": 1,  # Any position
        },
    },
}

rnn_param_grid = {
    "step_size": [1],
    "hidden_size": [32],
    "num_layers": [1, 5],
    "dropout_rate": [0.2],
    "learning_rate": [0.001, 0.01],
    "rnn_type": ["LSTM", "GRU"],
    "train_window": [4, 8],
    "epochs": [15, 20],
    "batch_size": [32, 64],
    "lookback": [4, 8],
    "salary_threshold": [0.4, 0.6],
}

best_params = {
    "step_size": 1,
    "hidden_size": 64,
    "num_layers": 3,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "rnn_type": "LSTM",
    "train_window": 4,
    "epochs": 20,
    "batch_size": 32,
    "lookback": 4,
    "salary_threshold": 0.6,
}


def select_device():
    """
    Chooses 'cuda' if available, else 'mps' (Apple Silicon) if available and functional, else CPU.
    Prints which device is selected.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS is actually functional (sometimes available but not usable)
        try:
            x = torch.ones(1, device="mps")
            device = torch.device("mps")
            print("Using device: mps")
        except Exception:
            device = torch.device("cpu")
            print("Using device: cpu (MPS not functional)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device
