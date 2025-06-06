"""Global configuration.

These configuration settings are related to hardware; they should not affect the network.

Exported settings:
device   Used by torch to select GPU/CPU.

Exported functions:
abort_handler    Catch Ctrl-C and set abort to True
check_abort      Returns true if the user wants to abort training
"""
import torch

MAX_LENGTH = 25

# characters that are mutated in preprocessing of strings
MC_PREFIXES = ['!', ']', '@']

# Special signs in dicts that give characters numeric values.
PAD_IDX = 0
SOS_token = 1
EOS_token = 2

# word level analysis (wla) url
wla_url = 'https://jakob.etcbc.vu.nl/api/wla'

# Data and models folders
TRAIN_DATA_FOLDER = '../data'
PREDICTION_DATA_FOLDER = '../new_data'
EVALUATION_RESULTS_PATH = '../evaluation_results_transformer'
MODEL_PATH = '../transformer_models'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abort = False  # Global variable to catch Ctrl-C


def abort_handler(signal_received, frame):
    """Catch Ctrl-C and set abort to True."""
    global abort
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    abort = True


def check_abort():
    """Check if abort is requested.

    NOTE: This needs to be a function, just importing abort and checking it will not work
    """
    global abort
    return abort
