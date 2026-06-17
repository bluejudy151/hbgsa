from pathlib import Path

RUN_SEED = 7342
BATCH_SIZE = 32
N_EPOCH = 20
SAVE_BEST_RATIO = 0.4
TOP_K_MODELS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

SMI_EMBED_SIZE = 128
SEQ_EMBED_SIZE = 128
SEQ_OUT_CHANNELS = 128
PKT_OUT_CHANNELS = 128
SMI_OUT_CHANNELS = 128
HBOND_OUT_DIM = 128
DROPOUT_RATE = 0.2
CLASSIFIER_DROPOUT = 0.5

MAX_SEQ_LEN = 1000
MAX_PKT_LEN = 63
MAX_SMI_LEN = 150
MAX_HBONDS = 20

DATA_PATH = Path('./data')
RUNS_PATH = Path('./runs')
HBOND_CSV_PATH = Path('./data/Bond/hbond_3d_flattened.csv')

NUM_WORKERS = 0

CHAR_SMI_SET = {
    "(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
    "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
    "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
    "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
    "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
    "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
    "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
    "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64
}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40
