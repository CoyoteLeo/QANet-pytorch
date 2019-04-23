import os

BASE_DIR = os.path.expanduser(".")

# data config
DATA_DIR = os.path.join(BASE_DIR, "data")

GLOVE_DIR = os.path.join(DATA_DIR, "glove")
GLOVE_WORD_REPRESENTATION = os.path.join(GLOVE_DIR, "glove.840B.300d.txt")
GLOVE_WORD_REPRESENTATION_DIM = 300
CHAR_REPRESENTATION_DIM = 200

SQUAD_DIR = os.path.join(DATA_DIR, "squad")
ALL_SUPPORT_SQUAD_VERSION = ["v1.1", "v2.0"]

# limit
PARA_LIMIT = 400  # Limit length for paragraph
QUES_LIMIT = 50  # Limit length for question
ANS_LIMIT = 30  # Limit length for answers
CHAR_LIMIT = 16

# embedding
NULL_INDEX = 0
OOV_INDEX = 1

# preprocessing result
TRAIN_RECORD_FILE = "train.npz"
DEV_RECORD_FILE = "dev.npz"
WORD_EMB_FILE = "word_emb.json"
CHAR_EMB_FILE = "char_emb.json"
TRAIN_EVAL_FILE = "train_eval.json"
DEV_EVAL_FILE = "dev_eval.json"
WORD2IDX_FILE = "word2idx.json"
CHAR2IDX_FILE = "char2idx.json"

# model config
HIDDEN_SIZE = 128
WORD_EMBEDDING_DROPOUT = 0.1
CHAR_EMBEDDING_DROPOUT = 0.05
LAYERS_DROPOUT = 0.1
