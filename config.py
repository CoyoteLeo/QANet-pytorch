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
ENCODE_CONVOLUTION_NUMBER = 4
EMBEDDING_ENCODER_CONVOLUTION_KERNEL_SIZE = 7
ATTENTION_HEAD_NUMBER = 8

# train & test config
# step means epoch
STEPS = 60000
VALIDATION_STEPS = 150
TEST_STEPS = 150
CHECKPOINT = 200
BATCH_SIZE = 8  # Batch size
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.8
ADAM_BETA2 = 0.999
LEARNING_RATE_WARM_UP_STEPS = 1000
BASE_LEARNING_RATE = 1.0
EMA_DECAY = 0.9999
EARLY_STOP = 1000
