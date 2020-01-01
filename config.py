import os
import datetime

import torch
from absl import flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# global
BASE_DIR = os.path.expanduser(".")
SQUAD_VERSION = 'v1.1'
flags.DEFINE_string('squad_version', SQUAD_VERSION, '')
flags.DEFINE_string("mode", "debug", "train/debug/test")
flags.DEFINE_string("run_name", "", "")
flags.DEFINE_string("bert_type", "bert-base-multilingual-uncased", "")
flags.DEFINE_bool("with_no_answer", False, "")

# data
DATA_DIR = os.path.join(BASE_DIR, 'data', 'squad', SQUAD_VERSION)
flags.DEFINE_string("train_file", os.path.join(DATA_DIR, "train.json"), "")
flags.DEFINE_string("dev_file", os.path.join(DATA_DIR, "dev.json"), "")
flags.DEFINE_string("test_file", os.path.join(DATA_DIR, "dev.json"), "")
flags.DEFINE_string('glove_emb', os.path.join(BASE_DIR, 'data', "glove.840B.300d.txt"), '')

# preprocessing
flags.DEFINE_integer('word_emb_dim', 300, "")
flags.DEFINE_integer('char_emb_dim', 200, "")
flags.DEFINE_integer('para_limit', 400, "")
flags.DEFINE_integer('ques_limit', 50, "")
flags.DEFINE_integer('ans_limit', 30, "")
flags.DEFINE_integer('char_limit', 16, "")
PREPROCESS_DIR = os.path.join(DATA_DIR, 'preprocess')
flags.DEFINE_string("train_record_file", os.path.join(PREPROCESS_DIR, "train.npz"), "")
flags.DEFINE_string("dev_record_file", os.path.join(PREPROCESS_DIR, "dev.npz"), "")
flags.DEFINE_string("test_record_file", os.path.join(PREPROCESS_DIR, "test.npz"), "")
flags.DEFINE_string("word_emb_file", os.path.join(PREPROCESS_DIR, "word_emb.pkl"), "")
flags.DEFINE_string("char_emb_file", os.path.join(PREPROCESS_DIR, "char_emb.pkl"), "")
flags.DEFINE_string("train_example_file", os.path.join(PREPROCESS_DIR, "train_example.json"), "")
flags.DEFINE_string("dev_example_file", os.path.join(PREPROCESS_DIR, "dev_example.json"), "")
flags.DEFINE_string("train_eval_file", os.path.join(PREPROCESS_DIR, "train_eval.json"), "")
flags.DEFINE_string("dev_eval_file", os.path.join(PREPROCESS_DIR, "dev_eval.json"), "")
flags.DEFINE_string("test_eval_file", os.path.join(PREPROCESS_DIR, "test_eval.json"), "")
flags.DEFINE_string("word2idx_file", os.path.join(PREPROCESS_DIR, "word2idx.json"), "")
flags.DEFINE_string("char2idx_file", os.path.join(PREPROCESS_DIR, "char2idx.json"), "")

# model
flags.DEFINE_integer('global_hidden_size', 128, "")
flags.DEFINE_float('word_emb_dropout', 0.1, "")
flags.DEFINE_float('char_emb_dropout', 0.05, "")
flags.DEFINE_float('layer_dropout', 0.1, "")
flags.DEFINE_integer('emb_encoder_conv_num', 4, "")
flags.DEFINE_integer('emb_encoder_conv_kernel_size', 7, "")
flags.DEFINE_integer('emb_encoder_block_num', 2, "")
flags.DEFINE_integer('emb_encoder_ff_depth', 3, "")
flags.DEFINE_integer('output_encoder_conv_num', 2, "")
flags.DEFINE_integer('output_encoder_conv_kernel_size', 5, "")
flags.DEFINE_integer('output_encoder_block_num', 7, "")
flags.DEFINE_integer('output_encoder_ff_depth', 2, "")
flags.DEFINE_integer('attention_head_num', 8, "")

# train & test config
flags.DEFINE_integer('epoch_num', 40, "")
flags.DEFINE_integer('train_batch_size', 6, "")
flags.DEFINE_integer('eval_batch_size', 32, "")
flags.DEFINE_integer('checkpoint', 4440, "")
flags.DEFINE_float('lr', 0.001, "")
flags.DEFINE_integer('lr_warm_up_steps', 1000, "")
flags.DEFINE_float('adam_beta1', 0.8, "")
flags.DEFINE_float('adam_beta2', 0.999, "")
flags.DEFINE_float('adam_eps', 1e-7, "")
flags.DEFINE_float('adam_decay', 5e-8, "")
flags.DEFINE_float('ema_decay', 0.9999, "")
flags.DEFINE_integer('grad_clip', 10, "")
flags.DEFINE_integer("early_stop", 5, "Checkpoints for early stop")

# predict
RESULT_DIR = os.path.join(DATA_DIR, 'result')
flags.DEFINE_string("model_qanet_pretrain_file", os.path.join(RESULT_DIR, "model_qanet.pt"), "")
flags.DEFINE_string("model_bert_pretrain_file", os.path.join(RESULT_DIR, "model_bert.pt"), "")
flags.DEFINE_string("model_file", os.path.join(RESULT_DIR, "model.pt"), "")
flags.DEFINE_string("answer_file", os.path.join(RESULT_DIR, "answer.json"), "")
LOG_DIR = os.path.join(RESULT_DIR, f'log_{datetime.datetime.now().strftime("%Y%m%d%H%M")}')


config = flags.FLAGS

if not os.path.exists(PREPROCESS_DIR):
    os.makedirs(PREPROCESS_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
