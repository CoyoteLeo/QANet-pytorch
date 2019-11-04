import argparse
import os
from codecs import open
from collections import Counter
from typing import Tuple, Union

import numpy as np
import spacy
import ujson as json
from tqdm import tqdm

import config

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


class SQuAd:
    def __init__(self, version: str):
        self.version = version
        self.dir = os.path.join(config.SQUAD_DIR, self.version)
        self.word_counter = Counter()
        self.char_counter = Counter()

    def preprocess(self) -> None:
        train_examples, train_eval = self.process_file(os.path.join(self.dir, "dev.json"))
        dev_examples, dev_eval = self.process_file(os.path.join(self.dir, "train.json"))

        word_emb_mat, word2idx_dict = self.get_embedding(self.word_counter,
                                                         emb_file=config.GLOVE_WORD_REPRESENTATION,
                                                         vec_size=config.GLOVE_WORD_REPRESENTATION_DIM,
                                                         emb_type="word")
        char_emb_mat, char2idx_dict = self.get_embedding(self.char_counter, emb_file=None,
                                                         vec_size=config.CHAR_REPRESENTATION_DIM,
                                                         emb_type="char")

        self.build_features(os.path.join(self.dir, config.TRAIN_RECORD_FILE), train_examples,
                            word2idx_dict, char2idx_dict, "train")
        self.build_features(os.path.join(self.dir, config.DEV_RECORD_FILE), dev_examples,
                            word2idx_dict, char2idx_dict, "dev")

        save(os.path.join(self.dir, config.WORD_EMB_FILE), word_emb_mat, message="word embedding")
        save(os.path.join(self.dir, config.CHAR_EMB_FILE), char_emb_mat, message="char embedding")
        save(os.path.join(self.dir, config.TRAIN_EVAL_FILE), train_eval, message="train eval")
        save(os.path.join(self.dir, config.DEV_EVAL_FILE), dev_eval, message="dev eval")
        save(os.path.join(self.dir, config.WORD2IDX_FILE), word2idx_dict, message="word dictionary")
        save(os.path.join(self.dir, config.CHAR2IDX_FILE), char2idx_dict, message="char dictionary")

    def process_file(self, filename: str) -> Tuple[list, dict]:
        print(f"Generating {self.version} {filename} examples...")
        examples = list()
        eval_examples = dict()
        total = 0
        with open(filename, "r") as f:
            source = json.load(f)
            for article in tqdm(source["data"]):
                for para in article["paragraphs"]:
                    context = para["context"].replace("''", '" ').replace("``", '" ')
                    context_tokens = word_tokenize(context)
                    context_chars = [list(token) for token in context_tokens]
                    spans = convert_idx(context, context_tokens)
                    for token in context_tokens:
                        self.word_counter[token] += len(para["qas"])
                        for char in token:
                            self.char_counter[char] += len(para["qas"])
                    for qa in para["qas"]:
                        total += 1
                        ques = qa["question"].replace("''", '" ').replace("``", '" ')
                        ques_tokens = word_tokenize(ques)
                        ques_chars = [list(token) for token in ques_tokens]
                        for token in ques_tokens:
                            self.word_counter[token] += 1
                            for char in token:
                                self.char_counter[char] += 1
                        y1s, y2s = [], []
                        answer_texts = []
                        if len(qa["answers"]) == 0:
                            continue
                        for answer in qa["answers"]:
                            answer_text = answer["text"]
                            answer_start = answer['answer_start']
                            answer_end = answer_start + len(answer_text)
                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                            y1, y2 = answer_span[0], answer_span[-1]
                            y1s.append(y1)
                            y2s.append(y2)
                        example = {
                            "context_tokens": context_tokens,
                            "context_chars": context_chars,
                            "ques_tokens": ques_tokens,
                            "ques_chars": ques_chars,
                            "y1s": y1s,
                            "y2s": y2s,
                            "id": total
                        }
                        examples.append(example)
                        eval_examples[str(total)] = {
                            "context": context,
                            "spans": spans,
                            "answers": answer_texts,
                            "uuid": qa["id"]
                        }
            print(f"{len(examples)} questions in total")
        return examples, eval_examples

    def get_embedding(self, counter: Counter, emb_file: Union[None, str], vec_size: int,
                      emb_type: str, limit: int = -1) -> Tuple[list, dict]:
        print(f"Generating {self.version} {emb_type} embedding...")
        embedding_dict = {}
        filtered_elements = [key for key, count in counter.items() if count > limit]
        if emb_file is not None:
            # use pre-trained embedding vector
            with open(emb_file, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    array = line.split()
                    word = "".join(array[0:-vec_size])
                    vector = list(map(float, array[-vec_size:]))
                    if word in counter and counter[word] > limit:
                        embedding_dict[word] = vector
            print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have "
                  f"corresponding {emb_type} embedding vector")
        else:
            # randomly initial embedding vector
            for token in filtered_elements:
                embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
            print(f"{len(filtered_elements)} tokens have corresponding embedding vector")

        null_token = "--NULL--"
        oov_token = "--OOV--"
        token2idx_dict = {
            token: idx for idx, token in enumerate(embedding_dict.keys(),
                                                   max(config.NULL_INDEX, config.OOV_INDEX))
        }
        token2idx_dict[null_token] = config.NULL_INDEX
        token2idx_dict[oov_token] = config.OOV_INDEX
        embedding_dict[null_token] = [0. for _ in range(vec_size)]
        embedding_dict[oov_token] = [0. for _ in range(vec_size)]
        idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
        emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
        return emb_mat, token2idx_dict

    def build_features(self, out_file: str, examples: list, word2idx_dict: dict,
                       char2idx_dict: dict, data_type: str):
        print(f"Processing {self.version} {data_type} examples...")
        para_limit = config.PARA_LIMIT
        ques_limit = config.QUES_LIMIT
        ans_limit = config.ANS_LIMIT
        char_limit = config.CHAR_LIMIT

        def validate(qa_example):
            return len(qa_example["context_tokens"]) <= para_limit and \
                   len(qa_example["ques_tokens"]) <= ques_limit and \
                   (qa_example["y2s"][0] - qa_example["y1s"][0]) <= ans_limit

        def _get_word(_word):
            return word2idx_dict.get(_word) or word2idx_dict.get(
                _word.lower()) or word2idx_dict.get(_word.capitalize()) \
                   or word2idx_dict.get(_word.upper(), config.OOV_INDEX)

        def _get_char(_char):
            return char2idx_dict.get(_char, config.OOV_INDEX)

        total = 0
        total_valid = 0
        context_idxs = []
        context_char_idxs = []
        ques_idxs = []
        ques_char_idxs = []
        y1s = []
        y2s = []
        ids = []
        for n, example in tqdm(enumerate(examples)):
            total_valid += 1
            if not validate(example):
                continue
            total += 1

            context_idx = np.zeros([para_limit], dtype=np.int32)
            context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
            ques_idx = np.zeros([ques_limit], dtype=np.int32)
            ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

            for i, token in enumerate(example["context_tokens"]):
                context_idx[i] = _get_word(token)
            context_idxs.append(context_idx)

            for i, token in enumerate(example["ques_tokens"]):
                ques_idx[i] = _get_word(token)
            ques_idxs.append(ques_idx)

            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idx[i, j] = _get_char(char)
            context_char_idxs.append(context_char_idx)

            for i, token in enumerate(example["ques_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idx[i, j] = _get_char(char)
            ques_char_idxs.append(ques_char_idx)

            start, end = example["y1s"][-1], example["y2s"][-1]
            y1s.append(start)
            y2s.append(end)
            ids.append(example["id"])

        np.savez(out_file, context_idxs=np.array(context_idxs),
                 context_char_idxs=np.array(context_char_idxs),
                 ques_idxs=np.array(ques_idxs),
                 ques_char_idxs=np.array(ques_char_idxs),
                 y1s=np.array(y1s),
                 y2s=np.array(y2s), ids=np.array(ids))
        print(f"Built {total} / {total_valid} instances of features in total")


if __name__ == "__main__":
    if not os.path.exists(config.GLOVE_WORD_REPRESENTATION):
        raise Exception("please download glove embedding file")
    squad = SQuAd(version="v1.1")
    squad.preprocess()
