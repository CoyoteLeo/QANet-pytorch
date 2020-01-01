import os
import unicodedata

import re
from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np
from codecs import open

from transformers import BertTokenizer
from transformers.tokenization_bert import whitespace_tokenize

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
'''

nlp = spacy.blank("en")
NULL_TOKEN_IDX = 0
OOV_TOKEN_IDX = 1
CLS_TOKEN_IDX = 2
SEP_TOKEN_IDX = 3
invalid = []


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def strip_accents(text, ignore=False):
    """Strips accents from a piece of text."""
    skip_count = 0
    new_text = unicodedata.normalize("NFD", text)
    if not ignore:
        return new_text
    output = []
    for char in new_text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            skip_count += 1
            continue
        output.append(char)
    output_text = "".join(output)
    return output_text


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            raise Exception(f"Token {token} cannot be found")
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def convert_idx_bert(text, tokens, tokenizer: BertTokenizer):
    text = strip_accents(text)
    current = 0
    spans = []
    last_is_unknown = False
    for token in tokens:
        if token.startswith('##'):
            token = token[2:]
        if token == tokenizer.unk_token:
            last_is_unknown = True
            continue
        current = text.find(token, current)
        if current < 0:
            raise Exception(f"Token {token} cannot be found")
        if last_is_unknown:
            remain = text[spans[-1][1]:].encode()
            first_not_space = 0
            for byte in remain:
                if byte != ' '.encode():
                    break
                first_not_space += 1
            start = spans[-1][1] + first_not_space
            remain = text[spans[-1][1]:current].encode()[::-1]
            first_not_space = 0
            for byte in remain:
                if byte != ' '.encode():
                    break
                first_not_space += 1
            end = current - first_not_space
            if end <= start:
                raise Exception('end before start')
            spans.append((start, end))
            last_is_unknown = False
        spans.append((current, current + len(token)))
        current += len(token)
    if last_is_unknown:
        spans.append((spans[-1][1] + 1, len(text)))
    return spans


def process_file(filename, tokenizer, data_type, word_counter, char_counter):
    print(f"Generating {data_type} examples...")
    bad_pair = 0
    examples = []
    eval_examples = {}
    example_num = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        # iter articles
        for article in tqdm(source["data"]):
            # if article['title'] != 'Oklahoma':
            #     continue
            # iter paragraphs
            for para in article["paragraphs"]:
                raw_context = para["context"].replace("''", '" ').replace("``", '" ')
                context_chars = []
                context_orig_to_char_idx = []
                for index, char in enumerate(raw_context):
                    new_char = strip_accents(char)
                    context_orig_to_char_idx.append(len(context_chars))
                    context_chars.append(new_char[:1])
                    for output in new_char[1:]:
                        context_chars.append(output)
                context = "".join(context_chars)
                context_token_to_word_index = []
                context_tokens = []
                # avoid muti-space between words
                context_words = word_tokenize(' '.join(context.split()))
                # get tokens span in context
                raw_spans = convert_idx(context, context_words)
                spans = []
                for index, (word, span) in enumerate(zip(context_words, raw_spans)):
                    word = word.lower()
                    ignore_char_idx = []
                    for i in range(len(word)):
                        cat = unicodedata.category(word[i])
                        if cat == "Mn":
                            ignore_char_idx.append(i)
                    start = raw_spans[index][0]
                    last_is_unk = False
                    for token in tokenizer.tokenize(word):
                        context_token_to_word_index.append(index)
                        context_tokens.append(token)
                        if token.startswith('##'):
                            token = token[2:]
                        if token == tokenizer.unk_token:
                            last_is_unk = True
                            continue
                        if last_is_unk:
                            end = start + word.find(token, start - span[0]) - (start - span[0])
                            spans.append((start, end))
                            start = end
                            last_is_unk = False
                            while ignore_char_idx and ignore_char_idx[0] < start - span[0]:
                                ignore_char_idx = ignore_char_idx[1:]
                        end = start + len(token)
                        while ignore_char_idx and \
                                (ignore_char_idx[0] < start - span[0] + len(token) or
                                 start - span[0] + len(token) == ignore_char_idx[0]):
                            ignore_char_idx = ignore_char_idx[1:]
                            end += 1
                        spans.append((start, end))
                        start = end
                        if start < span[0] or end > span[1]:
                            raise Exception()
                    if last_is_unk:
                        spans.append((start, span[1]))
                # TODO: char ignore '##' from start(it doesn't skip now)
                context_chars = [list(token) for token in context_tokens]
                for token in context_words:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                # iter question-answer pairs
                for qa in para["qas"]:
                    example_num += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ').lower()
                    ques_token_to_word_index = []
                    ques_tokens = []
                    ques_words = word_tokenize(ques)
                    for index, word in enumerate(ques_words):
                        for token in tokenizer.tokenize(word):
                            ques_token_to_word_index.append(index)
                            ques_tokens.append(token)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_words:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1

                    # iter answers
                    answer_start_token_idxs, answer_end_token_idxs = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = strip_accents(answer["text"])
                        answer_start = context_orig_to_char_idx[answer['answer_start']]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_token_idx = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_token_idx.append(idx)
                        ans_from_span = context[spans[answer_token_idx[0]][0]:
                                                spans[answer_token_idx[-1]][1]]
                        if ans_from_span.find(answer_text) != 0:
                            invalid.append(
                                [context, ques, answer_text, answer_start, ans_from_span])
                            continue
                            # raise Exception()
                        answer_start_token_idxs.append(answer_token_idx[0])
                        answer_end_token_idxs.append(answer_token_idx[-1])

                    if not answer_start_token_idxs or not answer_end_token_idxs:
                        bad_pair += 1
                        continue

                    example = {
                        "context_words": context_words,
                        "context_token_to_word_index": context_token_to_word_index,
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "ques_words": ques_words,
                        "ques_token_to_word_index": ques_token_to_word_index,
                        "ques_tokens": ques_tokens,
                        "ques_chars": ques_chars,
                        "y1s": answer_start_token_idxs,
                        "y2s": answer_end_token_idxs,
                        "id": example_num,
                    }
                    examples.append(example)
                    eval_examples[str(example_num)] = {
                        "context": context,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa["id"]
                    }
        print(f"{len(examples)}/{len(examples) + bad_pair} questions in total")
    return examples, eval_examples


def get_embedding(counter, tokenizer, data_type, vec_size, emb_file=None):
    print(f"Generating {data_type} embedding...")
    embedding_dict = {}
    if emb_file is not None:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter:
                    embedding_dict[word] = vector
    else:
        for token in counter.keys():
            embedding_dict[token] = np.random.normal(size=(vec_size,), scale=0.1).tolist()
    print(f"{len(embedding_dict)} / {len(counter)} available {data_type} embedding")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 4)}
    token2idx_dict[tokenizer.cls_token] = CLS_TOKEN_IDX
    token2idx_dict[tokenizer.sep_token] = SEP_TOKEN_IDX
    token2idx_dict[NULL] = NULL_TOKEN_IDX
    token2idx_dict[OOV] = OOV_TOKEN_IDX
    embedding_dict[tokenizer.cls_token] = np.zeros((vec_size,)).tolist()
    embedding_dict[tokenizer.sep_token] = np.zeros((vec_size,)).tolist()
    embedding_dict[NULL] = np.zeros((vec_size,)).tolist()
    embedding_dict[OOV] = np.zeros((vec_size,)).tolist()
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, tokenizer: BertTokenizer, data_type, out_file, is_test=False):
    ques_limit = config.ques_limit
    para_limit = tokenizer.max_len_sentences_pair - ques_limit
    ans_limit = config.ans_limit

    def validate(example):
        if not example["y1s"] and not example["y2s"]:
            print()
        return len(example["context_tokens"]) <= para_limit \
               and len(example["ques_tokens"]) <= ques_limit \
               and (example["y2s"][0] - example["y1s"][0]) <= ans_limit

    print(f"Processing {data_type} examples...")
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples), total=len(examples)):
        if not is_test and not validate(example):
            continue

        context_idx = tokenizer.convert_tokens_to_ids(example['context_tokens'])
        while len(context_idx) < para_limit:
            context_idx.append(tokenizer.pad_token_id)
        context_idxs.append(context_idx)

        ques_idx = tokenizer.convert_tokens_to_ids(example['ques_tokens'])
        while len(ques_idx) < para_limit:
            ques_idx.append(tokenizer.pad_token_id)
        ques_idxs.append(ques_idx)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(
        file=out_file,
        context_idxs=np.array(context_idxs, dtype=np.long),
        ques_idxs=np.array(ques_idxs, dtype=np.long),
        y1s=np.array(y1s, dtype=np.long),
        y2s=np.array(y2s, dtype=np.long),
        ids=np.array(ids, dtype=np.long)
    )
    print(f"{len(ids)} / {len(examples)} available features")


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def preproc(config, force=False):
    tokenizer = BertTokenizer.from_pretrained(config.bert_type)
    word_counter, char_counter = Counter(), Counter()
    if not force and os.path.exists(config.train_example_file) and \
            os.path.exists(config.dev_example_file) and os.path.exists(config.train_eval_file) and \
            os.path.exists(config.dev_eval_file):
        with open(config.train_example_file, "rb") as fh:
            train_examples = json.load(fh)
        with open(config.dev_example_file, "rb") as fh:
            dev_examples = json.load(fh)
    else:
        train_examples, train_eval = process_file(
            filename=config.train_file,
            tokenizer=tokenizer,
            data_type="train",
            word_counter=word_counter,
            char_counter=char_counter
        )
        dev_examples, dev_eval = process_file(
            filename=config.dev_file,
            tokenizer=tokenizer,
            data_type="dev",
            word_counter=word_counter,
            char_counter=char_counter
        )
        # test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)
        save(config.train_example_file, train_examples, message="train example")
        save(config.dev_example_file, dev_examples, message="dev example")
        save(config.train_eval_file, train_eval, message="train eval")
        save(config.dev_eval_file, dev_eval, message="dev eval")
        save("invalid_example.json", invalid, message="invalid example")

    build_features(config, train_examples, tokenizer, "train", config.train_record_file)
    build_features(config, dev_examples, tokenizer, "dev", config.dev_record_file)
    # build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # save(config.test_eval_file, test_eval, message="test eval")
