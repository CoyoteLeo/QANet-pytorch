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


def build_features(config, examples, tokenizer: BertTokenizer, data_type, out_file, word2idx_dict,
                   char2idx_dict, is_test=False):
    ques_limit = config.ques_limit
    para_limit = tokenizer.max_len_sentences_pair - ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    def validate(example):
        if not example["y1s"] and not example["y2s"]:
            print()
        return len(example["context_tokens"]) <= para_limit \
               and len(example["ques_tokens"]) <= ques_limit \
               and (example["y2s"][0] - example["y1s"][0]) <= ans_limit

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return OOV_TOKEN_IDX

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return OOV_TOKEN_IDX

    print(f"Processing {data_type} examples...")
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    input_ids = []
    input_masks = []
    input_token_type_ids = []
    input_word_ids = []
    input_char_ids = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples), total=len(examples)):
        if not is_test and not validate(example):
            continue

        context_idx = np.zeros((para_limit,), dtype=np.int32)
        context_char_idx = np.zeros((para_limit, char_limit), dtype=np.int32)
        ques_idx = np.zeros((ques_limit,), dtype=np.int32)
        ques_char_idx = np.zeros((ques_limit, char_limit), dtype=np.int32)

        for i, token in enumerate(example["context_words"]):
            context_idx[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_idxs.append(context_idx)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_words"]):
            ques_idx[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_idxs.append(ques_idx)
        ques_char_idxs.append(ques_char_idx)

        # for bert
        context_word_id = []
        context_char_id = []
        for word_index in example["context_token_to_word_index"]:
            word = example['context_words'][word_index]
            context_word_id.append(_get_word(word))
            char_id = [NULL_TOKEN_IDX] * char_limit
            for j, char in enumerate(word):
                if j == char_limit:
                    break
                char_id[j] = _get_char(char)
            context_char_id.append(char_id)

        ques_word_id = []
        ques_char_id = []
        for word_index in example["ques_token_to_word_index"]:
            word = example['ques_words'][word_index]
            ques_word_id.append(_get_word(word))
            char_id = [NULL_TOKEN_IDX] * char_limit
            for j, char in enumerate(word):
                if j == char_limit:
                    break
                char_id[j] = _get_char(char)
            ques_char_id.append(char_id)

        input_word_id = [word2idx_dict[tokenizer.cls_token]] + \
                        context_word_id + \
                        [word2idx_dict[tokenizer.sep_token]] + \
                        ques_word_id + \
                        [word2idx_dict[tokenizer.sep_token]]
        input_char_id = [[char2idx_dict[tokenizer.cls_token]] * char_limit] + \
                        context_char_id + \
                        [[char2idx_dict[tokenizer.sep_token]] * char_limit] + \
                        ques_char_id + \
                        [[char2idx_dict[tokenizer.sep_token]] * char_limit]
        context_id = tokenizer.convert_tokens_to_ids(example['context_tokens'])
        query_id = tokenizer.convert_tokens_to_ids(example['ques_tokens'])
        input_id = tokenizer.build_inputs_with_special_tokens(context_id, query_id)
        input_token_type_id = tokenizer.create_token_type_ids_from_sequences(context_id,
                                                                             query_id)
        input_mask = [1] * len(input_id)
        while len(input_id) < tokenizer.max_len:
            input_word_id.append(NULL_TOKEN_IDX)
            input_id.append(tokenizer.pad_token_id)
            input_token_type_id.append(0)
            input_mask.append(0)
            input_char_id.append([NULL_TOKEN_IDX] * char_limit)
        assert len(input_id) == len(input_mask) == len(input_token_type_id) == len(
            input_word_id) == len(input_char_id) == tokenizer.max_len
        input_word_ids.append(input_word_id)
        input_char_ids.append(input_char_id)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        input_token_type_ids.append(input_token_type_id)

        start, end = example["y1s"][-1], example["y2s"][-1]
        start += 1
        end += 1
        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(
        file=out_file,
        context_idxs=np.array(context_idxs, dtype=np.long),
        context_char_idxs=np.array(context_char_idxs, dtype=np.long),
        ques_idxs=np.array(ques_idxs, dtype=np.long),
        ques_char_idxs=np.array(ques_char_idxs, dtype=np.long),

        # for bert
        input_ids=np.array(input_ids, dtype=np.long),
        input_masks=np.array(input_masks, dtype=np.long),
        input_token_type_ids=np.array(input_token_type_ids, dtype=np.long),
        input_word_ids=np.array(input_word_ids, dtype=np.long),
        input_char_ids=np.array(input_char_ids, dtype=np.long),

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


def preproc(config, force=True):
    tokenizer = BertTokenizer.from_pretrained(config.bert_type)
    word_counter, char_counter = Counter(), Counter()
    if not force and os.path.exists(config.train_example_file) and \
            os.path.exists(config.dev_example_file) and os.path.exists(config.train_eval_file) and \
            os.path.exists(config.dev_eval_file):
        with open(config.train_example_file, "rb") as fh:
            train_examples = json.load(fh)
        with open(config.dev_example_file, "rb") as fh:
            dev_examples = json.load(fh)
        with open(config.word2idx_file, "rb") as fh:
            word2idx_dict = json.load(fh)
        with open(config.char2idx_file, "rb") as fh:
            char2idx_dict = json.load(fh)
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
        word_emb_mat, word2idx_dict = get_embedding(
            counter=word_counter,
            tokenizer=tokenizer,
            data_type="word",
            emb_file=config.glove_emb,
            vec_size=config.word_emb_dim,
        )
        char_emb_mat, char2idx_dict = get_embedding(
            counter=char_counter,
            tokenizer=tokenizer,
            data_type="char",
            emb_file=None,
            vec_size=config.char_emb_dim
        )
        # test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)
        save(config.train_example_file, train_examples, message="train example")
        save(config.dev_example_file, dev_examples, message="dev example")
        save(config.train_eval_file, train_eval, message="train eval")
        save(config.dev_eval_file, dev_eval, message="dev eval")
        save(config.word2idx_file, word2idx_dict, message="word dictionary")
        save(config.char2idx_file, char2idx_dict, message="char dictionary")
        save(config.word_emb_file, word_emb_mat, message="word embedding")
        save(config.char_emb_file, char_emb_mat, message="char embedding")
        save("invalid_example.json", invalid, message="invalid example")

    build_features(config, train_examples, tokenizer, "train", config.train_record_file,
                   word2idx_dict, char2idx_dict)
    build_features(config, dev_examples, tokenizer, "dev", config.dev_record_file, word2idx_dict,
                   char2idx_dict)
    # build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # save(config.test_eval_file, test_eval, message="test eval")
