from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np
from codecs import open

'''
The content of this file is mostly copied from https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
'''

nlp = spacy.blank("en")
NULL_TOKEN_IDX = 0
OOV_TOKEN_IDX = 1


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


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


def process_file(filename, data_type, word_counter, char_counter):
    print(f"Generating {data_type} examples...")
    examples = []
    eval_examples = {}
    example_num = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        # iter articles
        for article in tqdm(source["data"]):
            # iter paragraphs
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)  # get tokens span in context
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                # iter question-answer pairs
                for qa in para["qas"]:
                    example_num += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    # iter answers
                    answer_start_token_idxs, answer_end_token_idxs = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_token_idx = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_token_idx.append(idx)

                        answer_start_token_idxs.append(answer_token_idx[0])
                        answer_end_token_idxs.append(answer_token_idx[-1])

                    example = {
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
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
                        "answers": answer_texts, "uuid": qa["id"]
                    }
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def get_embedding(counter, data_type, vec_size, emb_file=None):
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
        print(f"{len(embedding_dict)} / {len(counter)} available {data_type} embedding")
    else:
        for token in counter.keys():
            embedding_dict[token] = np.random.normal(size=(vec_size,), scale=0.1).tolist()
    print(f"{len(embedding_dict)} / {len(counter)} available {data_type} embedding")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = NULL_TOKEN_IDX
    token2idx_dict[OOV] = OOV_TOKEN_IDX
    embedding_dict[NULL] = np.zeros((vec_size,)).tolist()
    embedding_dict[OOV] = np.zeros((vec_size,)).tolist()
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict,
                   is_test=False):
    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    def validate(example):
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
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        if not is_test and not validate(example):
            continue

        context_idx = np.zeros((para_limit,), dtype=np.int32)
        context_char_idx = np.zeros((para_limit, char_limit), dtype=np.int32)
        ques_idx = np.zeros((ques_limit,), dtype=np.int32)
        ques_char_idx = np.zeros((ques_limit, char_limit), dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_idxs.append(context_idx)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_idxs.append(ques_idx)
        ques_char_idxs.append(ques_char_idx)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(
        file=out_file,
        context_idxs=np.array(context_idxs, dtype=np.long),
        context_char_idxs=np.array(context_char_idxs, dtype=np.long),
        ques_idxs=np.array(ques_idxs, dtype=np.long),
        ques_char_idxs=np.array(ques_char_idxs, dtype=np.long),
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


def preproc(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        filename=config.train_file,
        data_type="train",
        word_counter=word_counter,
        char_counter=char_counter
    )
    dev_examples, dev_eval = process_file(
        filename=config.dev_file,
        data_type="dev",
        word_counter=word_counter,
        char_counter=char_counter
    )
    # test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)

    word_emb_mat, word2idx_dict = get_embedding(
        counter=word_counter,
        data_type="word",
        emb_file=config.glove_emb,
        vec_size=config.word_emb_dim
    )
    char_emb_mat, char2idx_dict = get_embedding(
        counter=char_counter,
        data_type="char",
        emb_file=None,
        vec_size=config.char_emb_dim
    )

    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict,
                   char2idx_dict)
    build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict,
                   char2idx_dict)
    # build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    # save(config.test_eval_file, test_eval, message="test eval")
    save(config.word2idx_file, word2idx_dict, message="word dictionary")
    save(config.char2idx_file, char2idx_dict, message="char dictionary")
