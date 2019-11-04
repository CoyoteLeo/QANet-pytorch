import re
import string
from collections import Counter

import numpy as np
from torch.utils.data import Dataset


class SQuADDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.context_idxs = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs = data["ques_idxs"]
        self.ques_char_idxs = data["ques_char_idxs"]
        self.y1s = data["y1s"]
        self.y2s = data["y2s"]
        self.ids = data["ids"]
        self.num = len(self.ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.context_idxs[idx], self.context_char_idxs[idx], self.ques_idxs[idx], \
               self.ques_char_idxs[idx], self.y1s[idx], self.y2s[idx], self.ids[idx]


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1 + num_updates) / (10 + num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class Evaluator(object):

    def __init__(self, eval_file):
        self.result = dict()
        self.eval_file = eval_file

    @staticmethod
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = Evaluator.normalize_answer(prediction).split()
        ground_truth_tokens = Evaluator.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return Evaluator.normalize_answer(prediction) == Evaluator.normalize_answer(ground_truth)

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def convert_tokens(self, qa_ids, pp1, pp2):
        answer_dict = {}
        remapped_dict = {}
        for qid, p1, p2 in zip(qa_ids, pp1, pp2):
            context = self.eval_file[str(qid)]["context"]
            spans = self.eval_file[str(qid)]["spans"]
            uuid = self.eval_file[str(qid)]["uuid"]
            l = len(spans)
            if p1 >= l or p2 >= l:
                ans = ""
            else:
                start_idx = spans[p1][0]
                end_idx = spans[p2][1]
                ans = context[start_idx: end_idx]
            answer_dict[str(qid)] = ans
            remapped_dict[uuid] = ans
        return answer_dict, remapped_dict

    def update_result(self, qa_ids, pp1, pp2):
        answer_dict, _ = self.convert_tokens(qa_ids, pp1, pp2)
        self.result.update(answer_dict)
        return self.result

    def evaluate(self, eval_file=None):
        eval_file = eval_file or self.eval_file
        assert eval_file is not None, "eval file can't be None"
        f1 = exact_match = total = 0
        for key, value in self.result.items():
            total += 1
            ground_truths = eval_file[key]["answers"]
            prediction = value
            exact_match += Evaluator.metric_max_over_ground_truths(Evaluator.exact_match_score,
                                                                   prediction, ground_truths)
            f1 += Evaluator.metric_max_over_ground_truths(Evaluator.f1_score, prediction,
                                                          ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return {'exact_match': exact_match, 'f1': f1}


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # exclude.update('，', '。', '、', '；', '「', '」')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}
