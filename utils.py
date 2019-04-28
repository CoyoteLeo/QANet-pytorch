import random
import re
import string
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SQuADDataset(Dataset):
    def __init__(self, npz_file, num_steps, batch_size):
        super().__init__()
        data = np.load(npz_file)
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()
        num = len(self.ids)
        self.batch_size = batch_size
        self.num_steps = num_steps if num_steps >= 0 else num // batch_size
        num_items = num_steps * batch_size
        idxs = list(range(num))
        self.idx_map = []
        i, j = 0, num

        while j <= num_items:
            random.shuffle(idxs)
            self.idx_map += idxs.copy()
            i = j
            j += num
        random.shuffle(idxs)
        self.idx_map += idxs[:num_items - i]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
        res = (self.context_idxs[idxs],
               self.context_char_idxs[idxs],
               self.ques_idxs[idxs],
               self.ques_char_idxs[idxs],
               self.y1s[idxs],
               self.y2s[idxs], self.ids[idxs])
        return res


class EMA(object):
    """
    exponential moving average
    """

    def __init__(self, decay):
        self.decay = decay
        self.shadows = {}
        self.devices = {}

    def __len__(self):
        return len(self.shadows)

    def get(self, name: str):
        return self.shadows[name].to(self.devices[name])

    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data.to('cpu').clone()
        self.devices[name] = param.data.device

    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow.to('cpu').clone()


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
            exact_match += Evaluator.metric_max_over_ground_truths(Evaluator.exact_match_score, prediction,
                                                                   ground_truths)
            f1 += Evaluator.metric_max_over_ground_truths(Evaluator.f1_score, prediction, ground_truths)
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return {'exact_match': exact_match, 'f1': f1}
