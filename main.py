import math
import os

import numpy as np
import torch
import torch.cuda
import ujson as json
from absl import app
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.nn import functional as F
from transformers import AdamW

from config import config, LOG_DIR, device
from preproc_bert import preproc
# from preproc import preproc
from utils import EMA, SQuADDataset, convert_tokens, evaluate

writer = SummaryWriter(log_dir=LOG_DIR)
loss_func = F.cross_entropy


def test(config, model, global_step=0, validate=False):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = SQuADDataset(config.dev_record_file)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                batch_size=config.eval_batch_size,
                                num_workers=config.eval_batch_size)

    print("\nValidate" if validate else "Test")
    model.eval()
    answer_dict = {}
    losses = []
    with torch.no_grad():
        for step, (input_word_ids, input_char_id, input_id, input_mask, input_token_type_id, y1, y2,
                   ids) in enumerate(dev_dataloader):
            input_word_ids, input_char_id, input_id, input_mask, input_token_type_id = \
                input_word_ids.to(device), input_char_id.to(device), input_id.to(
                    device), input_mask.to(device), input_token_type_id.to(device)
            p1, p2 = model(input_word_ids, input_char_id, input_id, input_mask, input_token_type_id)
            y1, y2 = y1.to(device), y2.to(device)
            loss = loss_func(p1, y1) + loss_func(p2, y2)
            losses.append(loss.item())

            p1 = F.softmax(p1, dim=1)
            p2 = F.softmax(p2, dim=1)
            outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            for j in range(outer.size()[0]):
                outer[j] = torch.triu(outer[j])
            a1, _ = torch.max(outer, dim=2)
            a2, _ = torch.max(outer, dim=1)
            ymin = torch.argmax(a1, dim=1) - 1
            ymax = torch.argmax(a2, dim=1) - 1

            answer_dict_, _ = convert_tokens(dev_eval_file, ids.tolist(), ymin.tolist(),
                                             ymax.tolist())
            answer_dict.update(answer_dict_)
            print("\rSTEP {:6d}/{:<6d}  loss {:8f}".format(step, len(dev_dataloader), loss.item()),
                  end='')
    metrics = evaluate(dev_eval_file, answer_dict)
    metrics["loss"] = loss

    with open(config.answer_file, 'w') as f:
        json.dump(answer_dict, f)

    print(
        "\nEVAL loss {:8f}\tF1 {:8f}\tEM {:8f}".format(loss, metrics["f1"], metrics["exact_match"]))
    if config.mode == "train":
        writer.add_scalar('data/test_loss', np.mean(losses), global_step)
        writer.add_scalar('data/F1', metrics["f1"], global_step)
        writer.add_scalar('data/EM', metrics["exact_match"], global_step)
    return metrics


def test_entry(config):
    from models import QANet

    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    model = QANet(word_mat, char_mat).to(device)
    fn = config.model_file
    model.load_state_dict(torch.load(fn, map_location=device))
    test(config, model)


def train_entry(config, mode='total'):
    # model construct
    from models import QANet
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    model = QANet(word_mat, char_mat, train_target=mode).to(device)
    model.train()

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    if mode == 'qanet':
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.qanet_encoder.named_parameters()],
             'lr': 0.001, 'betas': (config.adam_beta1, config.adam_beta2), 'eps': config.adam_eps,
             'weight_decay': config.adam_decay},
            {'params': [p for n, p in model.qa_outputs.named_parameters()]},
        ]
    elif mode == 'bert':
        model.qanet_encoder.load_state_dict(
            torch.load(config.model_qanet_pretrain_file, map_location=device))
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.bert.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.bert.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.qa_outputs.named_parameters()]},
        ]
    else:
        model.qanet_encoder.load_state_dict(
            torch.load(config.model_qanet_pretrain_file, map_location=device))
        model.bert.load_state_dict(
            torch.load(config.model_bert_pretrain_file, map_location=device))
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if n not in 'bert' and n not in 'qanet_encoder'],
                'weight_decay': 0.0
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

    # data loader
    train_dataset = SQuADDataset(config.train_record_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=config.train_batch_size,
                                  num_workers=config.train_batch_size,
                                  pin_memory=True)

    # EMA
    ema = EMA(config.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    # training process
    print(f"#################################################\n"
          f"Start Training......\n"
          f"epoch: {config.epoch_num}\n"
          f"train batch size: {config.train_batch_size}\n"
          f"checkpoint: {config.checkpoint}\n"
          f"eval batch size: {config.eval_batch_size}\n"
          f"learning rate: {config.lr}\n"
          f"learning rate warning up num: {config.lr_warm_up_steps}\n"
          f"early stop: {config.early_stop}\n"
          f"#################################################")

    early_stop = False
    best_f1 = best_em = patience = global_step = 0
    for epoch in range(1, config.epoch_num + 1):
        if early_stop:
            break

        losses = []
        print(f"\nTraining Epoch {epoch}")
        for step, (input_word_ids, input_char_id, input_id, input_mask, input_token_type_id, y1, y2,
                   ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_word_ids, input_char_id, input_id, input_mask, input_token_type_id = \
                input_word_ids.to(device), input_char_id.to(device), input_id.to(
                    device), input_mask.to(device), input_token_type_id.to(device)
            p1, p2 = model(input_word_ids, input_char_id, input_id, input_mask, input_token_type_id)
            y1, y2 = y1.to(device), y2.to(device)
            loss = loss_func(p1, y1) + loss_func(p2, y2)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema(model, global_step)

            if global_step != 0 and global_step % config.checkpoint == 0:
                ema.assign(model)
                metrics = test(config, model, global_step, validate=True)
                ema.resume(model)
                model.train()

                f1 = metrics["f1"]
                em = metrics["exact_match"]
                if f1 - 0.05 < best_f1 and em - 0.05 < best_em:
                    patience += 1
                    if patience > config.early_stop:
                        early_stop = True
                        break
                else:
                    patience = 0
                    best_f1 = max(best_f1, f1)
                    best_em = max(best_em, em)
                if mode == 'qanet':
                    torch.save(model.qanet_encoder.state_dict(), config.model_qanet_pretrain_file)
                elif mode == 'bert':
                    torch.save(model.bert.state_dict(), config.model_bert_pretrain_file)
                else:
                    torch.save(model.state_dict(), config.model_file)
            for param_group in optimizer.param_groups:
                writer.add_scalar('data/lr', param_group['lr'], global_step)
            global_step += 1

            writer.add_scalar('data/loss', loss.item(), global_step)
            print("\rSTEP: {:6d}/{:<6d} loss: {:<8.3f}".format(
                step, len(train_dataloader), loss.item()), end='')
        loss_avg = sum(losses) / len(losses)
        print("\nAvg_loss {:8f}".format(loss_avg))

    # after training finished
    ema.assign(model)
    metrics = test(config, model, global_step, validate=True)
    best_f1 = max(best_f1, metrics["f1"])
    best_em = max(best_em, metrics["exact_match"])
    if mode == 'qanet':
        torch.save(model.qanet_encoder.state_dict(), config.model_qanet_pretrain_file)
    elif mode == 'bert':
        torch.save(model.bert.state_dict(), config.model_bert_pretrain_file)
    else:
        torch.save(model.state_dict(), config.model_file)
    ema.resume(model)

    print(f"Best Score: F1 {format(best_f1, '.6f')} | EM {format(best_em, '.6f')}")
    return model


def main(*args, **kwarg):
    if not os.path.exists(LOG_DIR) and config.mode == 'train':
        os.makedirs(LOG_DIR)
    if config.mode == "data":
        preproc(config)
    elif config.mode == "train":
        # train_entry(config, 'qanet')
        # train_entry(config, 'bert')
        train_entry(config, 'total')
    elif config.mode == "debug":
        config.epoch_num = 2
        config.train_record_file = config.dev_record_file
        train_entry(config)
    elif config.mode == "eval":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)
    print(config.run_name)


if __name__ == '__main__':
    app.run(main)
