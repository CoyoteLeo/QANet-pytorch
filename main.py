import math
import os

import numpy as np
import torch
import torch.cuda
import ujson as json
from absl import app
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.nn import functional as F
from config import config, LOG_DIR, device
from preproc_bert import preproc
from utils import EMA, SQuADDataset, convert_tokens, evaluate

writer = SummaryWriter(log_dir=LOG_DIR)
loss_func = F.cross_entropy


def test(config, model, global_step=0, validate=False):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = SQuADDataset(config.dev_record_file)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                batch_size=config.eval_batch_size)

    print("\nValidate" if validate else "Test")
    model.eval()
    answer_dict = {}
    losses = []
    with torch.no_grad():
        for step, (Cwid, Qwid, y1, y2, ids) in enumerate(dev_dataloader):
            Cwid, Qwid = Cwid.to(device), Qwid.to(device)
            p1, p2 = model(Cwid, Qwid)
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
            ymin = torch.argmax(a1, dim=1)
            ymax = torch.argmax(a2, dim=1)

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

    model = QANet().to(device)
    fn = config.model_file
    model.load_state_dict(torch.load(fn, map_location=device))
    test(config, model)


def train_entry(config):
    # model construct
    from models import QANet
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    model = QANet().to(device)
    model.train()

    # data loader
    train_dataset = SQuADDataset(config.train_record_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=config.train_batch_size,
                                  num_workers=config.train_batch_size, pin_memory=True)

    # EMA
    ema = EMA(config.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    # optimizer
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = Adam(lr=1, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
                     weight_decay=config.adam_decay, params=parameters)
    cr = config.lr / math.log2(config.lr_warm_up_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < config.lr_warm_up_steps else config.lr
    )

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
        for step, (Cwid, Qwid, y1, y2, ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            Cwid, Qwid = Cwid.to(device), Qwid.to(device)
            p1, p2 = model(Cwid, Qwid)
            y1, y2 = y1.to(device), y2.to(device)
            loss = loss_func(p1, y1) + loss_func(p2, y2)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema(model, global_step)
            scheduler.step(global_step)

            if global_step != 0 and global_step % config.checkpoint == 0:
                ema.assign(model)
                metrics = test(config, model, global_step, validate=True)
                ema.resume(model)
                model.train()

                f1 = metrics["f1"]
                em = metrics["exact_match"]
                if f1 < best_f1 and em < best_em:
                    patience += 1
                    if patience > config.early_stop:
                        early_stop = True
                        break
                else:
                    patience = 0
                    best_f1 = max(best_f1, f1)
                    best_em = max(best_em, em)
                torch.save(model.state_dict(), config.model_file)
            for param_group in optimizer.param_groups:
                writer.add_scalar('data/lr', param_group['lr'], global_step)
            global_step += 1

            writer.add_scalar('data/loss', loss.item(), global_step)
            print("\rSTEP: {:6d}/{:<6d} loss: {:<8.3f} lr: {:.6f}".format(
                step, len(train_dataloader), loss.item(), scheduler.get_lr()[0]
            ), end='')
        loss_avg = sum(losses) / len(losses)
        print("\nAvg_loss {:8f}".format(loss_avg))

    # after training finished
    ema.assign(model)
    metrics = test(config, model, global_step, validate=True)
    best_f1 = max(best_f1, metrics["f1"])
    best_em = max(best_em, metrics["exact_match"])
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
        train_entry(config)
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
