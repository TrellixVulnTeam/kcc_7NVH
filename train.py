import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sentencepiece as spm

from custom_dataset import CustomDataset
from models.transformer.model import Transformer
from models.transformer.optim import ScheduledOptim

# from loss import ce_loss, kl_loss

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    # trg, gold = trg[:-1, :], trg[1:, :].contiguous().view(-1)
    trg, gold = trg[:, :], trg[:, :].contiguous().view(-1)
    return trg, gold


def decode_sentence(target_list, output_list, tokenizer, opt):
    tokenizer.Load(opt.tokenizer_path)
    target = [tokenizer.DecodeIds(i) for i in target_list]
    output = [tokenizer.DecodeIds(j) for j in output_list]
    return (target, output)

def save_decode(epoch, decode_list, name, path):
    target_file_name = f"{name}_{epoch}_target_decode.txt"
    output_file_name = f"{name}_{epoch}_output_decode.txt"
    target_save_path = os.path.join(path, target_file_name)
    output_save_path = os.path.join(path, output_file_name)


    with open(target_save_path, "w", encoding="utf8") as tf, open(output_save_path, "w", encoding="utf8") as of:
        for t_line, o_line in decode_list:
            for tline in t_line:
                tf.write(f"{tline}\n")
            for oline in o_line:
                of.write(f"{oline}\n")
        tf.close()
        of.close()

def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    total = len(training_data)
    with tqdm(total=total, desc=desc, leave=False) as pbar:
        for _, (src, trg) in enumerate(training_data):
            # print("src:", src.size())
            # print("trg:", trg.size())

            # prepare data
            src_seq = patch_src(src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, opt.trg_pad_idx))
            # print("src_seq:", src_seq.size())
            # print("trg_seq:", trg_seq.size())
            # print("gold:", gold.size())

            # forward
            optimizer.zero_grad()
            pred, out = model(src_seq, trg_seq)

            # print("pred:", pred.size())
            # print("gold:", gold.size())
            # print("pred: ", pred)
            # print("gold: ", gold)
            # print("gold.T:", gold.transpose(0, -1))

            # backward and update parameters
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=smoothing)
            loss.backward()
            optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
            pbar.update(1)

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy, trg, out


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    total = len(validation_data)
    with torch.no_grad():
        with tqdm(total=total, desc=desc, leave=False) as pbar:
            for _, (src, trg) in enumerate(validation_data):
                src_seq = patch_src(src, opt.src_pad_idx).to(device)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, opt.trg_pad_idx))

                # forward
                pred, out = model(src_seq, trg_seq)
                loss, n_correct, n_word = cal_performance(
                    pred, gold, opt.trg_pad_idx, smoothing=False)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()
                pbar.update(1)

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total

    return loss_per_word, accuracy, trg, out


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''
    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print(f'[Info] Training performance will be written to file: {log_train_file} and {log_valid_file}')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, loss, start_time, lr):
        print(f'  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu*100:3.3f} %, loss: {loss:.5f}, lr: {lr:8.5f}, elapse: {(time.time() - start_time)/60:3.3f} min')


    # valid_accus = []
    valid_losses = []

    tokenizer = spm.SentencePieceProcessor()

    train_decode = []
    valid_decode = []

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_trg, train_out = train_epoch(model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, train_loss, start, lr)

        start = time.time()
        valid_loss, valid_accu, valid_trg, valid_out = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, valid_loss, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100 * train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100 * valid_accu))


        # Decode
        train_decode.append(decode_sentence(train_trg.tolist(), train_out.tolist(), tokenizer, opt))
        valid_decode.append(decode_sentence(valid_trg.tolist(), valid_out.tolist(), tokenizer, opt))

        save_decode(epoch_i, train_decode, "train", opt.decode_path)
        save_decode(epoch_i, valid_decode, "valid", opt.decode_path)




def main():
    '''
    Usage:
    python transformer_train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_pkl',
                        default="/HDD/yehoon/data/processed/tokenized/spm_tokenized_data.pkl")  # all-in-1 data pickle or bpe field

    # parser.add_argument('--train_path', default=None)  # bpe encoded data
    # parser.add_argument('--val_path', default=None)  # bpe encoded data
    parser.add_argument('--tokenizer_path', default="/HDD/yehoon/data/tokenizer/train_em_formal_spm.model")
    parser.add_argument('--decode_path', default="/HDD/yehoon/data/transformer_output/decode")


    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)



    parser.add_argument('--min_len', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--src_vocab_size', type=int, default=1800)
    parser.add_argument('--trg_vocab_size', type=int, default=1800)
    parser.add_argument('--src_pad_idx', type=int, default=0)
    parser.add_argument('--trg_pad_idx', type=int, default=0)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')
    parser.add_argument('--scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('--output_dir', type=str, default="/HDD/yehoon/data/transformer_output/")
    parser.add_argument('--use_tb', action='store_true')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
    #     print('[Warning] The warmup steps may be not enough.\n' \
    #           '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
    #           'Using smaller batch w/o longer warmup may cause ' \
    #           'the warmup stage ends with only little data trained.')

    # Device Setting
    device = torch.device('cuda' if opt.cuda else 'cpu')

    print(f'[Device] {device}')

    # ========= Loading Dataset =========#
    # Data Setting
    with open(opt.data_pkl, "rb") as f:
        data = pickle.load(f)
        f.close()

    em_informal = data["gyafc"]["train"]["em_informal"]
    em_formal = data["gyafc"]["train"]["em_formal"]
    pair_kor = data['korpora']['train']['pair_kor']
    pair_eng = data['korpora']['train']['pair_eng']

    # fr_informal_train = data["gyafc"]["train"]["fr_informal"]
    # fr_formal_train = data["gyafc"]["train"]["fr_formal"]

    train_ratio = 0.7
    valid_ratio = 0.2

    em_informal_train = em_informal[:int(len(em_informal)*train_ratio)]
    em_informal_valid = em_informal[int(len(em_informal)*train_ratio):int(len(em_informal)*(train_ratio+valid_ratio))]
    em_informal_test = em_informal[int(len(em_informal)*(train_ratio+valid_ratio)):]

    em_formal_train = em_formal[:int(len(em_formal)*train_ratio)]
    em_formal_valid = em_formal[int(len(em_formal)*train_ratio):int(len(em_formal)*(train_ratio+valid_ratio))]
    em_formal_test = em_formal[int(len(em_formal)*(train_ratio+valid_ratio)):]

    # pair_kor_train = pair_kor[:int(len(pair_kor) * train_ratio)]
    # pair_kor_valid = pair_kor[int(len(pair_kor) * train_ratio):int(len(pair_kor) * train_ratio + valid_ratio)]
    # pair_kor_test = pair_kor[int(len(pair_kor) * train_ratio + valid_ratio):]
    #
    # pair_eng_train = pair_eng[:int(len(pair_eng) * train_ratio)]
    # pair_eng_valid = pair_eng[int(len(pair_eng) * train_ratio):int(len(pair_eng) * train_ratio + valid_ratio)]
    # pair_eng_test = pair_eng[int(len(pair_eng) * train_ratio + valid_ratio):]


    train_data = CustomDataset(em_informal_train, em_formal_train, opt.min_len, opt.max_len)
    valid_data = CustomDataset(em_informal_valid, em_formal_valid, opt.min_len, opt.max_len)
    # tst_test_data = CustomDataset(em_informal_test, em_formal_test, min_len, max_len)

    # nmt_train_data = CustomDataset(pair_eng_train, pair_kor_train, opt.min_len, opt.max_len)
    # nmt_valid_data = CustomDataset(pair_eng_valid, pair_kor_valid, opt.min_len, opt.max_len)
    # nmt_test_data = CustomDataset(pair_eng_test, pair_kor_test, min_len, max_len)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)
    # test_loader = DataLoader(tst_test_data, batch_size=batch_size, drop_last=True, shuffle=True,
    #                              num_workers=num_workers)



    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        batch_size = opt.batch_size,
        max_len = opt.max_len,
        trg_vocab_size = opt.trg_vocab_size,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)
    transformer = nn.DataParallel(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, train_loader, valid_loader, optimizer, device, opt)


if __name__ == '__main__':
    main()
