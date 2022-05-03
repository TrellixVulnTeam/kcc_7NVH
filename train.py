<<<<<<< HEAD
import tqdm
import pickle
import os

import torch
from torch.utils.data import DataLoader

import sentencepiece as spm

from custom_dataset import CustomDataset
from model import Encoder, TSTDecoder, NMTDecoder, StyleTransfer, StylizedNMT
from loss import bce_loss, ce_loss, kl_loss

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.autograd.set_detect_anomaly(True)

def tst_train_one_epoch(tst_model, epoch, data_loader, tst_optimizer, device):
    tst_model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            # size ; [16, 300]=[batch, max_len]

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv, output_list = tst_model(src, trg)

            # TODO Loss 재정의
            # BCE_loss = bce_loss(tst_out, trg)
            # KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            # loss = (BCE_loss + KL_loss*KL_weight)

            tst_out = tst_out.transpose(0, 1)
            # size ; [max_len, batch, vocab_size] -> [batch, max_len, vocab_size]
            tst_out = tst_out[:, 1:].reshape(-1, tst_out.size(-1))
            # size ; [(max_len-1)*batch,vocab]

            trg_trg = trg[:, 1:].reshape(-1)
            # size ; [(max_len-1)*batch]

            CE_loss = ce_loss(tst_out, trg_trg)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            tst_loss = (CE_loss + KL_loss * KL_weight)
            loss_value = tst_loss.item()
            train_loss += loss_value

            tst_optimizer.zero_grad()   # optimizer 초기화
            tst_loss.backward(retain_graph=True)
            tst_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total, total_latent, trg[:, 1:].tolist(), output_list

def nmt_train_one_epoch(nmt_model, data_loader, nmt_optimizer, device):
    nmt_model.train()

    # Freeze
    for name, param in nmt_model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out, output_list = nmt_model(src, trg)

            nmt_out = nmt_out.transpose(0, 1)
            nmt_out = nmt_out[:, 1:].reshape(-1, nmt_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)

            CE_loss = ce_loss(nmt_out, trg_trg)
            nmt_loss = CE_loss
            loss_value = nmt_loss.item()
            train_loss += loss_value

            nmt_optimizer.zero_grad()   # optimizer 초기화
            nmt_loss.backward()
            nmt_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total, trg[:, 1:].tolist(), output_list

@torch.no_grad()    #no autograd (backpropagation X)
def tst_evaluate(tst_model, epoch, data_loader, device):
    tst_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv, output_list = tst_model(src, trg)

            # BCE_loss = ce_loss(tst_out, trg)
            # KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            # loss = (BCE_loss + KL_loss * KL_weight)

            tst_out = tst_out.transpose(0, 1)
            tst_out = tst_out[:, 1:].reshape(-1, tst_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)

            CE_loss = ce_loss(tst_out, trg_trg)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            tst_loss = (CE_loss + KL_loss * KL_weight)
            loss_value = tst_loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss/total, trg[:, 1:].tolist(), output_list

@torch.no_grad()    #no autograd (backpropagation X)
def nmt_evaluate(nmt_model, data_loader, device):
    target_list = []
    output_list = []
    nmt_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out, output_list = nmt_model(src, trg)

            nmt_out = nmt_out.transpose(0, 1)
            nmt_out = nmt_out[:, 1:].reshape(-1, nmt_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)


            CE_loss = ce_loss(nmt_out, trg_trg)
            nmt_loss = CE_loss
            loss_value = nmt_loss.item()
            valid_loss += loss_value


            pbar.update(1)

    return valid_loss/total, trg[:, 1:].tolist(), output_list

def train():
    # Device Setting
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')

    # Data Setting
    with open("/HDD/yehoon/data/processed/tokenized/spm_tokenized_data.pkl", "rb") as f:
        data = pickle.load(f)
        f.close()

    em_informal_train = data["gyafc"]["train"]["em_informal"]
    em_formal_train = data["gyafc"]["train"]["em_formal"]
    pair_kor_train = data['korpora']['train']['pair_kor']
    pair_eng_train = data['korpora']['train']['pair_eng']
    # fr_informal_train = data["gyafc"]["train"]["fr_informal"]
    # fr_formal_train = data["gyafc"]["train"]["fr_formal"]

    split_ratio = 0.8
    em_informal_train = em_informal_train[:int(len(em_informal_train) * split_ratio)]
    em_informal_valid = em_informal_train[int(len(em_informal_train) * split_ratio):]
    em_formal_train = em_formal_train[:int(len(em_formal_train) * split_ratio)]
    em_formal_valid = em_formal_train[int(len(em_formal_train) * split_ratio):]
    pair_kor_train = pair_kor_train[:int(len(pair_kor_train) * split_ratio)]
    pair_kor_valid = pair_kor_train[int(len(pair_kor_train) * split_ratio):]
    pair_eng_train = pair_eng_train[:int(len(pair_eng_train) * split_ratio)]
    pair_eng_valid = pair_eng_train[int(len(pair_eng_train) * split_ratio):]


    em_informal_test = data["gyafc"]["test"]["em_informal"]
    em_formal_test = data["gyafc"]["test"]["em_formal"]
    pair_kor_test = data['korpora']['test']['pair_kor']
    pair_eng_test = data['korpora']['test']['pair_eng']
    # fr_informal_test = data["gyafc"]["test"]["fr_informal"]
    # fr_formal_test = data["gyafc"]["test"]["fr_formal"]


    # TODO argparse
    min_len, max_len = 2, 300
    batch_size = 128
    num_workers = 0
    tst_vocab_size = 1800
    nmt_vocab_size = 2400

    tst_train_data = CustomDataset(em_informal_train, em_formal_train, min_len, max_len)
    tst_valid_data = CustomDataset(em_informal_valid, em_formal_valid, min_len, max_len)
    tst_test_data = CustomDataset(em_informal_test, em_formal_test, min_len, max_len)

    nmt_train_data = CustomDataset(pair_eng_train, pair_kor_train, min_len, max_len)
    nmt_valid_data = CustomDataset(pair_eng_valid, pair_kor_valid, min_len, max_len)
    nmt_test_data = CustomDataset(pair_eng_test, pair_kor_test, min_len, max_len)


    tst_train_loader = DataLoader(tst_train_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    tst_valid_loader = DataLoader(tst_valid_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    tst_test_loader = DataLoader(tst_test_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                 num_workers=num_workers)

    nmt_train_loader = DataLoader(nmt_train_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    nmt_valid_loader = DataLoader(nmt_valid_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    nmt_test_loader = DataLoader(nmt_test_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                 num_workers=num_workers)


    # TST Train
    tst_encoder = Encoder(input_size=tst_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    tst_decoder = TSTDecoder(output_size=tst_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    tst_model = StyleTransfer(tst_encoder, tst_decoder, d_hidden=1024, style_ratio=0.3, device=device)
    tst_model = tst_model.to(device)

    tst_optimizer = torch.optim.AdamW(tst_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 0

    print("Start TST Training..")

    tst_tokenizer = spm.SentencePieceProcessor()
    tst_tokenizer.Load("/HDD/yehoon/data/tokenizer/train_em_formal_spm.model")
    decode_output = []
    for epoch in range(start_epoch, epochs+1):
        print(f"Epoch: {epoch}")
        epoch_loss, total_latent, trg_list, out_list = tst_train_one_epoch(tst_model, epoch, tst_train_loader, tst_optimizer, device)

        print(f"Training Loss: {epoch_loss:.5f}")
        # print(f"train target: {[tst_tokenizer.DecodeIds(i) for i in trg_list]}")
        # print(f"train predict: {[tst_tokenizer.DecodeIds(j) for j in out_list]}")

        valid_loss, trg_trg, val_out = tst_evaluate(tst_model, epoch, tst_valid_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")

        # print(f"valid target: {[tst_tokenizer.DecodeIds(i) for i in trg_list]}")
        # print(f"valid predict: {[tst_tokenizer.DecodeIds(j) for j in out_list]}")
        target_decode = [tst_tokenizer.DecodeIds(i) for i in trg_list]
        output_decoder = [tst_tokenizer.DecodeIds(j) for j in out_list]
        decode_output.append((target_decode, output_decoder))

    with open ("/HDD/yehoon/data/target_decode.txt", "w", encoding="utf8") as tf, open("/HDD/yehoon/data/output_decode.txt", "w", encoding="utf8") as of:
        for t_line, o_line in decode_output:
            for tline in t_line:
                tf.write(f"{tline}\n")
            for oline in o_line:
                of.write(f"{oline}\n")
        tf.close()
        of.close()


    # NMT Train
    nmt_encoder = Encoder(input_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    nmt_decoder = NMTDecoder(output_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    nmt_model = StylizedNMT(nmt_encoder, nmt_decoder, d_hidden=1024, total_latent=total_latent, device=device)
    nmt_model = nmt_model.to(device)

    nmt_optimizer = torch.optim.AdamW(nmt_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 100

    print("Start NMT Training..")

    nmt_tokenizer = spm.SentencePieceProcessor()
    nmt_tokenizer.Load("/HDD/yehoon/data/tokenizer/train_pair_kor_spm.model")
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch: {epoch}")
        epoch_loss , trg_trg, train_out= nmt_train_one_epoch(nmt_model, nmt_train_loader, nmt_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")
        # print(f"train target: {nmt_tokenizer.DecodeIds(trg_trg)}, train predict: {nmt_tokenizer.DecodeIds(train_out)}")

        valid_loss, trg_trg, val_out = nmt_evaluate(nmt_model, nmt_valid_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")
        print(f"valid target: {nmt_tokenizer.DecodeIds(trg_trg)}, valid predict: {nmt_tokenizer.DecodeIds(val_out)}")

=======
import tqdm
import pickle
import os

import torch
from torch.utils.data import DataLoader

import sentencepiece as spm

from custom_dataset import CustomDataset
from model import Encoder, TSTDecoder, NMTDecoder, StyleTransfer, StylizedNMT
from loss import bce_loss, ce_loss, kl_loss

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def tst_train_one_epoch(tst_model, epoch, data_loader, tst_optimizer, device):
    tst_model.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            # size ; [16, 300]=[batch, max_len]

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv, output_list = tst_model(src, trg)

            # TODO Loss 재정의
            # BCE_loss = bce_loss(tst_out, trg)
            # KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            # loss = (BCE_loss + KL_loss*KL_weight)

            tst_out = tst_out.transpose(0, 1)
            # size ; [max_len, batch, vocab_size] -> [batch, max_len, vocab_size]
            tst_out = tst_out[:, 1:].reshape(-1, tst_out.size(-1))
            # size ; [(max_len-1)*batch,vocab]

            trg_trg = trg[:, 1:].reshape(-1)
            # size ; [(max_len-1)*batch]

            CE_loss = ce_loss(tst_out, trg_trg)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            loss = (CE_loss + KL_loss * KL_weight)
            loss_value = loss.item()
            train_loss += loss_value

            tst_optimizer.zero_grad()   # optimizer 초기화
            loss.backward()
            tst_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total, total_latent, trg[:, 1:].tolist(), output_list

def nmt_train_one_epoch(nmt_model, data_loader, nmt_optimizer, device):
    nmt_model.train()

    # Freeze
    for name, param in nmt_model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out = nmt_model(src, trg)

            nmt_out = nmt_out.transpose(0, 1)
            nmt_out = nmt_out[:, 1:].reshape(-1, nmt_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)

            CE_loss = ce_loss(nmt_out, trg_trg)
            loss = CE_loss
            loss_value = loss.item()
            train_loss += loss_value

            nmt_optimizer.zero_grad()   # optimizer 초기화
            loss.backward(retain_graph=True)
            nmt_optimizer.step()    # Gradient Descent 시작
            pbar.update(1)

    return train_loss/total, trg_trg, nmt_out

@torch.no_grad()    #no autograd (backpropagation X)
def tst_evaluate(tst_model, epoch, data_loader, device):
    tst_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            tst_out, total_latent, content_c, content_mu, content_logv, style_a, style_mu, style_logv, output_list = tst_model(src, trg)

            # BCE_loss = ce_loss(tst_out, trg)
            # KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            # loss = (BCE_loss + KL_loss * KL_weight)

            tst_out = tst_out.transpose(0, 1)
            tst_out = tst_out[:, 1:].reshape(-1, tst_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)

            CE_loss = ce_loss(tst_out, trg_trg)
            KL_loss, KL_weight = kl_loss(style_mu, style_logv, epoch, 0.0025, 2500)
            loss = (CE_loss + KL_loss * KL_weight)
            loss_value = loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss/total, trg[:, 1:].tolist(), output_list

@torch.no_grad()    #no autograd (backpropagation X)
def nmt_evaluate(nmt_model, data_loader, device):
    target_list = []
    output_list = []
    nmt_model.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            nmt_out = nmt_model(src, trg)

            nmt_out = nmt_out.transpose(0, 1)
            nmt_out = nmt_out[:, 1:].reshape(-1, nmt_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)


            CE_loss = ce_loss(nmt_out, trg_trg)
            loss = CE_loss
            loss_value = loss.item()
            valid_loss += loss_value


            pbar.update(1)

    return valid_loss/total, trg_trg, nmt_out

def train():
    # Device Setting
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')

    # Data Setting
    with open("/HDD/yehoon/data/processed/tokenized/spm_tokenized_data.pkl", "rb") as f:
        data = pickle.load(f)
        f.close()

    em_informal_train = data["gyafc"]["train"]["em_informal"]
    em_formal_train = data["gyafc"]["train"]["em_formal"]
    pair_kor_train = data['korpora']['train']['pair_kor']
    pair_eng_train = data['korpora']['train']['pair_eng']
    # fr_informal_train = data["gyafc"]["train"]["fr_informal"]
    # fr_formal_train = data["gyafc"]["train"]["fr_formal"]

    split_ratio = 0.8
    em_informal_train = em_informal_train[:int(len(em_informal_train) * split_ratio)]
    em_informal_valid = em_informal_train[int(len(em_informal_train) * split_ratio):]
    em_formal_train = em_formal_train[:int(len(em_formal_train) * split_ratio)]
    em_formal_valid = em_formal_train[int(len(em_formal_train) * split_ratio):]
    pair_kor_train = pair_kor_train[:int(len(pair_kor_train) * split_ratio)]
    pair_kor_valid = pair_kor_train[int(len(pair_kor_train) * split_ratio):]
    pair_eng_train = pair_eng_train[:int(len(pair_eng_train) * split_ratio)]
    pair_eng_valid = pair_eng_train[int(len(pair_eng_train) * split_ratio):]


    em_informal_test = data["gyafc"]["test"]["em_informal"]
    em_formal_test = data["gyafc"]["test"]["em_formal"]
    pair_kor_test = data['korpora']['test']['pair_kor']
    pair_eng_test = data['korpora']['test']['pair_eng']
    # fr_informal_test = data["gyafc"]["test"]["fr_informal"]
    # fr_formal_test = data["gyafc"]["test"]["fr_formal"]


    # TODO argparse
    min_len, max_len = 2, 300
    batch_size = 250
    num_workers = 0
    tst_vocab_size = 1800
    nmt_vocab_size = 2400

    tst_train_data = CustomDataset(em_informal_train, em_formal_train, min_len, max_len)
    tst_valid_data = CustomDataset(em_informal_valid, em_formal_valid, min_len, max_len)
    tst_test_data = CustomDataset(em_informal_test, em_formal_test, min_len, max_len)

    nmt_train_data = CustomDataset(pair_eng_train, pair_kor_train, min_len, max_len)
    nmt_valid_data = CustomDataset(pair_eng_valid, pair_kor_valid, min_len, max_len)
    nmt_test_data = CustomDataset(pair_eng_test, pair_kor_test, min_len, max_len)


    tst_train_loader = DataLoader(tst_train_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    tst_valid_loader = DataLoader(tst_valid_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    tst_test_loader = DataLoader(tst_test_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                 num_workers=num_workers)

    nmt_train_loader = DataLoader(nmt_train_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    nmt_valid_loader = DataLoader(nmt_valid_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                  num_workers=num_workers)
    nmt_test_loader = DataLoader(nmt_test_data, batch_size=batch_size, drop_last=True, shuffle=True,
                                 num_workers=num_workers)


    # TST Train
    tst_encoder = Encoder(input_size=tst_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    tst_decoder = TSTDecoder(output_size=tst_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    tst_model = StyleTransfer(tst_encoder, tst_decoder, d_hidden=1024, style_ratio=0.3, device=device)
    tst_model = tst_model.to(device)

    tst_optimizer = torch.optim.AdamW(tst_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 0

    print("Start TST Training..")

    tst_tokenizer = spm.SentencePieceProcessor()
    tst_tokenizer.Load("/HDD/yehoon/data/tokenizer/train_em_formal_spm.model")
    decode_output = []
    for epoch in range(start_epoch, epochs+1):
        print(f"Epoch: {epoch}")
        epoch_loss, total_latent, trg_list, out_list = tst_train_one_epoch(tst_model, epoch, tst_train_loader, tst_optimizer, device)

        print(f"Training Loss: {epoch_loss:.5f}")
        # print(f"train target: {[tst_tokenizer.DecodeIds(i) for i in trg_list]}")
        # print(f"train predict: {[tst_tokenizer.DecodeIds(j) for j in out_list]}")

        valid_loss, trg_trg, val_out = tst_evaluate(tst_model, epoch, tst_valid_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")

        # print(f"valid target: {[tst_tokenizer.DecodeIds(i) for i in trg_list]}")
        # print(f"valid predict: {[tst_tokenizer.DecodeIds(j) for j in out_list]}")
        target_decode = [tst_tokenizer.DecodeIds(i) for i in trg_list]
        output_decoder = [tst_tokenizer.DecodeIds(j) for j in out_list]
        decode_output.append((target_decode, output_decoder))

    with open ("/HDD/yehoon/data/target_decode.txt", "w", encoding="utf8") as tf, open("/HDD/yehoon/data/output_decode.txt", "w", encoding="utf8") as of:
        for t_line, o_line in decode_output:
            for tline in t_line:
                tf.write(f"{tline}\n")
            for oline in o_line:
                of.write(f"{oline}\n")
        tf.close()
        of.close()


    # NMT Train
    nmt_encoder = Encoder(input_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    nmt_decoder = NMTDecoder(output_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    nmt_model = StylizedNMT(nmt_encoder, nmt_decoder, d_hidden=1024, total_latent=total_latent, device=device)
    nmt_model = nmt_model.to(device)

    nmt_optimizer = torch.optim.AdamW(nmt_model.parameters(), lr=0.001)

    start_epoch = 0
    epochs = 100

    print("Start NMT Training..")

    nmt_tokenizer = spm.SentencePieceProcessor()
    nmt_tokenizer.Load("/HDD/yehoon/data/tokenizer/train_pair_kor_spm.model")
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch: {epoch}")
        epoch_loss , trg_trg, train_out= nmt_train_one_epoch(nmt_model, nmt_train_loader, nmt_optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")
        # print(f"train target: {nmt_tokenizer.DecodeIds(trg_trg)}, train predict: {nmt_tokenizer.DecodeIds(train_out)}")

        valid_loss, trg_trg, val_out = nmt_evaluate(nmt_model, nmt_valid_loader, device)
        print(f"Validation Loss: {valid_loss:.5f}")
        print(f"valid target: {nmt_tokenizer.DecodeIds(trg_trg)}, valid predict: {nmt_tokenizer.DecodeIds(val_out)}")

>>>>>>> ac1c64bc6348782202b342f8434e0482a58160aa
