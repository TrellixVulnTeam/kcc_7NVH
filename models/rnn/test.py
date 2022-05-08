import tqdm
import pickle

import torch
from torch.utils.data import DataLoader

import sentencepiece as spm

from custom_dataset import CustomDataset
from models.rnn.model import Encoder, NMTDecoder, StylizedNMT
from loss import ce_loss

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

@torch.no_grad()    #no autograd (backpropagation X)
def nmt_evaluate(tst_encoder, nmt_model, data_loader, device):
    tst_encoder.eval()
    nmt_model.eval()

    test_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            _, nmt_hidden, nmt_cell = tst_encoder(src)

            nmt_hidden = nmt_hidden.detach().to(device)
            nmt_cell = nmt_cell.detach().to(device)
            # nmt_hidden = nmt_hidden.to(device)
            # nmt_cell = nmt_cell.to(device)

            nmt_out, output_list = nmt_model(nmt_hidden, nmt_cell, trg)

            nmt_out = nmt_out[:, 1:].reshape(-1, nmt_out.size(-1))

            trg_trg = trg[:, 1:].reshape(-1)

            CE_loss = ce_loss(nmt_out, trg_trg)
            nmt_loss = CE_loss
            loss_value = nmt_loss.item()
            test_loss += loss_value

            pbar.update(1)

    return test_loss/total, trg[:, 1:].tolist(), output_list

def test(args):
    # Device Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing Device: {device}')
    print(f'Count of using GPUs:{torch.cuda.device_count()}')

    # Data Setting
    with open("/HDD/yehoon/data/processed/tokenized/spm_tokenized_data.pkl", "rb") as f:
        data = pickle.load(f)
        f.close()

    em_informal_test = data["gyafc"]["test"]["em_informal"]
    pair_kor_test = data['korpora']['test']['pair_kor']
    # fr_informal_test = data["gyafc"]["test"]["fr_informal"]
    # fr_formal_test = data["gyafc"]["test"]["fr_formal"]


    # TODO argparse
    min_len, max_len = 2, 300
    nmt_vocab_size = 2400
    batch_size = 100
    num_workers = 0


    # tst_test_data = CustomDataset(em_informal_test, em_formal_test, min_len, max_len)
    # nmt_test_data = CustomDataset(pair_eng_test, pair_kor_test, min_len, max_len)
    test_data = CustomDataset(em_informal_test, pair_kor_test, min_len, max_len)

    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)

    # Model Load
    # model.load_state_dict(torch.load(save_file_name)['model'])


    encoder = Encoder(input_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    encoder.load_state_dict(torch.load("/HDD/yehoon/data/tst_model.pth")['model'])
    print("encoder", encoder)
    encoder = encoder.encoder.eval()

    decoder = NMTDecoder(output_size=nmt_vocab_size, d_hidden=1024, d_embed=256, n_layers=2, dropout=0.1, device=device)
    decoder.load_state_dict((torch.load("/HDD/yehoon/data/nmt_model.pth")['model']))
    decoder = decoder.eval()

    total_latent = torch.load(torch.load("/HDD/yehoon/data/nmt_model.pth")['total_latent'])

    model = StylizedNMT(decoder, d_hidden=1024, total_latent=total_latent, device=device)
    model.load_state_dict(torch.load("/HDD/yehoon/data/nmt_model.pth"))
    model = model.to(device)
    model.eval()
    print("Start Testing..")

    test_tokenizer = spm.SentencePieceProcessor()
    test_tokenizer.Load("/HDD/yehoon/data/tokenizer/test_pair_kor_spm.model")

    test_decode_output = []

    test_loss, test_trg_list, test_out_list = nmt_evaluate(encoder, model, test_loader, device)
    print(f"Validation Loss: {test_loss:.5f}")

    test_out_list = list(map(list, zip(*test_out_list)))

    test_target_decode = [test_tokenizer.DecodeIds(i) for i in test_trg_list]
    test_output_decoder = [test_tokenizer.DecodeIds(j) for j in test_out_list]
    test_decode_output.append((test_target_decode, test_output_decoder))

    with open("/HDD/yehoon/data/test_target_decode.txt", "w", encoding="utf8") as tf, open("/HDD/yehoon/data/test_output_decode.txt", "w", encoding="utf8") as of:
        for t_line, o_line in test_decode_output:
            for tline in t_line:
                tf.write(f"{tline}\n")
            for oline in o_line:
                of.write(f"{oline}\n")
        tf.close()
        of.close()