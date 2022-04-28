import glob
import pickle

import sentencepiece as spm


tokenized_data = {}
tokenized_data['train'] = {}
tokenized_data['test'] = {}

for data_type in ["train", "test"]:
    files = glob.glob(f"/HDD/yehoon/data/processed/raw/*{data_type}.txt")

    parameter = '--input={} \
    --pad_id={} --pad_piece={} \
    --bos_id={} --bos_piece={} \
    --eos_id={} --eos_piece={} \
    --unk_id={} --unk_piece={} \
    --user_defined_symbols={} \
    --model_prefix={} \
    --vocab_size={} \
    --max_sentence_length={} \
    --character_coverage={} \
    --model_type={}'

    pad_id = 0
    pad_piece = "[PAD]"
    bos_id = 1
    bos_piece = "[BOS]"
    eos_id = 2
    eos_piece = "[EOS]"
    unk_id = 3
    unk_piece = "[UNK]"
    user_defined_symbols = "[SEP],[CLS],[MASK]"
    vocab_size = 1800
    max_sentence_length = 9999
    character_coverage = 1.0  # default
    model_type = 'unigram'  # default: unigram

    for train_input_file in files:
        prefix = (("_").join(train_input_file.split("\\")[-1].split("_")[:-1]))
        model_prefix = f'/HDD/yehoon/data/tokenizer/{data_type}_{prefix}_spm'

        cmd = parameter.format(train_input_file,
                               pad_id, pad_piece,
                               bos_id, bos_piece,
                               eos_id, eos_piece,
                               unk_id, unk_piece,
                               user_defined_symbols,
                               model_prefix,
                               vocab_size,
                               max_sentence_length,
                               character_coverage,
                               model_type)
        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(cmd)
        print(f"Train Compelte: {data_type} {prefix} model & vocab")

        sp = spm.SentencePieceProcessor()
        sp.Load(f"{model_prefix}.model")

        # BOS, EOS 추가
        sp.SetEncodeExtraOptions('bos:eos')

        # Tokenization And Padding
        with open(train_input_file, "r", encoding="utf-8") as f:
            tokenized_data[data_type][prefix] = [sp.EncodeAsIds(line) for line in f]
            print(f"Make Compelte: {data_type} {prefix} tokenized data")

# Save Data
processed_path = "/HDD/yehoon/data/processed/tokenized/spm_tokenized_data.pkl"
with open(processed_path, 'wb') as file:
    pickle.dump(tokenized_data, file)
print("Saving Tokenized Data is Done!")