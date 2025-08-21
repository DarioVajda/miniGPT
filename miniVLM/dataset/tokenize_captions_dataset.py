import sentencepiece as spm
import tqdm

tsv_train_dataset_path = "../../captions_dataset/GCC-training.tsv"
tsv_val_dataset_path = "../../captions_dataset/GCC-1.1.0-Validation.tsv"

tokenized_train_dataset_path = "tokenized_train_captions_dataset.txt"
tokenized_val_dataset_path = "tokenized_val_captions_dataset.txt"

sp = spm.SentencePieceProcessor(model_file="../tokenizer/mini_vlm_sp16k.model")

def tokenizeLine(line):
    return sp.encode(line, out_type=int)

with open(tokenized_train_dataset_path, "w") as output_file:
    with open(tsv_train_dataset_path, "r") as file:
        for line in tqdm.tqdm(file, desc="Tokenizing lines"):
            # take the first column of the TSV line
            caption = line.split("\t")[0]

            tokens = tokenizeLine(caption.strip())
            output_file.write(" ".join(map(str, tokens)) + "\n")

with open(tokenized_val_dataset_path, "w") as output_file:
    with open(tsv_val_dataset_path, "r") as file:
        for line in tqdm.tqdm(file, desc="Tokenizing lines"):
            # take the first column of the TSV line
            caption = line.split("\t")[0]

            tokens = tokenizeLine(caption.strip())
            output_file.write(" ".join(map(str, tokens)) + "\n")