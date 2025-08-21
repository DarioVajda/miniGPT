import sentencepiece as spm
import tqdm

txt_dataset_path = "/ceph/hpc/data/s24o01-42-users/corpuses/cc_news/ccnews_5b_tokens.txt"
tokenized_dataset_path = "tokenized_ccnews_5b.txt"

sp = spm.SentencePieceProcessor(model_file="../tokenizer/mini_vlm_sp16k.model")

def tokenizeLine(line):
    return sp.encode(line, out_type=int)

# Optimize the above code to count the number of lines
total_lines = sum(1 for _ in open(txt_dataset_path, "r"))
print(f"Total lines in dataset: {total_lines}")

# test_size = 10
test_size = -1 # tokenize all lines
with open(tokenized_dataset_path, "w") as output_file:
    with open(txt_dataset_path, "r") as file:
        for line in tqdm.tqdm(file, desc="Tokenizing lines"):
            if test_size == 0:
                break
            test_size -= 1

            tokens = tokenizeLine(line.strip())
            output_file.write(" ".join(map(str, tokens)) + "\n")