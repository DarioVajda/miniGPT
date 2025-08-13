import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="/workspace/corpuses/cc_news/ccnews_100m_tokens.txt",
    model_prefix="mini_vlm_sp16k",
    model_type="unigram",
    vocab_size=16000,
    byte_fallback=True,
    character_coverage=0.9995,
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=",".join([
        "<start_user>","<end_user>","<start_agent>","<end_agent>",
        "<start_img>","<end_img>",
        *[f"<special_token_{i}>" for i in range(8)]
    ]),

    # >>> Speed + stability for big corpora <<<
    input_sentence_size=5_000_000,      # sample 5M lines instead of reading everything
    shuffle_input_sentence=True,        # random sampling
    train_extremely_large_corpus=True,  # stream-friendly, less RAM
    hard_vocab_limit=False,             # don’t stall trying to hit 16k exactly
    max_sentence_length=10000,          # avoid skipping long lines
    seed_sentencepiece_size=200_000,    # smaller seed = faster EM (1e6 is overkill)
    num_threads=8
)
print("Done: mini_vlm_sp16k.model / .vocab")