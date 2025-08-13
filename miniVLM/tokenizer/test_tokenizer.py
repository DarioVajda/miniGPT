import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="mini_vlm_sp16k.model")

ids = sp.encode("A photo of a red car.<start_img><end_img>", out_type=int)   # e.g., [123, 456, ...]
print(ids)  # print token IDs
text = sp.decode(ids)                                    # back to string (sanity check)
print(text)  # should match the original input

# grab special IDs once and reuse
BOS = sp.bos_id()
EOS = sp.eos_id()
PAD = sp.pad_id()
IMG_START = sp.piece_to_id("<start_img>")
IMG_END   = sp.piece_to_id("<end_img>")
print(f"BOS: {BOS}, EOS: {EOS}, PAD: {PAD}, IMG_START: {IMG_START}, IMG_END: {IMG_END}")
