# MiniGPT

This is a personal project divided into three parts:
1. Implementation of a Generative Pretrained Transformer (GPT) model trained on Tiny Shakespeare
2. Implementation of a Vision Transformer (ViT) model trained for image classification
3. Implementation of a Vision Language Model (VLM) trained on a large corpora of text and images.

*Note* - each model was implemented in plain PyTorch to demonstrate the fundemental understanding of both the library and the underlying architectures of the transformer and models that rely on it.

## Generative Pretrained Transformer
A basic decoder-only transformer model was trained on the Tiny Shakespeare datasest. The resulting model was very small, but managed to generate outputs that resemble the training dataset.
More detailed explanation and source code are available in the miniGPT directoty.

## Vision Transformer
A 20M parameter Vision Transformer model was trained on the Tiny Imagenet dataset for image classification. 
The final model performed very well considering its size and the fact that the model was trained from scratch, without using any pretrained models.
My model achieved acuracy of 45% on the validation data.

## Vision Language Model (work in progress)
This is a larger project demonstrating a basic implementation of the entire pretraining pipeline for a Vision Language Model (VLM):
* Training a text tokenizer with a 16k token vocabulary
* Preparing a text dataset and tokenizing it - Over 5B tokens from CC News
* Preparing a image + text dataset - 2 million image-caption pairs from Conceptual Captions dataset
* VLM implementation in plain PyTorch
* Distributed training loop with torch distributed data parallel
* Pretraining the model on total of around 10B parameters (work in progress)
