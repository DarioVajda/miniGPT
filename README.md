# MiniGPT

In this project I implemented and trained a small (decoder-only) transformer on the [Tiny Shakespeare](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt&ved=2ahUKEwjP8fri5OGLAxXN2AIHHdoGNR4QFnoECBUQAQ&usg=AOvVaw1IimzpEutw_xJxKH0xyDb1) dataset using PyTorch

The project was inspired by a youtube lecture by Andrej Karpathy on Generative Pre-trained Transformers.

Key components of the project:
1. miniGPT.py - implementation of the transformer module.
2. miniGPT_train.py - contains the training loop for the transformer.
3. miniGPT_inference.py - running this program will generate some text by loading the model weights from a saved file.
4. playground.ipynb - this is where I wrote some notes and where I tested some code while watching the lecture.
5. output.txt - SAMPLE OUTPUT with 2000 tokens

The dataset used for training is also available in the repository.

### Sample output:

*Servant:*  
*Had he counts to make him very way.*  
  
*Page:*  
*I will plead with him to serve for him*  
*As fear our service: he's no more recorder than him:*  
*Who with his land, madam, she was the sun affection*  
*Had proclaim'd with my dagger she in our hands*  
*To unfolding a goodness of the city's face.*  
  
*DUKE OF AUMERLE:*  
*I speak not a sorry,*  
*Do lost you, the villain of my heart,*  
*Is not he wounds forth my father's life.*  
*This Unresheard effect on the deputy*  
*The strong of his gratifice next known.*  
*...*  
  
### Final thoughts:

- I would like to note that the actual generated text doesn't carry much meaning, but we can see that even a small model like this can learn some structure in language.
- **Potential improvement** - using a more sophisticated tokenisation method instead of characterwise tokenisation would probably give better results, but this may require a larger model which would be computationally expensive to train.
