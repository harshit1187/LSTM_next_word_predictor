# Next-Word Prediction Using LSTM in PyTorch

This repository contains a PyTorch implementation of a character/word-level language model designed to perform **next-word prediction** using a Long Short-Term Memory (LSTM) network. The model is trained on a custom FAQ document containing detailed information about a Data Science Mentorship Program.

## 🚀 Project Overview

The objective of this project is to build an autoregressive language model that can generate coherent text sequentially. Given an input prompt, the model predicts the most probable next word in the sequence. This approach is highly relevant for applications such as auto-complete systems, smart reply generators, and domain-specific conversational assistants.

### Key Features
- **NLTK-based Tokenization**: Text preprocessing and word tokenization using NLTK.
- **Custom Dataset Pipeline**: Custom PyTorch `Dataset` and `DataLoader` pipelines with sequential padding for variable-length inputs.
- **LSTM Architecture**: Recurrent neural network with an embedding layer, LSTM cell, and a fully connected output projection layer.
- **GPU Acceleration**: Built-in CUDA compatibility for high-speed model training.
- **Text Generation Loop**: Inference script capable of producing multi-token text continuations given an arbitrary prompt.

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Deep Learning Framework**: [PyTorch](https://pytorch.org/)
* **NLP Library**: [NLTK](https://www.nltk.org/)
* **Numeric Processing**: NumPy

---

## 📊 Dataset & Preprocessing

The model is trained on structural FAQ text. The text processing steps include:
1. **Normalization**: Case-folding text to lowercase.
2. **Tokenization**: Segmenting sentences using `nltk.word_tokenize`.
3. **Vocabulary Building**: Mapping unique tokens to numeric indices, incorporating an `<unk>` token for out-of-vocabulary terms.
4. **Sequence Creation**: Generating continuous sub-sequences of tokens (n-grams) to simulate predictive contexts.
5. **Pre-Padding**: Pad training sequences from the left with `0` tokens to ensure stable, equal-length inputs (`max_len = 62`) while preserving the target sequence end-positions.

---

## 🧠 Model Architecture

The model uses a recurrent architecture optimized for sequence processing:

```
Input Indices (batch_size, seq_len)
       │
[ Embedding Layer ]  (vocab_size -> 100 dimensions)
       │
[   LSTM Layer   ]  (Hidden dimension = 150, Single layer)
       │  (Extracting final hidden state)
[ Linear Output  ]  (Linear projection to vocab_size)
       │
Softmax Activation (Implicit in CrossEntropyLoss)
```

---

## 📈 Training and Performance

* **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
* **Optimizer**: Adam Optimizer (`torch.optim.Adam`)
* **Learning Rate**: 0.01
* **Epochs**: 50
* **Final Training Accuracy**: ~**95.54%**

```bash
# Sample loss progression
Epoch 1: Loss 153.80
Epoch 25: Loss 3.76
Epoch 50: Loss 3.37
```

---

## 🔮 Inference & Generation

To run next-word prediction using the trained model, you can run the inference loop with a custom prompt:

```python
# Usage Example
input_text = "the course follows a monthly"
for i in range(20):
    input_text = prediction(model, vocab, input_text)
print(input_text)
```
**Expected Output:**
> `the course follows a monthly subscription model where you have to make monthly payments of rs 799/month . so the total course fee becomes 799`
```
