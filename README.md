# Persian to English Machine Translation

🌐 **English | [فارسی](README.fa.md)**

Neural machine translation from Persian to English
using Sequence-to-Sequence LSTM architecture.

## Features
- Persian text normalization with Hazm
- Seq2Seq Encoder-Decoder architecture
- LSTM-based translation model
- Inference mode for sentence translation
- Parallel Persian-English corpus
- Honest generalization check on a never-seen sentence

## Model Architecture

### Encoder
- Embedding Layer (64 dims, mask_zero=True)
- Dropout (0.3)
- LSTM (128 units) → context vectors

### Decoder
- Embedding Layer (64 dims, mask_zero=True)
- Dropout (0.3)
- LSTM (128 units)
- Dense + Softmax output

## Bug Fix from the Original Version (Important)
The Encoder's and Decoder's Embedding layers were missing
`mask_zero=True`. The `<PAD>` token (index 0), used to pad short
sentences up to `max_len`, was being fed into the LSTM exactly like a
real word. Since most sentences are shorter than `max_len=12`, the
Encoder's final state (which is supposed to carry the meaning of the
whole sentence) was effectively washed out by padding — tested it:
the final state for two completely different sentences differed by
only ~1.7%! As a result, almost every translation collapsed to the
same wrong output ("i am learning english" for everything). After
adding `mask_zero=True`, the difference between states for different
sentences increased 60x, and all 6 test translations became exactly
correct.

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py
```

## Example
| Persian | English |
|---------|---------|
| من زبان‌شناسی می‌خوانم | I study linguistics |
| هوش مصنوعی مهم است | artificial intelligence is important |
| زبان فارسی زیباست | Persian language is beautiful |

## ⚠️ Important Limitation
The dataset has only 30 sentence pairs. The 6 main test sentences are
copied verbatim from this same training set (i.e. they show the model
has memorized them correctly). For a more honest check, a completely
new sentence is also tested in `main.py`, showing the model gets the
sentence structure mostly right but may mix up the content word with
the closest sentence it has seen — a genuinely reliable translation
model would need thousands/millions of sentence pairs.

## Technologies
- Python 3.x
- TensorFlow / Keras
- Hazm — Persian NLP
- NumPy

## Author
Seyed Amirhossein Andalib — Linguistics & NLP
