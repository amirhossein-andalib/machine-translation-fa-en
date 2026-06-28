import numpy as np
from hazm import Normalizer, WordTokenizer
import re


class TranslationDataPreparation:
    def __init__(self):
        self.normalizer = Normalizer()
        self.fa_tokenizer = WordTokenizer()

        # Persian to English parallel corpus
        self.parallel_corpus = [
            ("سلام", "hello"),
            ("خداحافظ", "goodbye"),
            ("ممنون", "thank you"),
            ("لطفاً", "please"),
            ("بله", "yes"),
            ("خیر", "no"),
            ("من زبان‌شناسی می‌خوانم", "I study linguistics"),
            ("او کتاب می‌خواند", "he reads a book"),
            ("ما در ایران زندگی می‌کنیم", "we live in Iran"),
            ("هوش مصنوعی مهم است", "artificial intelligence is important"),
            ("زبان فارسی زیباست", "Persian language is beautiful"),
            ("دانشگاه شیراز بزرگ است", "Shiraz university is large"),
            ("من پایتون یاد می‌گیرم", "I am learning Python"),
            ("پردازش زبان طبیعی جالب است",
             "natural language processing is interesting"),
            ("شبکه عصبی آموزش می‌بیند", "neural network is training"),
            ("این پروژه مهم است", "this project is important"),
            ("زبان‌شناسی علم زبان است", "linguistics is the science of language"),
            ("ترجمه ماشینی دشوار است", "machine translation is difficult"),
            ("مدل یاد می‌گیرد", "the model is learning"),
            ("داده‌ها مهم هستند", "data is important"),
            ("کامپیوتر سریع است", "computer is fast"),
            ("ایران کشور زیبایی است", "Iran is a beautiful country"),
            ("تهران پایتخت ایران است", "Tehran is the capital of Iran"),
            ("او فارسی صحبت می‌کند", "he speaks Persian"),
            ("من انگلیسی یاد می‌گیرم", "I am learning English"),
            ("این کتاب جالب است", "this book is interesting"),
            ("دانشجو درس می‌خواند", "the student is studying"),
            ("استاد تدریس می‌کند", "the professor is teaching"),
            ("برنامه‌نویسی مهارت مهمی است",
             "programming is an important skill"),
            ("یادگیری ماشین آینده است",
             "machine learning is the future"),
        ]

        self.fa_word2idx = {'<PAD>': 0, '<START>': 1,
                            '<END>': 2, '<UNK>': 3}
        self.en_word2idx = {'<PAD>': 0, '<START>': 1,
                            '<END>': 2, '<UNK>': 3}
        self.fa_idx2word = {}
        self.en_idx2word = {}

    def normalize_persian(self, text):
        """Normalize Persian text"""
        text = self.normalizer.normalize(text)
        text = re.sub(r'[،؛؟!.«»\(\)\[\]{}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_persian(self, text):
        """Tokenize Persian text"""
        return self.fa_tokenizer.tokenize(text)

    def tokenize_english(self, text):
        """Tokenize English text"""
        return text.lower().split()

    def build_vocabularies(self):
        """Build vocabularies for both languages"""
        fa_words = set()
        en_words = set()

        for fa, en in self.parallel_corpus:
            fa_tokens = self.tokenize_persian(
                self.normalize_persian(fa)
            )
            en_tokens = self.tokenize_english(en)
            fa_words.update(fa_tokens)
            en_words.update(en_tokens)

        for word in sorted(fa_words):
            if word not in self.fa_word2idx:
                self.fa_word2idx[word] = len(self.fa_word2idx)

        for word in sorted(en_words):
            if word not in self.en_word2idx:
                self.en_word2idx[word] = len(self.en_word2idx)

        self.fa_idx2word = {
            idx: word for word, idx in self.fa_word2idx.items()
        }
        self.en_idx2word = {
            idx: word for word, idx in self.en_word2idx.items()
        }

        print(f"   واژگان فارسی : {len(self.fa_word2idx)}")
        print(f"   واژگان انگلیسی: {len(self.en_word2idx)}")

    def encode_sentence(self, tokens, word2idx, max_len):
        """Encode tokens to indices with padding"""
        indices = [word2idx.get(t, 3) for t in tokens]
        indices = indices[:max_len]
        indices += [0] * (max_len - len(indices))
        return indices

    def prepare_data(self, max_fa_len=12, max_en_len=12):
        """Prepare all training data"""
        print("\n⚙️  آماده‌سازی داده‌ها...")
        self.build_vocabularies()

        X = []
        y = []

        for fa, en in self.parallel_corpus:
            fa_tokens = self.tokenize_persian(
                self.normalize_persian(fa)
            )
            en_tokens = ['<START>'] + \
                self.tokenize_english(en) + ['<END>']

            fa_encoded = self.encode_sentence(
                fa_tokens, self.fa_word2idx, max_fa_len
            )
            en_encoded = self.encode_sentence(
                en_tokens, self.en_word2idx, max_en_len
            )

            X.append(fa_encoded)
            y.append(en_encoded)

        print(f"   تعداد جفت‌های ترجمه: {len(X)}")
        return np.array(X), np.array(y)
