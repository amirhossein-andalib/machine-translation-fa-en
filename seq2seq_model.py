import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class Seq2SeqTranslator:
    def __init__(self, fa_vocab_size, en_vocab_size,
                 embedding_dim=64, lstm_units=128):
        self.fa_vocab_size = fa_vocab_size
        self.en_vocab_size = en_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self._build_model()

    def _build_model(self):
        """Build Seq2Seq model with Encoder-Decoder"""

        # ── Encoder ──
        encoder_inputs = Input(shape=(None,), name='encoder_input')
        # باگ بحرانی: mask_zero=True نبود. توکن <PAD> (ایندکس ۰) که برای
        # پر کردن جمله‌های کوتاه تا max_len استفاده می‌شود، بدون این پارامتر
        # عیناً مثل یک کلمه‌ی واقعی وارد LSTM می‌شود. چون اکثر جمله‌ها کوتاه‌تر
        # از max_len=12 هستند، LSTM چندین قدم پشت‌سرهم توکن padding را پردازش
        # می‌کند و state نهایی (که قراره معنای جمله را نگه دارد) عملاً محو
        # می‌شود — تست کردم: state نهایی برای دو جمله‌ی کاملاً متفاوت فقط ~۱.۷٪
        # تفاوت داشت! نتیجه: مدل تقریباً همیشه یک ترجمه‌ی ثابت تولید می‌کرد.
        encoder_embedding = Embedding(
            self.fa_vocab_size,
            self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        encoder_dropout = Dropout(0.3)(encoder_embedding)
        encoder_outputs, state_h, state_c = LSTM(
            self.lstm_units,
            return_state=True,
            name='encoder_lstm'
        )(encoder_dropout)
        encoder_states = [state_h, state_c]

        # ── Decoder ──
        decoder_inputs = Input(shape=(None,), name='decoder_input')
        # همان باگ، سمت دیکودر هم وجود داشت
        decoder_embedding = Embedding(
            self.en_vocab_size,
            self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        decoder_dropout = Dropout(0.3)(decoder_embedding)
        decoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_dropout,
            initial_state=encoder_states
        )
        decoder_dense = Dense(
            self.en_vocab_size,
            activation='softmax',
            name='decoder_output'
        )
        decoder_outputs = decoder_dense(decoder_outputs)

        # ── Training Model ──
        self.model = Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs
        )
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # ── Inference Encoder ──
        self.encoder_model = Model(
            encoder_inputs,
            encoder_states
        )

        # ── Inference Decoder ──
        decoder_state_input_h = Input(shape=(self.lstm_units,))
        decoder_state_input_c = Input(shape=(self.lstm_units,))
        decoder_states_inputs = [
            decoder_state_input_h,
            decoder_state_input_c
        ]
        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
            decoder_embedding,
            initial_state=decoder_states_inputs
        )
        decoder_states_inf = [state_h_inf, state_c_inf]
        decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs_inf] + decoder_states_inf
        )

    def train(self, X, y, epochs=100, batch_size=8):
        """Train the model"""
        print("\n🤖 آموزش مدل Seq2Seq...")
        print(self.model.summary())

        # Prepare decoder input and target
        decoder_input = y[:, :-1]
        decoder_target = y[:, 1:]

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            [X, decoder_input],
            np.expand_dims(decoder_target, -1),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def translate(self, input_seq, en_word2idx,
                  en_idx2word, max_len=12):
        """Translate a single sentence"""
        # Encode input
        states_value = self.encoder_model.predict(
            input_seq, verbose=0
        )

        # Generate empty target sequence
        target_seq = np.array([[en_word2idx['<START>']]])

        translated_words = []
        stop = False

        while not stop:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value, verbose=0
            )

            # Sample token
            sampled_token_idx = np.argmax(output_tokens[0, -1, :])
            sampled_word = en_idx2word.get(sampled_token_idx, '')

            if (sampled_word == '<END>' or
                    len(translated_words) >= max_len):
                stop = True
            else:
                if sampled_word not in ['<PAD>', '<START>', '<UNK>']:
                    translated_words.append(sampled_word)

            target_seq = np.array([[sampled_token_idx]])
            states_value = [h, c]

        return ' '.join(translated_words)
