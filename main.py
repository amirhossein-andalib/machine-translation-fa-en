import numpy as np
from data_preparation import TranslationDataPreparation
from seq2seq_model import Seq2SeqTranslator


def main():
    print("=" * 55)
    print("   PERSIAN TO ENGLISH MACHINE TRANSLATION")
    print("   Seq2Seq with LSTM Encoder-Decoder")
    print("=" * 55)

    # ── Step 1: Prepare Data ──
    prep = TranslationDataPreparation()
    X, y = prep.prepare_data(max_fa_len=12, max_en_len=12)

    print(f"\n📊 اطلاعات داده:")
    print(f"   شکل X (فارسی) : {X.shape}")
    print(f"   شکل y (انگلیسی): {y.shape}")
    print(f"   واژگان فارسی  : {len(prep.fa_word2idx)}")
    print(f"   واژگان انگلیسی: {len(prep.en_word2idx)}")

    # ── Step 2: Train Model ──
    translator = Seq2SeqTranslator(
        fa_vocab_size=len(prep.fa_word2idx),
        en_vocab_size=len(prep.en_word2idx),
        embedding_dim=64,
        lstm_units=128
    )

    history = translator.train(X, y, epochs=100, batch_size=8)

    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    print(f"\n✅ نتایج آموزش:")
    print(f"   Loss نهایی    : {final_loss:.4f}")
    print(f"   Accuracy نهایی: {final_acc * 100:.2f}%")

    # ── Step 3: Test Translation ──
    print("\n" + "=" * 55)
    print("🔍 تست ترجمه:")
    print("=" * 55)

    test_sentences = [
        "من زبان‌شناسی می‌خوانم",
        "هوش مصنوعی مهم است",
        "زبان فارسی زیباست",
        "دانشگاه شیراز بزرگ است",
        "پردازش زبان طبیعی جالب است",
        "یادگیری ماشین آینده است",
    ]

    for sentence in test_sentences:
        # Preprocess
        normalized = prep.normalize_persian(sentence)
        tokens = prep.tokenize_persian(normalized)
        encoded = prep.encode_sentence(
            tokens, prep.fa_word2idx, max_len=12
        )
        input_seq = np.array([encoded])

        # Translate
        translation = translator.translate(
            input_seq,
            prep.en_word2idx,
            prep.en_idx2word
        )

        print(f"\n🇮🇷 فارسی  : {sentence}")
        print(f"🇬🇧 انگلیسی: {translation}")
        print("-" * 55)

    # ── Step 4: Honest Generalization Check ──
    # نکته‌ی مهم: ۶ جمله‌ی تست بالا عیناً از دیتای آموزشی (parallel_corpus)
    # کپی شده‌اند — یعنی نشان می‌دهند مدل آن‌ها را درست «حفظ» کرده، نه
    # اینکه روی جمله‌ی واقعاً جدید چطور عمل می‌کند. اینجا یک جمله‌ی کاملاً
    # جدید (که عیناً در دیتای آموزشی نیست، فقط از کلمات شناخته‌شده ساخته
    # شده) امتحان می‌کنیم تا تعمیم‌پذیری واقعی را صادقانه ببینیم.
    print("\n" + "=" * 55)
    print("🧪 تست تعمیم‌پذیری (جمله‌ای که مدل عیناً ندیده):")
    print("=" * 55)
    novel_sentence = "این دانشگاه جالب است"
    normalized = prep.normalize_persian(novel_sentence)
    tokens = prep.tokenize_persian(normalized)
    encoded = prep.encode_sentence(tokens, prep.fa_word2idx, max_len=12)
    novel_translation = translator.translate(
        np.array([encoded]), prep.en_word2idx, prep.en_idx2word
    )
    print(f"\n🇮🇷 فارسی  : {novel_sentence}")
    print(f"🇬🇧 انگلیسی: {novel_translation}")
    print(
        "\nℹ️  نکته: با فقط ۳۰ جفت جمله‌ی آموزشی، انتظار ترجمه‌ی کامل و "
        "دقیق برای جمله‌ی کاملاً جدید واقع‌بینانه نیست — مدل معمولاً "
        "ساختار جمله را تا حدی درست می‌گیرد ولی ممکن است کلمه‌ی محتوایی "
        "را با نزدیک‌ترین جمله‌ی آموزشی‌دیده قاطی کند. برای ترجمه‌ی "
        "واقعاً قابل‌اعتماد، این مدل به هزاران جفت جمله نیاز دارد."
    )
    print("-" * 55)


if __name__ == "__main__":
    main()
