# Persian to English Machine Translation

🌐 **[English](README.md) | فارسی**

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

## Bug Fix از نسخه‌ی اولیه (مهم)
Embedding لایه‌های Encoder و Decoder، `mask_zero=True` نداشتند. توکن
`<PAD>` (ایندکس ۰) که برای پر کردن جمله‌های کوتاه تا `max_len` استفاده
می‌شود، بدون این پارامتر عیناً مثل یک کلمه‌ی واقعی وارد LSTM می‌شد.
چون اکثر جمله‌ها کوتاه‌تر از `max_len=12` بودند، state نهایی Encoder
(که باید معنای کل جمله را نگه دارد) عملاً توسط padding محو می‌شد —
تست شد: state نهایی برای دو جمله‌ی کاملاً متفاوت فقط ~۱.۷٪ تفاوت
داشت! نتیجه: تقریباً همه‌ی ترجمه‌ها یک خروجی ثابت و غلط بودند
("i am learning english" برای همه). با افزودن `mask_zero=True`،
تفاوت state بین جمله‌های مختلف ۶۰ برابر شد و همه‌ی ۶ ترجمه‌ی تستی
دقیقاً درست شدند.

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py
```

## Example
| فارسی | English |
|-------|---------|
| من زبان‌شناسی می‌خوانم | I study linguistics |
| هوش مصنوعی مهم است | artificial intelligence is important |
| زبان فارسی زیباست | Persian language is beautiful |

## ⚠️ محدودیت مهم
دیتاست فقط ۳۰ جفت جمله دارد. ۶ جمله‌ی تست اصلی عیناً از همین دیتاست
آموزشی هستند (یعنی نشان می‌دهند مدل آن‌ها را درست حفظ کرده). برای یک
تست صادقانه‌تر، یک جمله‌ی کاملاً جدید هم در `main.py` امتحان شده که
نشان می‌دهد مدل ساختار جمله را تا حدی درست می‌گیرد ولی ممکن است
کلمه‌ی محتوایی را با نزدیک‌ترین جمله‌ی دیده‌شده قاطی کند — برای ترجمه‌ی
واقعی و قابل‌اعتماد، چنین مدلی به هزاران/میلیون‌ها جفت جمله نیاز دارد.

## Technologies
- Python 3.x
- TensorFlow / Keras
- Hazm — Persian NLP
- NumPy

## Author
Seyed Amirhossein Andalib — Linguistics & NLP
