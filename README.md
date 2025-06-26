# Multimodal Telegram‑bot ChatGPT

Мультимодальный Telegram‑бот на базе OpenAI GPT-4o-mini с поддержкой текста, голоса и изображений.

## Возможности

- **Текст** ↔ GPT‑4o‑mini (Vision)
- **Голос** → текст (Whisper‑1)
- **Фото** → анализ (подпись = вопрос)
- **/img** — генерация изображений (DALL·E 3)
- **Учёт затрат**, отчёт **/stats** (только для администратора)
- **SQLite‑история**, команда **/reset** очищает историю
- Reply‑клавиатура: админ — /img /reset /stats, остальные — /img /reset

## Быстрый старт

### 1. Клонируйте репозиторий

git clone https://github.com/zergont/Bot_GPT.git cd Bot_GPT

### 2. Создайте и активируйте виртуальное окружение

python -m venv venv

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

### 3. Установите зависимости

pip install -r requirements.txt

### 4. Установите ffmpeg

- **Windows:** [Скачать с сайта ffmpeg.org](https://ffmpeg.org/download.html) и добавить в PATH
- **Linux:** `sudo apt install ffmpeg`
- **Mac:** `brew install ffmpeg`

### 5. Настройте переменные окружения

Скопируйте файл `.env.example` в `.env` и заполните его:

cp .env.example .env

- `TELEGRAM_BOT_TOKEN` — токен вашего Telegram-бота
- `OPENAI_API_KEY` — API-ключ OpenAI
- `ADMIN_ID` — ваш Telegram user ID (для доступа к статистике)

### 6. Управление ботом

Для удобства используйте скрипт управления:

python bot_control.py start			# Запуск бота python 
bot_control.py stop					# Остановка бота python 
bot_control.py status			    # Проверка статуса python 
bot_control.py update			    # Обновление из GitHub и перезапуск

## Структура проекта

- `Bot_GPT.py` — основной код бота
- `bot_control.py` — скрипт управления (start/stop/status/update)
- `requirements.txt` — зависимости
- `.env.example` — пример файла настроек
- `tmp/` — временные файлы (игнорируется git)
- `history.db` — база данных SQLite (игнорируется git)

## Советы

- Для работы с голосом и изображениями необходимы ffmpeg и ffprobe.
- Все чувствительные данные храните только в `.env`.
- Для обновления используйте команду `python bot_control.py update`.

## Лицензия

MIT

---

**Репозиторий:** [https://github.com/zergont/Bot_GPT](https://github.com/zergont/Bot_GPT)

