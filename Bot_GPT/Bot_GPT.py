"""
Multimodal Telegram‑bot ChatGPT
===============================
python‑telegram‑bot 22 · openai ≥ 1.0

* **Текст** ↔ GPT‑4o‑mini (Vision)
* **Голос** → текст (Whisper‑1)
* **Фото** → анализ (подпись = вопрос)
* **/img** — генерация (DALL·E 3)
* **Учёт затрат**, отчёт **/stats** (только админ)
* **SQLite‑история**, **/reset** её очищает
* Reply‑клавиатура: админ — /img /reset /stats, остальные — /img /reset
"""
from __future__ import annotations
import base64, json, logging, os, shutil, uuid, subprocess
import asyncio
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
import openai
import html
import atexit
import aiosqlite

# ──────────── CONFIG ────────────
ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp"; TMP_DIR.mkdir(exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL_CHAT = "gpt-4o-mini"; COST_IN, COST_OUT = 0.0005, 0.001
MODEL_TRANSCRIBE = "whisper-1"; WHISPER_RATE = 0.006 / 60
MODEL_IMAGE = "dall-e-3"; IMG_PRICE = 0.04

# ffmpeg/ffprobe — работает и в Windows, и в Linux
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise SystemExit("ffmpeg и ffprobe должны быть установлены и доступны в PATH")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot")
logging.getLogger("telegram.request").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

KB_USER = ReplyKeyboardMarkup([["/img", "/reset"]], resize_keyboard=True)
KB_ADMIN = ReplyKeyboardMarkup([["/img", "/reset", "/stats"]], resize_keyboard=True)
kb = lambda uid: KB_ADMIN if uid == ADMIN_ID else KB_USER

# ──────────── DB ────────────
DB: aiosqlite.Connection = None

async def init_db():
    global DB
    DB = await aiosqlite.connect(ROOT / "history.db")
    await DB.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
      user_id INTEGER, role TEXT, content TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS usage (
      user_id INTEGER, event TEXT, prompt_t INTEGER, compl_t INTEGER,
      seconds REAL, img_cnt INTEGER, cost_usd REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS users (
      user_id INTEGER PRIMARY KEY, username TEXT, first_name TEXT, last_name TEXT);
    """)
    await DB.commit()

@atexit.register
def cleanup():
    if DB is not None:
        asyncio.get_event_loop().run_until_complete(DB.close())
        log.info("Database connection closed.")

async def add_user(u):
    await DB.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (u.id, u.username, u.first_name, u.last_name))
    await DB.commit()

async def save_msg(uid: int, role: str, content: Any):
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)
    await DB.execute("INSERT INTO messages VALUES (?,?,?,?)", 
                    (uid, role, content, datetime.now(UTC).isoformat()))
    await DB.commit()

async def save_usage(**kw):
    await DB.execute(
        """INSERT INTO usage (user_id,event,prompt_t,compl_t,seconds,img_cnt,cost_usd)
           VALUES (:user_id,:event,:prompt_t,:compl_t,:seconds,:img_cnt,:cost)""",
        kw,
    )
    await DB.commit()
        
async def history(uid: int, limit: int = 20):
    async with DB.execute(
        "SELECT role,content FROM messages WHERE user_id=? ORDER BY ts DESC LIMIT ?",
        (uid, limit),
    ) as cursor:
        rows = await cursor.fetchall()
    rows = rows[::-1]
    out = []
    for r, c in rows:
        try:
            out.append({"role": r, "content": json.loads(c)})
        except json.JSONDecodeError:
            out.append({"role": r, "content": c})
    return out

# ──────────── Вспомогательные функции для аудио ────────────
def convert_ogg_to_wav(ogg_path: Path, wav_path: Path):
    result = subprocess.run(
        [FFMPEG, "-y", "-i", str(ogg_path), str(wav_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка конвертации ogg в wav: {result.stderr.decode()}")

def get_audio_duration(wav_path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка определения длительности аудио: {result.stderr.decode()}")
    return float(result.stdout.decode().strip())

# ──────────── OpenAI ────────────
async def chat_cost(uid: int, msgs):
    try:
        resp = await client.chat.completions.create(model=MODEL_CHAT, messages=msgs)
        u = resp.usage
        cost = (u.prompt_tokens * COST_IN + u.completion_tokens * COST_OUT) / 1000
        await save_usage(user_id=uid, event="chat", prompt_t=u.prompt_tokens, compl_t=u.completion_tokens, seconds=None, img_cnt=None, cost=cost)
        log.info("[COST] %s chat $%.4f", uid, cost)
        return resp.choices[0].message.content.strip()
    except openai.APIError as e:
        log.error(f"OpenAI API error: {e}")
        return "Извините, произошла ошибка при обработке запроса."
    except Exception as e:
        log.error(f"Unexpected error in chat_cost: {e}")
        return "Произошла внутренняя ошибка."

async def transcribe_cost(uid: int, wav: Path):
    sec = get_audio_duration(wav)
    with open(wav, "rb") as f:
        tr = await client.audio.transcriptions.create(model=MODEL_TRANSCRIBE, file=f)
    cost = sec * WHISPER_RATE
    await save_usage(user_id=uid, event="voice", prompt_t=None, compl_t=None, seconds=sec, img_cnt=None, cost=cost)
    log.info("[COST] %s voice $%.4f", uid, cost)
    return tr.text.strip()

async def generate_image(prompt: str):
    return (await client.images.generate(model=MODEL_IMAGE, prompt=prompt, n=1, size="1024x1024")).data[0].url

async def image_cost(uid: int, prompt: str):
    url = await generate_image(prompt)
    await save_usage(user_id=uid, event="img", prompt_t=None, compl_t=None, seconds=None, img_cnt=1, cost=IMG_PRICE)
    log.info("[COST] %s img $%.2f", uid, IMG_PRICE)
    return url

# ──────────── Handlers ────────────
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await add_user(update.effective_user)
    await update.message.reply_text("Привет! Я мультимодальный бот.", reply_markup=kb(update.effective_user.id))

async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await DB.execute("DELETE FROM messages WHERE user_id=?", (uid,))
    await DB.execute("DELETE FROM usage WHERE user_id=?", (uid,))
    await DB.commit()
    await update.message.reply_text("История очищена.", reply_markup=kb(uid))

async def stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    async with DB.execute(
        """SELECT u.username, COALESCE(u.first_name,''), COALESCE(u.last_name,''), SUM(cost_usd)
           FROM usage JOIN users u USING(user_id)
          GROUP BY user_id ORDER BY 4 DESC""") as cursor:
        rows = await cursor.fetchall()

    total = sum(r[3] or 0 for r in rows)
    body = "\n".join(
        f"{i+1}. {html.escape('@'+r[0]) if r[0] else html.escape((r[1]+' '+r[2]).strip() or 'Anon')} — ${r[3]:.4f}"
        for i, r in enumerate(rows)
    )
    text = f"💰 <b>Всего:</b> ${total:.4f}\n{body}"
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb(ADMIN_ID))

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Конвертируем голос в текст и пересылаем в GPT."""
    log.info(f"Processing voice message from user {update.effective_user.id}")
    u = update.effective_user; await add_user(u)
    v = update.message.voice
    work = TMP_DIR / f"v_{uuid.uuid4().hex}"; work.mkdir()
    ogg, wav = work / "v.oga", work / "v.wav"
    text = ""
    try:
        log.info(f"Downloading voice message from user {u.id}")
        await (await ctx.bot.get_file(v.file_id)).download_to_drive(str(ogg))
        log.info(f"Converting voice message to WAV format for user {u.id}")
        convert_ogg_to_wav(ogg, wav)
        log.info(f"Transcribing voice message for user {u.id}")
        text = await transcribe_cost(u.id, wav)
        log.info(f"Transcription completed for user {u.id}: {text}")
    except Exception as e:
        log.error(f"Error processing voice: {e}", exc_info=True)
        await update.message.reply_text("Ошибка обработки голосового сообщения.")
        return
    finally:
        log.info(f"Cleaning up temporary files for user {u.id}")
        shutil.rmtree(work, ignore_errors=True)

    await save_msg(u.id, "user", text)
    reply = await chat_cost(u.id, await history(u.id))
    await save_msg(u.id, "assistant", reply)
    await update.message.reply_text(reply, reply_markup=kb(u.id))

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Фото → описание (Vision)"""
    u = update.effective_user; await add_user(u)
    p = update.message.photo[-1]
    cap = (update.message.caption or "Опиши, что на фото").strip()
    work = TMP_DIR / f"p_{uuid.uuid4().hex}"; work.mkdir(); jpg = work / "img.jpg"
    try:
        await (await ctx.bot.get_file(p.file_id)).download_to_drive(str(jpg))
        uri = "data:image/jpeg;base64," + base64.b64encode(jpg.read_bytes()).decode()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": cap}
    ]}]
    reply = await chat_cost(u.id, msg)
    await save_msg(u.id, "user", msg[0]["content"]); await save_msg(u.id, "assistant", reply)
    await update.message.reply_text(reply, reply_markup=kb(u.id))

async def cmd_img(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Генерируем изображение DALL·E 3"""
    u = update.effective_user; await add_user(u)
    prompt = " ".join(ctx.args).strip()
    if not prompt:
        await update.message.reply_text("⚠️ Укажите промпт: /img <описание>", reply_markup=kb(u.id))
        return
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_PHOTO)
    url = await image_cost(u.id, prompt)
    await update.message.reply_photo(photo=url, caption=prompt, reply_markup=kb(u.id))

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user; await add_user(u)
    text = update.message.text.strip()
    await save_msg(u.id, "user", text)
    reply = await chat_cost(u.id, await history(u.id))
    await save_msg(u.id, "assistant", reply)
    await update.message.reply_text(reply, reply_markup=kb(u.id))

async def _notify_startup(app):
    try:
        await app.bot.send_message(chat_id=ADMIN_ID, text="🤖 Бот успешно запущен и готов к работе!")
    except Exception as e:
        log.warning("Не удалось отправить сообщение администратору: %s", e)

async def main() -> None:
    if not TG_TOKEN or not OPENAI_API_KEY:
        raise SystemExit("TELEGRAM_BOT_TOKEN или OPENAI_API_KEY не заданы")

    log.info("🛡️ ADMIN_ID: %s", ADMIN_ID)
    await init_db()
    app = (
        ApplicationBuilder()
        .token(TG_TOKEN)
        .post_init(_notify_startup)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("img", cmd_img))

    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Бот запущен (polling)…")
    await app.run_polling(poll_interval=10, drop_pending_updates=True)

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win") and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass  # Если nest_asyncio не установлен, просто продолжаем

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Только создаём задачу, не пытаемся run_until_complete!
        loop.create_task(main())
    else:
        loop.run_until_complete(main())