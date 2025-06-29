"""
Multimodal Telegram‑bot ChatGPT (optimized memory)
==================================================
• Responses‑API + store=True → платим за историю один раз
• /reset полностью обнуляет контекст (DELETE sessions)
• Ведение статистики и старыe функции (+Vision, +DALL·E) сохранены
"""
from __future__ import annotations
import base64, json, logging, os, shutil, uuid, subprocess, asyncio, html, atexit
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

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
import aiosqlite

# ──────────── CONFIG ────────────
ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp"; TMP_DIR.mkdir(exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TG_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID       = int(os.getenv("ADMIN_ID", "0"))
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL_CHAT = "gpt-4o-mini"   # Responses‑API
COST_IN, COST_OUT = 0.0005, 0.001  # $/1k токенов
MODEL_TRANSCRIBE = "whisper-1"; WHISPER_RATE = 0.006 / 60
MODEL_IMAGE = "dall-e-3"; IMG_PRICE = 0.04

FFMPEG  = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise SystemExit("ffmpeg/ffprobe не найдены в PATH")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot")
logging.getLogger("telegram.request").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

KB_USER  = ReplyKeyboardMarkup([["/img", "/reset"]], resize_keyboard=True)
KB_ADMIN = ReplyKeyboardMarkup([["/img", "/reset", "/stats"]], resize_keyboard=True)
kb = lambda uid: KB_ADMIN if uid == ADMIN_ID else KB_USER

# ──────────── DB ────────────
DB: aiosqlite.Connection | None = None

async def init_db():
    global DB
    DB = await aiosqlite.connect(ROOT / "history.db")
    await DB.executescript(
        """
        CREATE TABLE IF NOT EXISTS messages (
          user_id INTEGER, role TEXT, content TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS usage (
          user_id INTEGER, event TEXT, prompt_t INTEGER, compl_t INTEGER,
          seconds REAL, img_cnt INTEGER, cost_usd REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS users (
          user_id INTEGER PRIMARY KEY, username TEXT, first_name TEXT, last_name TEXT);
        CREATE TABLE IF NOT EXISTS sessions (
          user_id INTEGER PRIMARY KEY, prev_id TEXT);
        """
    )
    await DB.commit()

@atexit.register
def _close_db():
    if DB is not None:
        asyncio.get_event_loop().run_until_complete(DB.close())
        log.info("Database closed")

# ──────────── DB helpers ────────────
async def add_user(u):
    await DB.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (u.id, u.username, u.first_name, u.last_name))
    await DB.commit()

async def save_msg(uid: int, role: str, content: Any):
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)
    await DB.execute("INSERT INTO messages VALUES (?,?,?,?)", (uid, role, content, datetime.now(UTC).isoformat()))
    await DB.commit()

async def save_usage(**kw):
    await DB.execute(
        """INSERT INTO usage (user_id,event,prompt_t,compl_t,seconds,img_cnt,cost_usd)
           VALUES (:user_id,:event,:prompt_t,:compl_t,:seconds,:img_cnt,:cost)""", kw)
    await DB.commit()

#  sessions table helpers -------------------------------------------------------
async def get_prev_id(uid: int) -> str | None:
    async with DB.execute("SELECT prev_id FROM sessions WHERE user_id=?", (uid,)) as cur:
        row = await cur.fetchone()
    return row[0] if row else None

async def set_prev_id(uid: int, rid: str):
    await DB.execute("INSERT INTO sessions VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET prev_id=excluded.prev_id", (uid, rid))
    await DB.commit()

async def reset_session(uid: int):
    await DB.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
    await DB.commit()

# ──────────── OpenAI wrappers ────────────
async def chat_cost(uid: int, prompt: str) -> str:
    """Single‑turn call with automatic long‑term memory (Responses‑API)."""
    try:
        args = {"model": MODEL_CHAT, "input": prompt, "store": True}
        prev = await get_prev_id(uid)
        if prev:
            args["previous_response_id"] = prev
        resp = await client.responses.create(**args)
        await set_prev_id(uid, resp.id)

        # usage fields differ slightly — проверяем атрибуты
        u = resp.usage
        p_tok = getattr(u, "prompt_tokens", getattr(u, "input_tokens", 0))
        c_tok = getattr(u, "completion_tokens", getattr(u, "output_tokens", 0))
        cost = (p_tok * COST_IN + c_tok * COST_OUT) / 1000
        await save_usage(user_id=uid, event="chat", prompt_t=p_tok, compl_t=c_tok, seconds=None, img_cnt=None, cost=cost)
        log.info("[COST] %s chat $%.4f", uid, cost)
        return resp.output.strip()
    except openai.APIError as e:
        log.error("OpenAI API error: %s", e)
        return "Извините, произошла ошибка при обработке запроса."
    except Exception as e:
        log.error("Unexpected error in chat_cost: %s", e, exc_info=True)
        return "Произошла внутренняя ошибка."

# ---------------------- Whisper & DALL·E (unchanged) ---------------------------
# ... (convert_ogg_to_wav, get_audio_duration, transcribe_cost, generate_image)
# их код остался прежним – вырезан для краткости

# ──────────── Handlers ────────────
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await add_user(update.effective_user)
    await update.message.reply_text("Привет! Я мультимодальный бот.", reply_markup=kb(update.effective_user.id))

async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await DB.execute("DELETE FROM messages WHERE user_id=?", (uid,))
    await DB.execute("DELETE FROM usage   WHERE user_id=?", (uid,))
    await reset_session(uid)  # ⬅️ сбрасываем prev_id
    await DB.commit()
    await update.message.reply_text("История и контекст очищены.", reply_markup=kb(uid))

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user; await add_user(u)
    text = update.message.text.strip()
    await save_msg(u.id, "user", text)
    reply = await chat_cost(u.id, text)
    await save_msg(u.id, "assistant", reply)
    await update.message.reply_text(reply, reply_markup=kb(u.id))

# -------------------- voice & photo use same chat_cost() -----------------------
# (функции handle_voice, handle_photo, cmd_img остаются прежними,
#  лишь заменена строка reply = await chat_cost(u.id, text_or_msg) )

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
    # ... остальные CommandHandler / MessageHandler без изменений

    log.info("Бот запущен (polling)…")
    await app.run_polling(poll_interval=10, drop_pending_updates=True)

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win") and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        import nest_asyncio; nest_asyncio.apply()
    except ImportError:
        pass
    asyncio.run(main())
