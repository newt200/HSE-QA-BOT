import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Optional, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# Подключаем поиск (hybrid_search) + загрузку кеша вопросов + инициализацию моделей (ST + FAISS) один раз на старте
from find_candidates import hybrid_search, load_all_questions, init_models_once


# берем папку, где лежит bot.py, и ожидаем, что qa.db лежит рядом
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "qa.db")

# сколько символов показываем в "коротком" ответе (в основном сообщении)
SHORT_LIMIT = 900

FULL_CHUNK = 3500


def norm(s: str) -> str:
    """Нормализуем текст: схлопываем пробелы/переносы и обрезаем по краям."""
    return re.sub(r"\s+", " ", (s or "").strip())


@dataclass
class Row:
    """
    Удобная структура для одного результата из БД:
    - id: идентификатор записи
    - page: страница/раздел (если есть)
    - question: исходный вопрос из базы
    - answer: ответ из базы
    - source_url: ссылка на источник (если есть)
    """
    id: int
    page: str
    question: str
    answer: str
    source_url: str


def row_tuple_to_obj(t: tuple) -> Row:
    """
    Превращаем кортеж, который вернула SQLite (SELECT ...),
    в объект Row с безопасными значениями по умолчанию.
    """
    return Row(
        id=int(t[0]),
        page=t[1] or "unknown",
        question=t[2] or "",
        answer=t[3] or "",
        source_url=t[4] or "",
    )


def chunk_text(s: str, n: int) -> List[str]:
    """
    Делим длинный текст на части по n символов, чтобы корректно отправлять в Telegram.
    Если строка пустая — возвращаем список из одного пустого элемента (для единообразия).
    """
    s = s or ""
    return [s[i: i + n] for i in range(0, len(s), n)] if s else [""]


# ---------------- UI: клавиатура и форматирование сообщений ----------------

def make_keyboard(idx: int, total: int, show_q: bool, has_source: bool, has_full: bool) -> InlineKeyboardMarkup:
    """
    Собираем inline-клавиатуру под сообщением:
    - ряд навигации (Назад/Далее), чтобы листать результаты
    - ряд действий:
        * показать/скрыть вопрос в карточке
        * полный ответ (если короткий был обрезан)
        * источник (если в записи есть URL)
    """
    nav_row = []
    if idx > 0:
        nav_row.append(InlineKeyboardButton("⬅️ Назад", callback_data="prev"))
    if idx < total - 1:
        nav_row.append(InlineKeyboardButton("➡️ Далее", callback_data="next"))

    actions_row = [
        InlineKeyboardButton("Скрыть вопрос ❓" if show_q else "Показать вопрос ❓", callback_data="toggle_q"),
    ]
    if has_full:
        actions_row.append(InlineKeyboardButton("Полный ответ 📄", callback_data="full"))
    if has_source:
        actions_row.append(InlineKeyboardButton("Источник 🔗", callback_data="source"))

    buttons = []
    if nav_row:
        buttons.append(nav_row)
    buttons.append(actions_row)

    return InlineKeyboardMarkup(buttons)


def format_answer_message(query: str, row: Row, idx: int, total: int, show_q: bool) -> str:
    """
    Формируем "короткую карточку" ответа:
    - шапка: запрос пользователя + номер ответа из топа
    - тело: ответ (A), при необходимости обрезаем до SHORT_LIMIT
    - опционально добавляем текст вопроса (Q), если show_q=True
    """
    header = f"Запрос: {norm(query)}\n\nОтвет {idx + 1} из {total}"
    ans = norm(row.answer)

    short = ans
    if len(short) > SHORT_LIMIT:
        short = short[:SHORT_LIMIT] + "…"

    if show_q:
        return f"{header}\n\nQ: {norm(row.question)}\n\nA: {short}"

    return f"{header}\n\nA: {short}"


def format_full_answer(row: Row, show_q: bool) -> str:
    """
    Формируем "полный ответ" отдельным сообщением:
    - опционально показываем Q
    - показываем A полностью
    - если есть ссылка — добавляем строку "Источник: ..."
    """
    parts = []
    if show_q:
        parts.append(f"Q: {norm(row.question)}")
    parts.append(f"A: {norm(row.answer)}")
    if row.source_url:
        parts.append(f"Источник: {row.source_url}")
    return "\n\n".join(parts)


# держим соединение с SQLite глобально, чтобы не пересоздавать на каждый апдейт
con: Optional[sqlite3.Connection] = None

# кеш всех вопросов (id, question), грузим один раз на старте.
# даже если в текущем алгоритме FAISS+threshold этот кеш почти не нужен,
# мы оставляем его для совместимости и возможных будущих fallback-стратегий.
all_q_cache = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start — приветствие и краткая инструкция.
    """
    text = (
        "Привет! Я FAQ-бот.\n\n"
        "Напиши вопрос обычным текстом — я найду ответы и покажу их по одному.\n\n"
        "Команды:\n"
        "/help — как пользоваться\n"
        "/id <число> — показать полный ответ по ID\n"
    )
    await update.message.reply_text(text, disable_web_page_preview=True)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /help — более подробная инструкция по кнопкам и сценариям.
    """
    text = (
        "Как пользоваться:\n"
        "1) Просто отправь сообщение с вопросом.\n"
        "2) Я покажу лучший ответ.\n"
        "3) Нажимай кнопки «Назад/Далее», чтобы листать.\n"
        "4) Нажми «Полный ответ 📄», чтобы получить полный текст отдельным сообщением.\n\n"
        "Подсказки:\n"
        "• Пиши ключевые слова: «документы», «сроки», «справка», «приглашение», «общежитие».\n"
        "• Если видишь id:123 — можешь запросить полный ответ: /id 123\n"
    )
    await update.message.reply_text(text, disable_web_page_preview=True)


async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /id <число> — быстрый доступ к записи по ID:
    - проверяем аргумент
    - достаём запись из qa
    - отправляем полный ответ (возможно несколькими кусками)
    """
    if not context.args:
        await update.message.reply_text("Использование: /id 123", disable_web_page_preview=True)
        return

    try:
        qa_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("ID должен быть числом. Пример: /id 123", disable_web_page_preview=True)
        return

    if con is None:
        await update.message.reply_text("База не подключена.", disable_web_page_preview=True)
        return

    # достаем запись по ID
    with closing(con.cursor()) as cur:
        r = cur.execute(
            "SELECT id, page, question, answer_text, source_url FROM qa WHERE id = ?;",
            (qa_id,),
        ).fetchone()

    if not r:
        await update.message.reply_text("Не нашла запись с таким ID.", disable_web_page_preview=True)
        return

    row = row_tuple_to_obj(r)
    msg = format_full_answer(row, show_q=True)

    # Если ответ длинный — отправляем частями
    for part in chunk_text(msg, FULL_CHUNK):
        if part.strip():
            await update.message.reply_text(part, disable_web_page_preview=True)



async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатываем обычные сообщения:
    1) берем текст запроса
    2) вызываем hybrid_search из find_candidates
       (в нашей текущей версии: FAISS dense retrieval + reject по sem_thr)
    3) сохраняем результаты в context.user_data, чтобы кнопки работали
    4) отправляем первую карточку с inline-клавиатурой
    """
    text = (update.message.text or "").strip()
    if not text:
        return

    # проверяем, что база и кеш вопросов инициализированы
    if con is None or all_q_cache is None:
        await update.message.reply_text("База еще не инициализирована.", disable_web_page_preview=True)
        return

    # ищем top результатов.
    best, top, dbg = hybrid_search(
        con,
        text,
        final_k=5,
    )

    # превращаем кортежи из базы в Row-объекты
    rows = [row_tuple_to_obj(t) for t in (top or [])]
    if not rows:
        # если результатов нет,
        # очищаем состояние и просим переформулировать
        context.user_data.pop("results", None)
        context.user_data.pop("query", None)
        context.user_data.pop("idx", None)
        context.user_data.pop("show_q", None)
        await update.message.reply_text(
            "Не нашла подходящих ответов. Попробуй переформулировать вопрос.",
            disable_web_page_preview=True,
        )
        return

    # сохраняем состояние выдачи для кнопок
    context.user_data["query"] = text
    context.user_data["results"] = rows
    context.user_data["idx"] = 0
    context.user_data["show_q"] = False

    # формируем и отправляем первое сообщение
    idx = 0
    show_q = False
    msg = format_answer_message(text, rows[idx], idx, len(rows), show_q=show_q)
    has_full = len(norm(rows[idx].answer)) > SHORT_LIMIT
    kb = make_keyboard(idx, len(rows), show_q, has_source=bool(rows[idx].source_url), has_full=has_full)

    await update.message.reply_text(msg, reply_markup=kb, disable_web_page_preview=True)



async def on_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обрабатываем нажатия на inline-кнопки:
    - next/prev: переключаем индекс результата и обновляем карточку
    - toggle_q: показываем/скрываем вопрос внутри карточки
    - source: отправляем ссылку источника отдельным сообщением
    - full: отправляем полный ответ отдельным сообщением (возможно частями)
    """
    q = update.callback_query
    await q.answer()

    rows: List[Row] = context.user_data.get("results") or []
    if not rows:
        await q.edit_message_text("Нет сохраненных результатов. Напиши новый запрос 🙂", disable_web_page_preview=True)
        return

    query_text: str = context.user_data.get("query") or ""
    idx: int = int(context.user_data.get("idx") or 0)
    show_q: bool = bool(context.user_data.get("show_q") or False)

    data = q.data

    if data == "next" and idx < len(rows) - 1:
        idx += 1
        context.user_data["idx"] = idx

    elif data == "prev" and idx > 0:
        idx -= 1
        context.user_data["idx"] = idx

    elif data == "toggle_q":
        show_q = not show_q
        context.user_data["show_q"] = show_q

    elif data == "source":
        r = rows[idx]
        if r.source_url:
            await q.message.reply_text(f"Источник: {r.source_url}", disable_web_page_preview=True)
        return

    elif data == "full":
        r = rows[idx]
        full_msg = format_full_answer(r, show_q=show_q)
        for part in chunk_text(full_msg, FULL_CHUNK):
            if part.strip():
                await q.message.reply_text(part, disable_web_page_preview=True)
        return
    
    r = rows[idx]
    msg = format_answer_message(query_text, r, idx, len(rows), show_q=show_q)
    has_full = len(norm(r.answer)) > SHORT_LIMIT
    kb = make_keyboard(idx, len(rows), show_q, has_source=bool(r.source_url), has_full=has_full)

    await q.edit_message_text(msg, reply_markup=kb, disable_web_page_preview=True)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Логируем ошибки в stdout, чтобы видеть их в консоли/логах хостинга.
    """
    try:
        print("ERROR:", context.error)
    except Exception:
        pass


def main() -> None:
    """
    Запускаем бота:
    1) читаем BOT_TOKEN из окружения
    2) подключаем SQLite
    3) настраиваем PRAGMA для адекватной работы с WAL
    4) инициализируем модели/индекс один раз (SentenceTransformer + FAISS)
    5) грузим кеш вопросов
    6) регистрируем handlers и запускаем polling
    """
    global con, all_q_cache

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Set BOT_TOKEN env var: export BOT_TOKEN='123:ABC...'")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"qa.db not found рядом со скриптом: {DB_PATH}")

    # открываем соединение с SQLite (check_same_thread=False нужно, т.к. Telegram обработчики могут быть в разных потоках)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)

    # улучшаем параметры SQLite под нагрузку на чтение
    with closing(con.cursor()) as cur:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        con.commit()

    # инициализируем модель и FAISS индекс один раз, чтобы не строить их на каждый запрос
    # внутри init_models_once мы грузим эмбеддинги из qa_vec и строим индекс
    init_models_once(con=con)

    # грузим кеш вопросов (id, question) один раз
    all_q_cache = load_all_questions(con)

    # собираем приложение и регистрируем обработчики
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("id", id_cmd))

    # любой текст, который не команда, отправляем в on_text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # все callback_data от inline-кнопок обрабатываем в on_buttons
    app.add_handler(CallbackQueryHandler(on_buttons))

    # логируем ошибки через on_error
    app.add_error_handler(on_error)

    print("Bot is running. DB:", DB_PATH)

    app.run_polling(close_loop=False)

    # После остановки пытаемся аккуратно закрыть соединение
    try:
        con.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
