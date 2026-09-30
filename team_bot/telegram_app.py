"""Команды Telegram и запись сообщений группового чата."""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone

import telebot
from haystack import Document
from telebot import types

from team_bot.memory import TeamMemory
from team_bot.sessions import SessionRegistry

logger = logging.getLogger(__name__)

_TELEGRAM_LIMIT = 3900
_HELP = (
    "Я запоминаю рабочий чат и отвечаю по его истории.\n\n"
    "/listen — начать запись сообщений\n"
    "/stop — остановить запись и прислать итог\n"
    "Те же действия: /start_listen и /stop_listen\n\n"
    "Пока запись включена, каждое текстовое сообщение сохраняется "
    "вместе с автором и временем. На /stop весь этот фрагмент уходит в модель. "
    "Если это был спор, я дам своё мнение. Если договорились о действиях — "
    "зафиксирую решение и итог.\n\n"
    "Чтобы спросить по всей памяти этого чата, напишите вопрос с упоминанием "
    "бота или ответьте на любое моё сообщение. Я найду, кто и когда это говорил, "
    "и отдельно скажу, как считаю правильным поступить.\n\n"
    "Чтобы я видел обычные сообщения группы, а не только команды и упоминания, "
    "в @BotFather отключите Group Privacy (Bot Settings → Group Privacy → Turn off) "
    "и заново добавьте меня в чат."
)


class _Worker:
    """Одна очередь: реплики индексируются в том порядке, в котором пришли."""

    def __init__(self) -> None:
        """Запускает фоновый поток обработки сообщений."""

        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="team-bot-worker", daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> None:
        """Ставит задачу в очередь.

        :param job: Функция без аргументов.
        """

        self._jobs.put(job)

    def _loop(self) -> None:
        """Выполняет задачи по одной и не даёт исключению остановить поток."""

        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception:
                logger.exception("Задача бота завершилась с ошибкой")
            finally:
                self._jobs.task_done()


def build_bot(token: str, memory: TeamMemory) -> telebot.TeleBot:
    """Создаёт бота, регистрирует команды и начинает слушать очередь апдейтов.

    Сам long polling запускает вызывающий код.

    :param token: Токен Telegram.
    :param memory: Пайплайны записи, ответа и итога.
    :returns: Экземпляр TeleBot с зарегистрированными обработчиками.
    """

    bot = telebot.TeleBot(token, parse_mode=None, threaded=False)
    me = bot.get_me()
    bot_username = me.username or ""
    sessions = SessionRegistry()
    worker = _Worker()
    logger.info("Бот @%s готов к командам", bot_username)

    bot.set_my_commands(
        [
            types.BotCommand("listen", "Начать запись диалога"),
            types.BotCommand("stop", "Остановить запись и прислать итог"),
            types.BotCommand("start", "Как пользоваться ботом"),
        ]
    )

    def enqueue(handler, message: types.Message) -> None:
        """Кладёт обработку сообщения в общую очередь.

        :param handler: Функция, которая примет message.
        :param message: Апдейт Telegram.
        """

        worker.submit(lambda message=message: handler(message))

    @bot.message_handler(commands=["start", "help"])
    def on_help(message: types.Message) -> None:
        """Отвечает справкой. Не включает запись.

        :param message: Команда /start или /help.
        """

        enqueue(_send_help, message)

    @bot.message_handler(commands=["listen", "start_listen"])
    def on_listen(message: types.Message) -> None:
        """Включает запись контекста в этом чате.

        :param message: Команда старта слушания.
        """

        enqueue(_start_listening, message)

    @bot.message_handler(commands=["stop", "stop_listen"])
    def on_stop(message: types.Message) -> None:
        """Выключает запись, сохраняет итог и отправляет его в чат.

        :param message: Команда остановки слушания.
        """

        enqueue(_stop_listening, message)

    @bot.message_handler(content_types=["text"])
    def on_text(message: types.Message) -> None:
        """Пишет реплику в память, если запись включена, и отвечает на обращение.

        :param message: Обычное текстовое сообщение.
        """

        enqueue(_handle_text, message)

    @bot.message_handler(content_types=["photo", "video", "document", "audio", "voice", "animation"])
    def on_caption(message: types.Message) -> None:
        """Учитывает подпись к вложению как текстовую реплику.

        :param message: Сообщение с медиа и, возможно, caption.
        """

        enqueue(_handle_text, message)

    @bot.edited_message_handler(content_types=["text"])
    def on_edit(message: types.Message) -> None:
        """Перезаписывает уже сохранённую реплику, если её отредактировали во время записи.

        :param message: Отредактированное сообщение.
        """

        enqueue(_handle_text, message)

    def _send_help(message: types.Message) -> None:
        """Отправляет справку ответом на команду.

        :param message: Исходная команда.
        """

        _reply(bot, message, _HELP)

    def _start_listening(message: types.Message) -> None:
        """Открывает сессию записи и подтверждает это в чате.

        :param message: Команда старта.
        """

        session, created = sessions.start(message.chat.id)
        started = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if not created:
            _reply(
                bot,
                message,
                f"Запись уже идёт с {started}. Чтобы подвести итог, отправьте /stop.",
            )
            return
        logger.info("Старт записи chat_id=%s session=%s", message.chat.id, session.session_id)
        _reply(
            bot,
            message,
            "Слушаю этот чат и записываю следующие сообщения с автором и временем.\n"
            f"Старт: {started}.\n"
            "Когда обсуждение закончится, отправьте /stop: я сохраню фрагмент "
            "и пришлю итог. Если это был спор — добавлю своё мнение, "
            "если договорились о действиях — зафиксирую решение.\n\n"
            "Если я не вижу обычные реплики, в @BotFather отключите Group Privacy "
            "и добавьте меня в чат заново.",
        )

    def _stop_listening(message: types.Message) -> None:
        """Закрывает сессию, просит модель подвести итог и кладёт итог в Pinecone.

        :param message: Команда остановки.
        """

        session = sessions.stop(message.chat.id)
        if session is None:
            _reply(bot, message, "Запись не была включена. Начните с /listen.")
            return
        if not session.lines:
            logger.info("Стоп без реплик chat_id=%s session=%s", message.chat.id, session.session_id)
            _reply(bot, message, "Запись остановлена. В этом фрагменте не было текстовых сообщений.")
            return

        _reply(bot, message, "Останавливаю запись, отправляю диалог в модель и готовлю итог.")
        started = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            summary = memory.summarize(session.transcript(), started, len(session.lines))
        except Exception:
            sessions.restore(session)
            logger.exception("Не удалось построить итог chat_id=%s", message.chat.id)
            _reply(
                bot,
                message,
                "Итог сейчас не собрался, запись снова включена. Отправьте /stop ещё раз.",
            )
            return

        summary_document = _build_document(
            message,
            text=f"Итог записи с {started}:\n{summary}",
            session_id=session.session_id,
            kind="summary",
            bot_username=bot_username,
        )
        try:
            memory.index_message(summary_document)
        except Exception:
            logger.exception("Итог не записан в Pinecone chat_id=%s", message.chat.id)
            _reply(bot, message, "Итог готов, но записать его в векторную память не удалось.")

        _reply(bot, message, summary)
        logger.info(
            "Стоп записи chat_id=%s session=%s реплик=%s",
            message.chat.id,
            session.session_id,
            len(session.lines),
        )

    def _handle_text(message: types.Message) -> None:
        """Индексирует реплику во время записи и отвечает, если обратились к боту.

        :param message: Текст или подпись.
        """

        if message.chat.type not in {"private", "group", "supergroup"}:
            return
        if _is_command(message):
            return

        text = _extract_text(message)
        session = sessions.get(message.chat.id)
        if session is not None and text:
            sessions.add_line(message.chat.id, _format_line(message, text))
            document = _build_document(
                message,
                text=text,
                session_id=session.session_id,
                kind="message",
                bot_username=bot_username,
            )
            try:
                memory.index_message(document)
            except Exception:
                logger.exception(
                    "Не удалось записать сообщение chat_id=%s message_id=%s",
                    message.chat.id,
                    message.message_id,
                )
                _reply(
                    bot,
                    message,
                    "Это сообщение осталось в текущей стенограмме, но в векторную память не записалось.",
                )

        if _addressed_to_bot(message, me.id, bot_username):
            question = _question_without_mention(text, bot_username)
            if not question:
                _reply(bot, message, "Напишите вопрос в том же сообщении, где меня упоминаете.")
                return
            _reply(bot, message, "Смотрю память этого чата.")
            try:
                answer = memory.answer(message.chat.id, question)
            except Exception:
                logger.exception("Не удалось ответить chat_id=%s", message.chat.id)
                _reply(bot, message, "Не получилось пройтись по памяти чата. Подробности есть в логе.")
                return
            _reply(bot, message, answer)

    return bot


def _build_document(
    message: types.Message,
    *,
    text: str,
    session_id: str,
    kind: str,
    bot_username: str,
) -> Document:
    """Собирает документ Haystack с метаданными автора.

    :param message: Исходное сообщение Telegram. Для итога используется сообщение команды /stop.
    :param text: Текст, который попадёт в вектор.
    :param session_id: Идентификатор фрагмента записи.
    :param kind: ``message`` для реплики и ``summary`` для итога.
    :param bot_username: Имя бота, чтобы в итоге автором был он сам.
    :returns: Документ без эмбеддинга. Эмбеддинг считает пайплайн.
    """

    if kind == "summary":
        author_name = f"@{bot_username}" if bot_username else "бот"
        author_username = bot_username
        author_id = "bot"
        message_id = f"summary-{session_id}"
        document_id = f"summary-{message.chat.id}-{session_id}"
    else:
        author = _author(message)
        author_name = author["author_name"]
        author_username = author["author_username"]
        author_id = author["author_id"]
        message_id = str(message.message_id)
        document_id = f"msg-{message.chat.id}-{message.message_id}"

    timestamp = datetime.fromtimestamp(message.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return Document(
        id=document_id,
        content=text,
        meta={
            "chat_id": str(message.chat.id),
            "chat_title": message.chat.title or message.chat.username or "",
            "message_id": message_id,
            "author_id": author_id,
            "author_name": author_name,
            "author_username": author_username,
            "timestamp": timestamp,
            "kind": kind,
            "session_id": session_id,
        },
    )


def _author(message: types.Message) -> dict[str, str]:
    """Достаёт имя и идентификатор отправителя.

    :param message: Сообщение Telegram.
    :returns: Строковые поля автора, пригодные для метаданных Pinecone.
    """

    user = message.from_user
    if user is None:
        return {"author_id": "0", "author_name": "неизвестный участник", "author_username": ""}
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part).strip()
    return {
        "author_id": str(user.id),
        "author_name": full_name or "без имени",
        "author_username": user.username or "",
    }


def _format_line(message: types.Message, text: str) -> str:
    """Форматирует реплику так, чтобы в векторе сразу были время и автор.

    :param message: Сообщение Telegram.
    :param text: Текст или подпись.
    :returns: Одна строка стенограммы.
    """

    author = _author(message)
    username = f" (@{author['author_username']})" if author["author_username"] else ""
    timestamp = datetime.fromtimestamp(message.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"[{timestamp}] {author['author_name']}{username}: {text}"


def _extract_text(message: types.Message) -> str:
    """Возвращает текст сообщения или подпись к вложению.

    :param message: Сообщение Telegram.
    :returns: Текст без крайних пробелов или пустая строка.
    """

    raw = message.text if message.text is not None else message.caption
    return (raw or "").strip()


def _is_command(message: types.Message) -> bool:
    """Проверяет, является ли сообщение командой бота.

    :param message: Сообщение Telegram.
    :returns: True, если текст начинается с bot_command.
    """

    text = message.text or ""
    if not text.startswith("/"):
        return False
    return any(entity.type == "bot_command" and entity.offset == 0 for entity in message.entities or [])


def _addressed_to_bot(message: types.Message, bot_id: int, bot_username: str) -> bool:
    """Понимает, зовут ли бота: упоминание или ответ на его сообщение.

    :param message: Сообщение Telegram.
    :param bot_id: Числовой id бота.
    :param bot_username: Username бота без @.
    :returns: True, если сообщение адресовано боту.
    """

    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.id == bot_id:
        return True

    source = message.text if message.text is not None else (message.caption or "")
    entities = list(message.entities or []) + list(message.caption_entities or [])
    username = bot_username.lower()
    for entity in entities:
        if entity.type == "mention":
            mention = source[entity.offset : entity.offset + entity.length]
            if mention.lower() == f"@{username}":
                return True
        if entity.type == "text_mention" and entity.user is not None and entity.user.id == bot_id:
            return True
    return False


def _question_without_mention(text: str, bot_username: str) -> str:
    """Убирает @бота из текста, чтобы в поиск ушёл сам вопрос.

    :param text: Исходный текст.
    :param bot_username: Username бота без @.
    :returns: Вопрос без упоминания.
    """

    if not bot_username:
        return text.strip()
    cleaned = re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _reply(bot: telebot.TeleBot, message: types.Message, text: str) -> None:
    """Отправляет ответ частями: у Telegram лимит длины сообщения.

    :param bot: Экземпляр бота.
    :param message: Сообщение, на которое отвечаем.
    :param text: Полный текст ответа.
    """

    parts = _chunks(text)
    for index, part in enumerate(parts):
        if index == 0:
            bot.reply_to(message, part)
        else:
            bot.send_message(message.chat.id, part)


def _chunks(text: str, limit: int = _TELEGRAM_LIMIT) -> list[str]:
    """Режет длинный ответ по абзацам, не разрывая их без нужды.

    :param text: Полный текст.
    :param limit: Максимальная длина одной части.
    :returns: Список частей короче лимита Telegram.
    """

    cleaned = text.strip() or "Пустой ответ."
    if len(cleaned) <= limit:
        return [cleaned]

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in cleaned.split("\n"):
        piece = paragraph + "\n"
        if len(piece) > limit:
            if current:
                parts.append("".join(current).rstrip())
                current = []
                size = 0
            for start in range(0, len(piece), limit):
                parts.append(piece[start : start + limit].rstrip())
            continue
        if size + len(piece) > limit and current:
            parts.append("".join(current).rstrip())
            current = [piece]
            size = len(piece)
        else:
            current.append(piece)
            size += len(piece)
    if current:
        parts.append("".join(current).rstrip())
    return parts
