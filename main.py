"""Точка входа: читает .env, собирает пайплайны и запускает long polling."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

from team_bot.config import Settings
from team_bot.logging_setup import setup_logging
from team_bot.memory import TeamMemory
from team_bot.telegram_app import build_bot

logger = logging.getLogger(__name__)


def main() -> None:
    """Запускает бота и держит соединение с Telegram.

    Запись в Pinecone и ответы модели происходят только после команд в чате.
    """

    setup_logging()
    settings = Settings.from_env()
    memory = TeamMemory(settings)
    bot = build_bot(settings.telegram_bot_token, memory)
    logger.info("Long polling запущен")
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=30)


if __name__ == "__main__":
    main()
