"""Единый формат логов для бота, Haystack и клиента Telegram."""

from __future__ import annotations

import logging
import sys


def setup_logging() -> None:
    """Включает журналирование в консоль.

    Сообщения чата целиком пишутся только на уровне DEBUG.
    На INFO остаётся, кто писал, в каком чате и чем закончился вызов пайплайна.
    На Windows консоль переключается на UTF-8, чтобы русские строки не роняли лог.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("telebot").setLevel(logging.INFO)
