"""Настройки бота из переменных окружения и файла .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# text-embedding-3-small по умолчанию отдаёт 1536 чисел.
# Параметр dimensions у этой модели позволяет сузить вектор под уже созданный индекс.
_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass(frozen=True)
class Settings:
    """Ключи и имена моделей, без которых бот не стартует.

    :param openai_api_key: Ключ OpenAI. Эмбеддинги и генерация идут в API OpenAI напрямую.
    :param chat_model: Имя chat-модели, например gpt-4o-mini.
    :param embedding_model: Имя embedding-модели, например text-embedding-3-small.
    :param embedding_dimensions: Размерность вектора модели по умолчанию.
    :param pinecone_api_key: Ключ Pinecone.
    :param pinecone_index_name: Имя уже существующего или создаваемого индекса.
    :param telegram_bot_token: Токен бота от BotFather.
    :param pinecone_namespace: Отдельное пространство внутри индекса для сообщений чатов.
    """

    openai_api_key: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    pinecone_api_key: str
    pinecone_index_name: str
    telegram_bot_token: str
    pinecone_namespace: str = "company-chat"

    @classmethod
    def from_env(cls) -> "Settings":
        """Читает .env в корне проекта и проверяет, что обязательные значения не пустые.

        :returns: Заполненный набор настроек.
        :raises SystemExit: Если хотя бы одна обязательная переменная пустая.
        """

        load_dotenv(PROJECT_ROOT / ".env")
        values = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
            "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "").strip() or "gpt-4o-mini",
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small",
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", "").strip(),
            "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "").strip() or "nemo",
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        }
        missing = [
            name
            for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "TELEGRAM_BOT_TOKEN")
            if not values[name]
        ]
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(f"В .env не заполнены переменные: {joined}")

        embedding_model = values["OPENAI_EMBEDDING_MODEL"]
        return cls(
            openai_api_key=values["OPENAI_API_KEY"],
            chat_model=values["OPENAI_CHAT_MODEL"],
            embedding_model=embedding_model,
            embedding_dimensions=_EMBEDDING_DIMENSIONS.get(embedding_model, 1536),
            pinecone_api_key=values["PINECONE_API_KEY"],
            pinecone_index_name=values["PINECONE_INDEX_NAME"],
            telegram_bot_token=values["TELEGRAM_BOT_TOKEN"],
        )

    @property
    def embedding_model_accepts_dimensions(self) -> bool:
        """Модели семейства text-embedding-3 умеют отдавать вектор заданной длины.

        :returns: True, если в запрос эмбеддинга можно передать dimensions.
        """

        return "text-embedding-3" in self.embedding_model
