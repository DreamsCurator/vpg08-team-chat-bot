"""Пайплайны Haystack: запись в Pinecone, поиск по чату и итог обсуждения.

Эмбеддинги считает OpenAIDocumentEmbedder / OpenAITextEmbedder.
Текст генерирует OpenAIChatGenerator. Оба компонента вызывают официальный
клиент ``openai`` и ходят в API OpenAI напрямую, без промежуточного провайдера.
Схема RAG повторяет учебный пример: text embedder → retriever → prompt → llm.
"""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone

from team_bot.config import Settings

logger = logging.getLogger(__name__)

_TOP_K = 12
_TRANSCRIPT_LIMIT = 60_000
_STORED_TEXT_LIMIT = 12_000

_ANSWER_TEMPLATE = [
    ChatMessage.from_system(
        "Ты память рабочего чата одной компании. Отвечай по-русски.\n"
        "Ниже фрагменты сообщений, которые нашлись по вопросу. "
        "Это не вся переписка, а ближайшие по смыслу реплики.\n"
        "Правила:\n"
        "1. Опирайся на эти реплики. Не выдумывай, кто и что говорил.\n"
        "2. Если в контексте есть ответ, обязательно напиши, кто и когда это сказал, "
        "и коротко процитируй формулировку.\n"
        "3. Если данных мало, прямо скажи, чего в памяти нет.\n"
        "4. В конце отдельным абзацем «Как я считаю» дай своё мнение: "
        "что команде логичнее сделать и почему. Отдели мнение от фактов чата."
    ),
    ChatMessage.from_user(
        "Контекст:\n"
        "{% for document in documents %}\n"
        "- [{{ document.meta.timestamp | default('без даты') }}] "
        "{{ document.meta.author_name | default('неизвестный') }}: {{ document.content }}\n"
        "{% endfor %}\n"
        "\nВопрос: {{ question }}\n"
    ),
]

_SUMMARY_TEMPLATE = [
    ChatMessage.from_system(
        "Ты секретарь рабочей команды. Тебе дают стенограмму куска чата "
        "между командами «начать слушать» и «остановить». Отвечай по-русски.\n"
        "Структура ответа:\n"
        "1. Тема — одно-два предложения.\n"
        "2. Тип фрагмента: спор, решение, обсуждение или смешанный. Коротко почему.\n"
        "3. Ход разговора: кто какую позицию занял. Имена бери из стенограммы.\n"
        "4. Если это спор — блок «Мнение»: чья аргументация сильнее и что ты "
        "рекомендуешь команде. Не прячься за нейтральный пересказ, "
        "но не выдавай свою оценку за слова участников.\n"
        "5. Если договорились о действиях — блок «Решение»: что решили, "
        "кто что делает, какой итоговый результат. Если решение неполное, "
        "напиши, чего не хватает.\n"
        "6. Если в куске и спор, и решение — закрой оба блока.\n"
        "Не добавляй реплик, которых нет в стенограмме."
    ),
    ChatMessage.from_user(
        "Начало записи: {{ started_at }}\n"
        "Число реплик: {{ message_count }}\n"
        "\nСтенограмма:\n{{ transcript }}\n"
    ),
]


@component
class SortDocumentsByTime:
    """Ставит найденные реплики в хронологический порядок перед промптом."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Сортирует документы по полю meta.timestamp.

        :param documents: Документы, которые вернул ретривер.
        :returns: Те же документы, отсортированные по времени.
        """

        ordered = sorted(documents, key=lambda document: str((document.meta or {}).get("timestamp") or ""))
        return {"documents": ordered}


def _chat_filter(chat_id: int) -> dict[str, Any]:
    """Фильтр Pinecone: искать только в том чате, где задали вопрос.

    :param chat_id: Идентификатор чата Telegram.
    :returns: Фильтр в синтаксисе Haystack. Интеграция Pinecone сама убирает префикс meta.
    """

    return {"field": "meta.chat_id", "operator": "==", "value": str(chat_id)}


def _clip(text: str, limit: int) -> str:
    """Обрезает текст, если он не помещается в лимит метаданных Pinecone.

    :param text: Исходный текст.
    :param limit: Максимальная длина в символах.
    :returns: Исходный или укороченный текст с пометкой.
    """

    if len(text) <= limit:
        return text
    return text[: limit - 40].rstrip() + "\n…[текст обрезан при сохранении]"


def _reply_text(result: dict[str, Any]) -> str:
    """Достаёт текст первого ответа генератора.

    :param result: Словарь, который вернул Pipeline.run.
    :returns: Текст ответа модели.
    :raises RuntimeError: Если модель не вернула ни одной реплики.
    """

    replies = result.get("llm", {}).get("replies") or []
    if not replies:
        raise RuntimeError("Модель не вернула ответ")
    text = replies[0].text or ""
    if not text.strip():
        raise RuntimeError("Модель вернула пустой ответ")
    return text.strip()


class TeamMemory:
    """Три пайплайна: индексация, вопрос по памяти чата, итог фрагмента."""

    def __init__(self, settings: Settings) -> None:
        """Собирает хранилище Pinecone и пайплайны Haystack.

        :param settings: Настройки из .env.
        """

        self._settings = settings
        dimension = self._resolve_dimension(settings)
        self.document_store = PineconeDocumentStore(
            index=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            dimension=dimension,
            metric="cosine",
            show_progress=False,
        )
        self.index_pipeline = self._build_index_pipeline(settings, dimension)
        self.answer_pipeline = self._build_answer_pipeline(settings, dimension)
        self.summary_pipeline = self._build_summary_pipeline(settings)
        logger.info(
            "Пайплайны готовы: индекс=%s namespace=%s chat=%s embedding=%s dimension=%s",
            settings.pinecone_index_name,
            settings.pinecone_namespace,
            settings.chat_model,
            settings.embedding_model,
            dimension,
        )

    def index_message(self, document: Document) -> int:
        """Прогоняет одно сообщение через пайплайн эмбеддинга и записи.

        :param document: Реплика с текстом и метаданными автора.
        :returns: Сколько документов записал DocumentWriter.
        """

        if document.content:
            document = Document(
                id=document.id,
                content=_clip(document.content, _STORED_TEXT_LIMIT),
                meta=dict(document.meta or {}),
            )
        logger.info(
            "Индексация chat_id=%s message_id=%s author=%s",
            (document.meta or {}).get("chat_id"),
            (document.meta or {}).get("message_id"),
            (document.meta or {}).get("author_name"),
        )
        logger.debug("Текст на индексацию: %s", document.content)
        result = self.index_pipeline.run({"embedder": {"documents": [document]}})
        written = int(result["writer"]["documents_written"])
        logger.info("В Pinecone записано документов: %s", written)
        return written

    def answer(self, chat_id: int, question: str) -> str:
        """Ищет реплики этого чата и просит модель ответить со ссылкой на авторов.

        :param chat_id: Чат, в котором обратились к боту.
        :param question: Вопрос без упоминания бота.
        :returns: Ответ модели.
        """

        logger.info("Вопрос к памяти chat_id=%s длина=%s", chat_id, len(question))
        logger.debug("Вопрос: %s", question)
        result = self.answer_pipeline.run(
            {
                "text_embedder": {"text": question},
                "retriever": {"filters": _chat_filter(chat_id), "top_k": _TOP_K},
                "prompt_builder": {"question": question},
            }
        )
        found = result.get("sorter", {}).get("documents") or result.get("retriever", {}).get("documents") or []
        logger.info("Ретривер вернул документов: %s", len(found))
        text = _reply_text(result)
        logger.info("Ответ модели готов, символов: %s", len(text))
        return text

    def summarize(self, transcript: str, started_at: str, message_count: int) -> str:
        """Отправляет весь фрагмент диалога в модель и возвращает итог.

        :param transcript: Стенограмма между /listen и /stop.
        :param started_at: Время старта записи строкой.
        :param message_count: Сколько реплик попало в стенограмму.
        :returns: Резюме, мнение по спору и/или зафиксированное решение.
        """

        prepared = transcript
        if len(prepared) > _TRANSCRIPT_LIMIT:
            logger.warning(
                "Стенограмма длиннее %s символов, в модель уйдёт хвост",
                _TRANSCRIPT_LIMIT,
            )
            prepared = (
                "[Начало стенограммы обрезано из-за длины]\n"
                + prepared[-_TRANSCRIPT_LIMIT:]
            )
        logger.info("Итог диалога: реплик=%s символов=%s", message_count, len(prepared))
        result = self.summary_pipeline.run(
            {
                "prompt_builder": {
                    "transcript": prepared,
                    "started_at": started_at,
                    "message_count": message_count,
                }
            }
        )
        text = _reply_text(result)
        logger.info("Итог готов, символов: %s", len(text))
        return text

    def _resolve_dimension(self, settings: Settings) -> int:
        """Берёт размерность уже созданного индекса, иначе размерность модели.

        :param settings: Настройки из .env.
        :returns: Число координат вектора.
        :raises SystemExit: Если индекс короче или длиннее, а модель не умеет менять размерность.
        """

        client = Pinecone(api_key=settings.pinecone_api_key)
        names = client.list_indexes().names()
        if settings.pinecone_index_name not in names:
            logger.info(
                "Индекс %s ещё не создан, будет размерность %s",
                settings.pinecone_index_name,
                settings.embedding_dimensions,
            )
            return settings.embedding_dimensions

        description = client.describe_index(settings.pinecone_index_name)
        index_dimension = int(description.dimension)
        if index_dimension == settings.embedding_dimensions:
            return index_dimension
        if settings.embedding_model_accepts_dimensions:
            logger.info(
                "Индекс %s имеет размерность %s, эмбеддинги OpenAI будут запрошены той же длины",
                settings.pinecone_index_name,
                index_dimension,
            )
            return index_dimension
        raise SystemExit(
            f"Индекс {settings.pinecone_index_name} имеет размерность {index_dimension}, "
            f"а модель {settings.embedding_model} всегда отдаёт {settings.embedding_dimensions} "
            "и не принимает параметр dimensions."
        )

    def _build_index_pipeline(self, settings: Settings, dimension: int) -> Pipeline:
        """Пайплайн записи: документ → эмбеддинг OpenAI → DocumentWriter → Pinecone.

        :param settings: Настройки из .env.
        :param dimension: Длина вектора, совпадающая с индексом.
        :returns: Готовый indexing pipeline.
        """

        pipeline = Pipeline()
        pipeline.add_component(
            "embedder",
            OpenAIDocumentEmbedder(
                model=settings.embedding_model,
                dimensions=dimension if settings.embedding_model_accepts_dimensions else None,
                meta_fields_to_embed=["author_name"],
                progress_bar=False,
                raise_on_failure=True,
            ),
        )
        pipeline.add_component(
            "writer",
            DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE),
        )
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    def _build_answer_pipeline(self, settings: Settings, dimension: int) -> Pipeline:
        """RAG-пайплайн вопроса: эмбеддинг запроса, поиск, сортировка по времени, ответ.

        :param settings: Настройки из .env.
        :param dimension: Длина вектора, совпадающая с индексом.
        :returns: Готовый query pipeline.
        """

        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            OpenAITextEmbedder(
                model=settings.embedding_model,
                dimensions=dimension if settings.embedding_model_accepts_dimensions else None,
            ),
        )
        pipeline.add_component(
            "retriever",
            PineconeEmbeddingRetriever(document_store=self.document_store, top_k=_TOP_K),
        )
        pipeline.add_component("sorter", SortDocumentsByTime())
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(template=_ANSWER_TEMPLATE, required_variables=["documents", "question"]),
        )
        pipeline.add_component(
            "llm",
            OpenAIChatGenerator(
                model=settings.chat_model,
                generation_kwargs={"temperature": 0.4},
            ),
        )
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "sorter.documents")
        pipeline.connect("sorter.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    def _build_summary_pipeline(self, settings: Settings) -> Pipeline:
        """Пайплайн итога: вся стенограмма уходит в промпт и затем в OpenAI.

        :param settings: Настройки из .env.
        :returns: Готовый summary pipeline.
        """

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=_SUMMARY_TEMPLATE,
                required_variables=["transcript", "started_at", "message_count"],
            ),
        )
        pipeline.add_component(
            "llm",
            OpenAIChatGenerator(
                model=settings.chat_model,
                generation_kwargs={"temperature": 0.4},
            ),
        )
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline
