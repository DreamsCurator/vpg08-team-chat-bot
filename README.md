# Память командного чата в Telegram

Бот на PyTelegramBotAPI сидит в общем рабочем чате. По команде он записывает реплики вместе с автором и временем в Pinecone, а когда запись останавливают — отправляет весь фрагмент в OpenAI и присылает итог. Если это был спор, в итоге есть мнение бота. Если команда договорилась о действиях — зафиксированы решение и результат.

Отдельный вопрос с упоминанием бота ищет по всей сохранённой памяти этого чата и отвечает, кто и когда что сказал.

Эмбеддинги и генерация идут в API OpenAI напрямую через компоненты Haystack: `OpenAIDocumentEmbedder`, `OpenAITextEmbedder` и `OpenAIChatGenerator`.

## Команды

| Команда | Действие |
| --- | --- |
| `/start`, `/help` | Как пользоваться ботом |
| `/listen`, `/start_listen` | Начать запись сообщений |
| `/stop`, `/stop_listen` | Остановить запись, сохранить фрагмент и прислать итог |
| `@бот` или ответ на его сообщение | Вопрос по всей памяти этого чата |

Пока запись включена, каждое текстовое сообщение и подпись к вложению проходят пайплайн индексации. Команды в стенограмму не попадают.

## Как устроена память

Индекс Pinecone подключён через `PineconeDocumentStore`, метрика `cosine`. Модель эмбеддингов по умолчанию — `text-embedding-3-small` (1536 измерений). Записи лежат в namespace `company-chat`. Поиск фильтруется по `chat_id`, чаты не смешиваются.

Три пайплайна Haystack:

- индексация: `OpenAIDocumentEmbedder` → `DocumentWriter`;
- вопрос: `OpenAITextEmbedder` → `PineconeEmbeddingRetriever` → сортировка по времени → `ChatPromptBuilder` → `OpenAIChatGenerator`;
- итог: `ChatPromptBuilder` → `OpenAIChatGenerator` по полной стенограмме фрагмента между `/listen` и `/stop`.

У каждого документа в метаданных есть автор, username, id сообщения, время, id сессии и тип (`message` или `summary`). Итог после `/stop` тоже записывается в индекс.

## Файлы

| Файл | Назначение |
| --- | --- |
| `main.py` | Точка входа: настройки, пайплайны, long polling |
| `team_bot/config.py` | Чтение `.env` |
| `team_bot/memory.py` | Пайплайны Haystack и Pinecone |
| `team_bot/sessions.py` | Фрагмент чата между стартом и стопом записи |
| `team_bot/telegram_app.py` | Команды Telegram и очередь сообщений |
| `team_bot/logging_setup.py` | Логи в консоль |
| `example.py` | Учебный ноутбук Haystack, по схеме которого собран RAG |
| `requirements.txt` | Зависимости |

## Запуск

Нужны Python 3.11+, ключи OpenAI, Pinecone и Telegram. Индекс из `PINECONE_INDEX_NAME` может уже существовать: размерность должна совпадать с моделью эмбеддингов. Для `text-embedding-3-small` это 1536 и метрика `cosine`. Если индекса нет, хранилище Haystack создаёт serverless-индекс в `us-east-1`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Файл `.env` в корне проекта:

```
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=
PINECONE_INDEX_NAME=nemo
TELEGRAM_BOT_TOKEN=
```

```bash
python main.py
```

Чтобы бот видел обычные сообщения группы, а не только команды и упоминания, в @BotFather отключите Group Privacy (Bot Settings → Group Privacy → Turn off) и заново добавьте бота в чат.
