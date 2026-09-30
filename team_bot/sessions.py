"""Сессии записи: что бот копит в чате между /listen и /stop."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ListeningSession:
    """Один фрагмент обсуждения в конкретном чате.

    :param chat_id: Идентификатор чата Telegram.
    :param session_id: Идентификатор фрагмента, общий для всех его реплик в Pinecone.
    :param started_at: Момент команды старта, UTC.
    :param lines: Реплики в порядке появления, уже с автором и временем.
    """

    chat_id: int
    session_id: str
    started_at: datetime
    lines: list[str] = field(default_factory=list)

    def transcript(self) -> str:
        """Собирает стенограмму, которую целиком уходит в модель на /stop.

        :returns: Реплики, разделённые переводом строки.
        """

        return "\n".join(self.lines)


class SessionRegistry:
    """Хранит не больше одной активной записи на чат."""

    def __init__(self) -> None:
        """Создаёт пустой реестр сессий."""

        self._sessions: dict[int, ListeningSession] = {}
        self._lock = threading.Lock()

    def start(self, chat_id: int) -> tuple[ListeningSession, bool]:
        """Открывает запись в чате.

        :param chat_id: Чат, в котором пришла команда старта.
        :returns: Сессия и флаг: True, если запись только что включили.
        """

        with self._lock:
            current = self._sessions.get(chat_id)
            if current is not None:
                return current, False
            session = ListeningSession(
                chat_id=chat_id,
                session_id=uuid.uuid4().hex,
                started_at=datetime.now(timezone.utc),
            )
            self._sessions[chat_id] = session
            return session, True

    def get(self, chat_id: int) -> ListeningSession | None:
        """Возвращает активную сессию чата, если запись включена.

        :param chat_id: Чат Telegram.
        :returns: Сессия или None.
        """

        with self._lock:
            return self._sessions.get(chat_id)

    def add_line(self, chat_id: int, line: str) -> bool:
        """Добавляет реплику в открытую сессию.

        :param chat_id: Чат Telegram.
        :param line: Уже оформленная строка стенограммы.
        :returns: True, если запись была включена и строка сохранена.
        """

        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return False
            session.lines.append(line)
            return True

    def stop(self, chat_id: int) -> ListeningSession | None:
        """Закрывает запись и отдаёт накопленный фрагмент.

        :param chat_id: Чат Telegram.
        :returns: Закрытая сессия или None, если бот в этом чате не слушал.
        """

        with self._lock:
            return self._sessions.pop(chat_id, None)

    def restore(self, session: ListeningSession) -> None:
        """Возвращает сессию, если итог не удалось собрать и запись надо повторить.

        :param session: Фрагмент, который только что сняли с записи.
        """

        with self._lock:
            if session.chat_id not in self._sessions:
                self._sessions[session.chat_id] = session
