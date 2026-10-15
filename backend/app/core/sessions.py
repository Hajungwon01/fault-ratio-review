"""질문 단계의 세션 보관 — 서버 메모리 + 만료(TTL).

사용자에게 나가면 안 되는 내부 결과(FaultReport: 보기별 도표·과실 등)를 서버가 쥐고 있고, 사용자는 session_id 만 가진다.
프로세스 하나에서만 동작하고 서버를 재시작하면 세션이 사라진다 (학생 프로젝트 1차 범위; 후속은 Redis/DB — docs/USER_API_DESIGN.md §5).
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable

from app.models.schemas import ClarifyingQuestion, FaultReport

DEFAULT_TTL_SEC = 1800
DEFAULT_MAX_SESSIONS = 200


class SessionNotFound(Exception):
    """없거나 만료된 세션 (둘을 구분해 알려 주지 않는다)."""


class AlreadyAnswered(Exception):
    pass


class TooManySessions(Exception):
    pass


@dataclass
class Session:
    report: FaultReport
    question: ClarifyingQuestion
    seed: int
    expires_at: float
    answered: bool = False


class SessionStore:
    def __init__(self, ttl_sec: int = DEFAULT_TTL_SEC, max_sessions: int = DEFAULT_MAX_SESSIONS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_sec = ttl_sec
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    def _purge_expired(self) -> None:
        now = self._clock()
        for sid in [k for k, s in self._sessions.items() if s.expires_at <= now]:
            del self._sessions[sid]

    def create(self, report: FaultReport, question: ClarifyingQuestion, seed: int) -> str:
        self._purge_expired()
        if len(self._sessions) >= self.max_sessions:
            raise TooManySessions()
        sid = secrets.token_urlsafe(24)  # 추측할 수 없는 무작위 토큰
        self._sessions[sid] = Session(report, question, seed, self._clock() + self.ttl_sec)
        return sid

    def get(self, session_id: str) -> Session:
        """열려 있는 세션. 없거나 만료면 SessionNotFound, 이미 답했으면 AlreadyAnswered."""
        self._purge_expired()
        s = self._sessions.get(session_id)
        if s is None:
            raise SessionNotFound()
        if s.answered:
            raise AlreadyAnswered()
        return s

    def mark_answered(self, session_id: str) -> None:
        """답변은 세션당 1회 — 보기를 바꿔 가며 결과를 비교하지 못하게 세션을 닫는다. 만료 때까지 '답함' 표시만 남긴다."""
        s = self._sessions[session_id]
        s.answered = True
        s.report = None  # type: ignore[assignment]  # 내부 결과는 더 보관하지 않는다

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._sessions)
