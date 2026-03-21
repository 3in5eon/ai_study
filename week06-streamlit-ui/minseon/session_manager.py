"""
세션 관리자 (SessionManager)

[역할]
  여러 대화 세션을 생성·전환·삭제하고, sessions.json 파일에 영속 저장합니다.

[저장 구조]
  sessions.json
  {
    "abc12345": {
      "id": "abc12345",
      "name": "청년 주거 질문",
      "created_at": "2026-03-19T10:30:00",
      "messages": [...],       ← Streamlit 화면 표시용 메시지
      "conversation": [...],   ← RAG 파이프라인 대화 히스토리 (LLM 문맥)
      "total_cost_usd": 0.00128,
      "total_tokens": {"input": 1200, "output": 400}
    },
    ...
  }

[세션 전환 시]
  1. 현재 세션 messages + conversation 저장
  2. 새 세션 messages + conversation 복원
  → rag.conversation을 세션마다 독립적으로 유지
"""

import json
import os
from datetime import datetime
from uuid import uuid4


class SessionManager:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self._sessions: dict[str, dict] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────

    def create(self, name: str | None = None) -> str:
        """새 세션 생성 → session_id 반환"""
        sid = uuid4().hex[:8]
        count = len(self._sessions) + 1
        self._sessions[sid] = {
            "id": sid,
            "name": name or f"대화 {count}",
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "conversation": [],
            "total_cost_usd": 0.0,
            "total_tokens": {"input": 0, "output": 0},
        }
        self._save()
        return sid

    def get(self, sid: str) -> dict | None:
        return self._sessions.get(sid)

    def list(self) -> list[dict]:
        """생성 시간 최신순 정렬"""
        return sorted(
            self._sessions.values(),
            key=lambda s: s["created_at"],
            reverse=True,
        )

    def rename(self, sid: str, name: str):
        if sid in self._sessions:
            self._sessions[sid]["name"] = name.strip() or self._sessions[sid]["name"]
            self._save()

    def delete(self, sid: str):
        self._sessions.pop(sid, None)
        self._save()

    # ── 세션 데이터 저장 ──────────────────────────────────────

    def save_messages(self, sid: str, messages: list, conversation: list):
        """현재 대화 내용을 세션에 저장"""
        if sid in self._sessions:
            self._sessions[sid]["messages"] = messages
            self._sessions[sid]["conversation"] = conversation
            self._save()

    def add_cost(self, sid: str, cost_usd: float, tokens: dict):
        """비용·토큰 누적"""
        if sid in self._sessions:
            self._sessions[sid]["total_cost_usd"] += cost_usd
            self._sessions[sid]["total_tokens"]["input"]  += tokens.get("input", 0)
            self._sessions[sid]["total_tokens"]["output"] += tokens.get("output", 0)
            self._save()

    # ── 대화 내보내기 ────────────────────────────────────────

    def export_markdown(self, sid: str) -> str:
        """세션 대화를 마크다운 문자열로 반환"""
        session = self.get(sid)
        if not session:
            return ""
        lines = [
            f"# {session['name']}",
            f"생성일: {session['created_at'][:10]}",
            f"총 비용: ${session['total_cost_usd']:.5f}",
            "",
        ]
        for msg in session["messages"]:
            role = "나" if msg["role"] == "user" else "AI"
            lines.append(f"**{role}:** {msg['content']}")
            lines.append("")
        return "\n".join(lines)

    # ── 영속 저장 ────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    self._sessions = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._sessions = {}
