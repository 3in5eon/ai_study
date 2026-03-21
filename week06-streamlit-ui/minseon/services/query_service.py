"""
 Multi-query Generation]
  하나의 질문을 여러 관점의 검색 쿼리로 확장
  → 각 쿼리로 검색 후 결과 병합 (Union)
  → 검색 커버리지 향상, 다양한 문서 발견 가능

예시)
  입력: "청년도약계좌 혜택이 뭐야?"
  출력:
    - "청년도약계좌 지원 금액 정부 기여금"
    - "청년도약계좌 가입 자격 조건 나이"
    - "청년도약계좌 납입 기간 만기 혜택"
"""

import json
import time

from openai import OpenAI
from services.llm_service import CHAT_MODEL


def generate_queries(
    user_message: str,
    conversation: list[dict],
    tracker=None,
) -> list[str]:
    """
    Multi-query Generation: 원본 질문을 다양한 관점의 검색 쿼리 3개로 확장

    Args:
        user_message: 사용자 질문
        conversation: 대화 히스토리 (맥락 파악용)
        tracker:      CostTracker 인스턴스 (None이면 추적 안 함)

    Returns:
        검색 쿼리 리스트 (3개, 실패 시 원본 질문 1개)
    """
    recent = conversation[-4:] if conversation else []
    history_text = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content'][:150]}"
        for m in recent
    ) if recent else "없음"

    prompt = (
        f"이전 대화:\n{history_text}\n\n"
        f"현재 질문: {user_message}\n\n"
        f"위 맥락을 고려해서, 청년 정책 문서 검색에 최적화된 서로 다른 관점의 검색 쿼리 3개를 생성하세요.\n\n"
        f"각 쿼리는:\n"
        f"1. 질문의 다른 측면을 커버해야 함 (자격조건 / 지원내용 / 신청방법 등)\n"
        f"2. 단독으로 검색해도 의미 있는 완전한 문장이어야 함\n"
        f"3. 핵심 키워드가 명확히 포함되어야 함\n\n"
        f"JSON 배열로만 반환하세요. 예: [\"쿼리1\", \"쿼리2\", \"쿼리3\"]"
    )

    try:
        client = OpenAI()
        t0 = time.time()
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        elapsed = time.time() - t0

        if tracker is not None:
            usage = response.usage
            tracker.record("pre", CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, elapsed)

        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [q for q in parsed[:3] if isinstance(q, str) and q.strip()]

    except Exception:
        pass

    return [user_message]
