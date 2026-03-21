"""
쿼리 분류기 (Query Classifier)

[역할]
  질문을 분석해 싱글홉 / 멀티홉 중 어느 검색 전략이 적합한지 판단합니다.

[싱글홉 (single)]
  하나의 정책·개념에 대한 직접 질문
  → 검색 1번으로 충분
  예) "청년도약계좌 가입 조건이 뭐야?"
      "복지로 신청 방법 알려줘"
      "청년 주거 지원 금액이 얼마야?"

[멀티홉 (multi)]
  여러 정책 비교 / 조건부 추론 / 연결된 정보 필요
  → 검색 여러 번, 이전 결과가 다음 검색에 영향
  예) "청년도약계좌랑 청년희망적금 중에 나한테 유리한 건?"
      "소득 200만원이면 동시에 받을 수 있는 지원이 뭐야?"
      "주거 지원 받으면 취업 지원도 신청 가능해?"

[비용]
  분류 API 1번 추가
  → 단순 질문(80%)은 멀티쿼리 생략 → 전체 평균 비용 절감
"""

import time

from openai import OpenAI
from services.llm_service import CHAT_MODEL


_CLASSIFY_PROMPT = """청년 정책 관련 질문을 분석해서 검색 전략을 결정하세요.

[single] 하나의 정책/개념에 대한 직접 질문
  - 단일 정책 조건, 금액, 기간, 신청 방법
  예) "청년도약계좌 조건이 뭐야?", "복지로 신청 방법", "주거 지원 금액"

[multi] 여러 정보를 연결해야 답할 수 있는 질문
  - 두 개 이상 정책 비교
  - 여러 조건을 동시에 확인해야 하는 경우
  - "동시에", "같이", "둘 다", "비교", "차이", "유리한", "중에" 키워드
  예) "A랑 B 중 뭐가 나아?", "소득 200만원이면 뭘 동시에 받을 수 있어?",
      "주거 지원 받으면 취업 지원도 되나?"

질문: {question}

single 또는 multi 중 하나만 출력하세요."""


def classify_query(
    user_message: str,
    tracker=None,
) -> str:
    """
    질문 유형 분류

    Returns:
        "single" 또는 "multi"
        (API 실패 시 "multi" 반환 — 안전한 방향으로 폴백)
    """
    try:
        client = OpenAI()
        t0 = time.time()
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": _CLASSIFY_PROMPT.format(question=user_message),
            }],
            temperature=0,
            max_tokens=10,
        )
        elapsed = time.time() - t0

        if tracker is not None:
            usage = response.usage
            tracker.record("classify", CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, elapsed)

        result = response.choices[0].message.content.strip().lower()
        return "single" if "single" in result else "multi"

    except Exception:
        return "multi"  # 실패 시 멀티홉으로 폴백 (더 안전)
