"""
컨텍스트 압축 서비스 - Post-retrieval 최적화

[문제: 검색된 청크의 노이즈]
  - 청크 안에는 질문과 무관한 내용도 포함됨
  - 불필요한 내용이 LLM 컨텍스트 창을 낭비
  - 노이즈가 많으면 LLM이 핵심 정보를 놓칠 수 있음 (Lost-in-the-Middle 문제)

[해결: Context Compression]
  각 청크에서 질문과 직접 관련된 부분만 추출
  → 컨텍스트 크기 감소 → 토큰 절약 + 답변 품질 향상
  → 완전히 무관한 청크는 제거 (노이즈 필터링)

[trade-off]
  장점: 정밀한 컨텍스트, 토큰 효율
  단점: 추가 LLM 호출 비용 발생
  → 긴 청크(200자 이상)에만 적용해 비용 최소화
"""

import time

from openai import OpenAI
from services.llm_service import CHAT_MODEL


def compress_context(
    query: str,
    hits: list[dict],
    tracker=None,
) -> list[dict]:
    """
    각 청크에서 질문과 관련된 핵심 내용만 추출

    Args:
        query:   사용자 질문
        hits:    검색된 청크 목록
        tracker: CostTracker 인스턴스 (None이면 추적 안 함)

    Returns:
        content가 압축된 hits 목록
        - 관련 없는 청크: 제거
        - 짧은 청크(200자 미만): 원본 유지
        - 압축 실패 시: 원본 유지 (fallback)
    """
    if not hits:
        return hits

    client = OpenAI()
    compressed = []

    for hit in hits:
        content = hit["content"]

        # 짧은 청크는 압축 불필요 (비용 절감)
        if len(content) < 200:
            compressed.append(hit)
            continue

        prompt = (
            f"질문: {query}\n\n"
            f"문서 내용:\n{content}\n\n"
            f"위 문서에서 질문에 답하는 데 직접 필요한 핵심 내용만 추출하세요.\n"
            f"규칙:\n"
            f"- 금액·나이·기간·자격 조건·신청 방법 등 구체적 수치는 반드시 포함\n"
            f"- 질문과 무관한 배경 설명은 제거\n"
            f"- 원문 표현을 최대한 유지 (재작성 금지)\n"
            f"- 관련 내용이 전혀 없으면 '없음'만 반환"
        )

        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            elapsed = time.time() - t0

            if tracker is not None:
                usage = response.usage
                tracker.record("compression", CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, elapsed)

            extracted = response.choices[0].message.content.strip()

            if extracted == "없음" or len(extracted) < 20:
                continue  # 완전히 무관한 청크 제거

            compressed.append({**hit, "content": extracted, "compressed": True})

        except Exception:
            compressed.append(hit)  # API 실패 시 원본 유지

    # 모든 청크가 제거됐을 경우 원본 반환 (안전장치)
    return compressed if compressed else hits
