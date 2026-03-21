"""
재순위화 서비스 - GPT 기반 검색 결과 재순위화 (Post-retrieval)

[Advanced RAG에서의 역할]
  벡터 검색인의 한계:
    - Bi-Encoder는 질문/문서를 따로 인코딩 → 표면적 유사도만 계산
    - 짧은 질문, 동음이의어, 도메 용어에 취약

  GPT Re-ranking의 장점:
    - 질문과 문서를 함께 분석 (Cross-Encoder 역할)
    - 실제 "얼마나 도움이 되는가"를 평가
    - 유사도 순위와 다른 최적 순위 발견 가능

[동작 흐름]
  초기 검색 (Hybrid Search, top_k*3 후보)
    → GPT가 각 청크를 0~10 점수로 평가
    → 점수 기준 재정렬 → 최종 top_k 반환
"""

import json
import time

from openai import OpenAI

RERANK_MODEL = "gpt-4o-mini"


def rerank(
    query: str,
    hits: list[dict],
    top_k: int,
    tracker=None,
) -> list[dict]:
    """
    GPT를 사용해 검색 결과를 재순위화

    Args:
        query:   사용자 원본 질문
        hits:    초기 검색 결과 목록
        top_k:   최종 반환할 개수
        tracker: CostTracker 인스턴스 (None이면 추적 안 함)

    Returns:
        GPT 관련성 점수 기준으로 재정렬된 상위 top_k 결과
        실패 시 원래 유사도 순서대로 top_k 반환 (fallback)
    """
    if len(hits) <= top_k:
        return hits

    candidates = [
        {"id": i, "content": h["content"][:400]}
        for i, h in enumerate(hits)
    ]

    prompt = (
        f"다음 질문에 대해 각 문서 청크의 관련성을 0~10 점수로 평가하세요.\n\n"
        f"질문: {query}\n\n"
        f"평가 기준:\n"
        f"- 10: 질문에 직접 답하는 핵심 내용\n"
        f"- 7~9: 관련성 높고 유용한 정보\n"
        f"- 4~6: 부분적으로 관련 있음\n"
        f"- 1~3: 약하게 관련됨\n"
        f"- 0: 전혀 관련 없음\n\n"
        f"문서 청크 목록 (JSON):\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"각 청크의 id와 score를 JSON 배열로만 반환하세요. "
        f'예시: [{{"id": 0, "score": 8}}, {{"id": 1, "score": 3}}]'
    )

    try:
        client = OpenAI()
        t0 = time.time()
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        elapsed = time.time() - t0

        if tracker is not None:
            usage = response.usage
            tracker.record("reranking", RERANK_MODEL, usage.prompt_tokens, usage.completion_tokens, elapsed)

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            scores_list = next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
        else:
            scores_list = parsed

        score_map = {
            item["id"]: item["score"]
            for item in scores_list
            if isinstance(item, dict) and "id" in item and "score" in item
        }

        ranked = sorted(range(len(hits)), key=lambda i: score_map.get(i, 0), reverse=True)
        return [hits[i] for i in ranked[:top_k]]

    except Exception:
        return hits[:top_k]
