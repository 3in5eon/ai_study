"""
임베딩 서비스 - 텍스트를 벡터로 변환
"""

import time

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(
    texts: list[str],
    tracker=None,
    stage: str = "embedding",
) -> list[list[float]]:
    """
    OpenAI 임베딩 API로 텍스트 리스트를 벡터로 일괄 변환

    Args:
        texts:   임베딩할 텍스트 리스트
        tracker: CostTracker 인스턴스 (None이면 추적 안 함)
        stage:   비용 추적 시 사용할 단계 이름
    """
    client = OpenAI()
    t0 = time.time()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    elapsed = time.time() - t0

    if tracker is not None:
        tracker.record(stage, EMBEDDING_MODEL, response.usage.total_tokens, 0, elapsed)

    return [item.embedding for item in response.data]
