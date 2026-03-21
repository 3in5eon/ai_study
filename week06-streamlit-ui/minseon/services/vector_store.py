"""
벡터 스토어 - ChromaDB + BM25 하이브리드 검색

[Naive RAG의 벡터 검색 한계]
  - 표면적 유사도만 측정: 질문 키워드가 많은 문서가 올라오지만 내용이 없을 수 있음
  - 짧은 문장 편향: 압축된 임베딩으로인해 짧은 문장이  유리
  - 도메인 용어 취약: 임베딩 모델이 학습하지 못한 전문 용어에 부정확

[해결: Hybrid Search]
  Vector Search: 의미론적 유사도 (코사인 유사도) → 문맥·의도 파악
  BM25 Search:   키워드 기반 정확도 (TF-IDF 계열) → 정확한 키워드 매칭
  RRF Fusion:    Reciprocal Rank Fusion으로 두 결과를 최적 병합

[RRF (Reciprocal Rank Fusion)]
  score(d) = Σ 1 / (k + rank(d))  (k=60, 표준값)
  → 두 검색에서 모두 상위에 오른 문서가 최종 상위에 위치
  → 한 검색에서만 상위여도 낮은 점수로 포함 가능
"""

import re
import uuid

import chromadb
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """한국어+영어 토크나이저: 단어 단위 분리"""
    return re.findall(r'\w+', text.lower())


class VectorStore:
    """
    ChromaDB(벡터 검색) + BM25(키워드 검색) 하이브리드 벡터 스토어

    문서 추가/삭제 시 BM25 인덱스를 자동 재구축.
    """

    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 인덱스 (in-memory, ChromaDB에서 재구축)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []  # [{id, content, metadata}]
        self._rebuild_bm25()

    # ── BM25 인덱스 관리 ──────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """ChromaDB의 모든 문서로 BM25 인덱스 재구축"""
        results = self.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        ids = results.get("ids") or []

        if not docs:
            self._bm25 = None
            self._bm25_docs = []
            return

        self._bm25_docs = [
            {"id": id_, "content": doc, "metadata": meta}
            for id_, doc, meta in zip(ids, docs, metas)
        ]
        tokenized = [_tokenize(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)

    # ── 데이터 추가/삭제 ──────────────────────────────────────

    def add(self, chunks: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        """청크 + 벡터 + 메타데이터를 ChromaDB에 추가하고 BM25 재구축"""
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=metadatas,
        )
        self._rebuild_bm25()

    def remove_source(self, source: str) -> None:
        """특정 소스의 모든 청크 제거 후 BM25 재구축"""
        results = self.collection.get(where={"source": source})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        self._rebuild_bm25()

    # ── 하이브리드 검색 (핵심 기능) ────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.2,
        max_per_source: int = 2,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> list[dict]:
        """
        BM25 + Vector Search → RRF Fusion 하이브리드 검색

        Args:
            query:          원본 질문 문자열 (BM25 검색용)
            query_vector:   질문 임베딩 벡터 (벡터 검색용)
            top_k:          최종 반환할 청크 수
            threshold:      벡터 유사도 최소 기준 (이 미만은 필터링)
            max_per_source: 동일 문서에서 최대 가져올 청크 수
            bm25_weight:    BM25 RRF 점수 가중치
            vector_weight:  벡터 RRF 점수 가중치
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(total, top_k * 5)

        # ── Step 1: 벡터 검색 ─────────────────────────────────
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # key = "출처_청크인덱스" (중복 병합용)
        all_hits: dict[str, dict] = {}
        vector_ranks: dict[str, int] = {}

        for rank, (doc, meta, dist) in enumerate(zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        )):
            similarity = 1.0 - dist
            if similarity < threshold:
                continue
            key = f"{meta['source']}_{meta.get('chunk_index', 0)}"
            all_hits[key] = {"content": doc, "similarity": similarity, "metadata": meta}
            vector_ranks[key] = rank

        # ── Step 2: BM25 검색 ─────────────────────────────────
        bm25_ranks: dict[str, int] = {}
        if self._bm25 and self._bm25_docs:
            query_tokens = _tokenize(query)
            bm25_scores = self._bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )

            for rank, idx in enumerate(ranked_indices[:n_results]):
                if bm25_scores[idx] <= 0:
                    break
                doc_info = self._bm25_docs[idx]
                meta = doc_info["metadata"]
                key = f"{meta['source']}_{meta.get('chunk_index', 0)}"
                bm25_ranks[key] = rank

                # 벡터 검색에서 누락된 청크 추가 (BM25 전용 히트)
                if key not in all_hits:
                    all_hits[key] = {
                        "content": doc_info["content"],
                        "similarity": 0.0,
                        "metadata": meta,
                    }

        # ── Step 3: RRF Fusion ────────────────────────────────
        # score = vector_weight * 1/(k+v_rank) + bm25_weight * 1/(k+b_rank)
        k = 60  # RRF 표준 상수
        rrf_scores: dict[str, float] = {}
        for key in all_hits:
            v_rank = vector_ranks.get(key, n_results)
            b_rank = bm25_ranks.get(key, n_results)
            rrf_scores[key] = (
                vector_weight * (1 / (k + v_rank)) +
                bm25_weight * (1 / (k + b_rank))
            )

        # ── Step 4: 소스 다양성 보장 후 top_k 반환 ────────────
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        source_counts: dict[str, int] = {}
        final = []
        for key in sorted_keys:
            hit = all_hits[key]
            source = hit["metadata"]["source"]
            count = source_counts.get(source, 0)
            if count < max_per_source:
                final.append(hit)
                source_counts[source] = count + 1
            if len(final) >= top_k:
                break

        return final

    # ── 단순 벡터 검색 (하위 호환) ────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.2,
        max_per_source: int = 2,
    ) -> list[dict]:
        """벡터 전용 검색 (하위 호환용)"""
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(total, top_k * max_per_source * 3)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        scored = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= threshold:
                scored.append({"content": doc, "similarity": similarity, "metadata": meta})

        source_counts: dict[str, int] = {}
        final = []
        for item in scored:
            source = item["metadata"]["source"]
            count = source_counts.get(source, 0)
            if count < max_per_source:
                final.append(item)
                source_counts[source] = count + 1
            if len(final) >= top_k:
                break

        return final

    # ── 조회 ──────────────────────────────────────────────────

    def get_sources(self) -> list[dict]:
        """인덱싱된 소스 목록과 청크 수 반환"""
        results = self.collection.get(include=["metadatas"])
        source_counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta["source"]
            source_counts[src] = source_counts.get(src, 0) + 1
        return [{"source": src, "chunks": cnt} for src, cnt in source_counts.items()]

    def total_chunks(self) -> int:
        return self.collection.count()
