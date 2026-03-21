"""
5주차: Advanced RAG 파이프라인 오케스트레이터

[Naive RAG vs Advanced RAG]

  Naive RAG:
    질문 → 임베딩 → 벡터 검색 → LLM → 답변
    문제: 모호한 질문, 벡터 검색 순위 부정확, 노이즈 컨텍스트

  Advanced RAG (이 파이프라인):
    ┌─ Pre-retrieval ──────────────────────────────────────────┐
    │  Multi-query Generation: 질문 → 3개의 검색 쿼리로 확장   │
    └──────────────────────────────────────────────────────────┘
    ┌─ Retrieval ──────────────────────────────────────────────┐
    │  Hybrid Search (BM25 + Vector) per query + RRF Fusion    │
    └──────────────────────────────────────────────────────────┘
    ┌─ Post-retrieval ─────────────────────────────────────────┐
    │  Re-ranking + Context Compression                        │
    └──────────────────────────────────────────────────────────┘
    → LLM → 답변

[비용·시간 추적]
  CostTracker가 각 API 호출을 기록
  → _last_cost_summary에 단계별 비용/시간 집계 결과 저장
  → UI에서 파이프라인 비용 분석 가능
"""

import os

from dotenv import load_dotenv

from services.document_service import load_document
from services.chunking_service import split_text
from services.embedding_service import embed_texts, EMBEDDING_MODEL
from services.vector_store import VectorStore
from services.llm_service import stream_response, CHAT_MODEL
from services.query_service import generate_queries
from services.query_classifier import classify_query
from services.reranker_service import rerank
from services.compression_service import compress_context
from services.cost_tracker import CostTracker


load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────
DB_PATH  = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── 시스템 프롬프트 ────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """당신은 청년 정책 전문 AI 상담사 '청년도우미'입니다.
청년을 위한 정부 지원 정책(주거·취업·금융·교육·복지)을 아래 참고 문서에 기반해 정확하고 친절하게 안내합니다.

## 답변 방식
- 사용자의 상황을 파악해 관련 정책을 먼저 제시하고, 자격 조건 → 지원 내용 → 신청 방법 순으로 구체적으로 설명하세요.
- 참고 문서에 관련 정책이 여러 개라면 모두 나열해 비교해주세요.
- 이전 대화 내용을 반드시 참고해 자연스럽게 이어서 답변하세요.

## 답변 규칙 (반드시 준수)
1. 반드시 아래 [참고 문서]에 있는 내용만 근거로 사용하세요. 추측하지 마세요.
2. 금액·나이·기간·소득 기준 등 수치는 문서에 명시된 그대로만 전달하세요. 임의로 계산하거나 변형하지 마세요.
3. 문서에 없는 내용은 반드시 "해당 정보는 보유 문서에서 확인되지 않습니다. 공식 기관에 문의하세요."라고 답하세요.
4. 각 정보 뒤에 반드시 **[출처: 파일명]** 형식으로 출처를 표시하세요.
5. 신청 방법 안내 시 반드시 공식 신청 링크를 [신청하기](URL) 형식으로 포함하세요.
   주요 신청 포털:
   - 복지로(복지 전반): https://www.bokjiro.go.kr
   - 청년포털(청년 정책 통합): https://www.youthcenter.go.kr
   - 마이홈(주거 지원): https://www.myhome.go.kr
   - 워크넷(취업 지원): https://www.work.go.kr
   - 한국장학재단(국가장학금): https://www.kosaf.go.kr
   - 청년도약계좌·청년희망적금: https://ylaccount.kinfa.or.kr
6. 여러 정책이 해당될 경우 표(markdown table)로 비교해 제시하세요.
7. 답변 마지막에 사용자가 추가로 확인할 만한 관련 질문을 1~2개 제안하세요.
8. 청년정책과 관련없는 내용을 물어봤을땐 관련없는 내용입니다라고 대답하세요. 

## 참고 문서
{context}
"""

SYSTEM_PROMPT_NO_DOC = """당신은 청년 정책 전문 AI 상담사 '청년도우미'입니다.

현재 인덱싱된 문서에서 관련 내용을 찾지 못했습니다.
임의로 정보를 만들어내지 말고, 아래와 같이 안내하세요:
- 어떤 공식 기관에서 확인할 수 있는지 알려주세요.
- 관련 공식 포털 링크를 [바로가기](URL) 형식으로 제공하세요.
- 한국어로 답변합니다."""


class AdvancedRagPipeline:
    """
    Advanced RAG 파이프라인

    Pre-retrieval → Retrieval → Post-retrieval 3단계.
    CostTracker로 각 단계의 API 비용·시간을 자동 측정.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.store = VectorStore(db_path)
        self.conversation: list[dict] = []

        # UI 시각화용 중간 결과 저장
        self._last_query_type: str = "single"
        self._last_queries: list[str] = []
        self._last_candidates: list[dict] = []
        self._last_hits: list[dict] = []
        self._last_compressed: list[dict] = []
        self._last_usage: dict = {"input": 0, "output": 0}
        self._last_cost_summary: dict = {}

    # ── 인덱싱 ────────────────────────────────────────────────

    def auto_index_data_dir(self) -> list[dict]:
        if not os.path.exists(DATA_DIR):
            return []
        indexed_sources = {s["source"] for s in self.store.get_sources()}
        results = []
        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.endswith((".md", ".txt")):
                continue
            if fname in indexed_sources:
                continue
            fpath = os.path.join(DATA_DIR, fname)
            result = self.index_document(fpath, source_name=fname)
            results.append(result)
        return results

    def index_document(self, file_path: str, source_name: str | None = None) -> dict:
        source = source_name or os.path.basename(file_path)
        self.store.remove_source(source)
        text = load_document(file_path)
        chunks = [c for c in split_text(text) if c.strip()]
        if not chunks:
            return {"source": source, "chunks": 0, "chars": len(text)}
        vectors = embed_texts(chunks)  # 인덱싱 시에는 비용 추적 생략
        metadatas = [
            {"source": source, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        self.store.add(chunks, vectors, metadatas)
        return {"source": source, "chunks": len(chunks), "chars": len(text)}

    # ── Pre-retrieval ─────────────────────────────────────────

    def _generate_queries(self, user_message: str, tracker: CostTracker) -> list[str]:
        """
        [Pre-retrieval] 질문 유형 분류 후 검색 전략 결정

        single → 원본 질문 1개만 사용 (API 절약)
        multi  → Multi-query Generation (3개 쿼리로 확장)
        """
        query_type = classify_query(user_message, tracker=tracker)
        self._last_query_type = query_type

        if query_type == "single":
            return [user_message]
        return generate_queries(user_message, self.conversation, tracker=tracker)

    # ── Retrieval ─────────────────────────────────────────────

    def _hybrid_search_all(
        self,
        queries: list[str],
        top_k: int,
        threshold: float,
        max_per_source: int,
        tracker: CostTracker,
    ) -> list[dict]:
        """[Retrieval] 각 쿼리에 대해 Hybrid Search 후 결과 병합"""
        seen: dict[str, dict] = {}

        for query in queries:
            query_vector = embed_texts([query], tracker=tracker, stage="embedding")
            hits = self.store.hybrid_search(
                query=query,
                query_vector=query_vector[0],
                top_k=top_k * 2,
                threshold=threshold,
                max_per_source=max_per_source + 1,
            )
            for hit in hits:
                meta = hit["metadata"]
                key = f"{meta['source']}_{meta.get('chunk_index', 0)}"
                if key not in seen or hit["similarity"] > seen[key]["similarity"]:
                    seen[key] = hit

        all_hits = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

        source_counts: dict[str, int] = {}
        final = []
        for hit in all_hits:
            source = hit["metadata"]["source"]
            count = source_counts.get(source, 0)
            if count < max_per_source:
                final.append(hit)
                source_counts[source] = count + 1
            if len(final) >= top_k * 3:
                break

        return final

    # ── Post-retrieval ────────────────────────────────────────

    def _post_process(
        self,
        user_message: str,
        candidates: list[dict],
        top_k: int,
        use_compression: bool,
        tracker: CostTracker,
    ) -> list[dict]:
        """[Post-retrieval] Re-ranking → Context Compression"""
        reranked = rerank(user_message, candidates, top_k, tracker=tracker)
        self._last_hits = reranked

        if use_compression:
            compressed = compress_context(user_message, reranked, tracker=tracker)
            self._last_compressed = compressed
            return compressed

        self._last_compressed = reranked
        return reranked

    # ── 프롬프트 구성 ─────────────────────────────────────────

    def _build_context(self, hits: list[dict]) -> str:
        parts = []
        for i, hit in enumerate(hits):
            source = hit["metadata"]["source"]
            sim = hit["similarity"]
            compressed_marker = " [압축됨]" if hit.get("compressed") else ""
            parts.append(
                f"[문서 {i+1}] 출처: {source} (유사도: {sim:.1%}){compressed_marker}\n"
                f"{hit['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_system_prompt(self, hits: list[dict]) -> str:
        if not hits:
            return SYSTEM_PROMPT_NO_DOC
        return SYSTEM_PROMPT_TEMPLATE.format(context=self._build_context(hits))

    # ── 메인: 스트리밍 응답 ────────────────────────────────────

    def chat_stream(
        self,
        user_message: str,
        top_k: int = 5,
        threshold: float = 0.2,
        max_per_source: int = 2,
        use_compression: bool = True,
    ):
        """
        Advanced RAG 전체 파이프라인 (스트리밍)

        각 단계의 비용·시간은 _last_cost_summary에 저장.

        Yields:
            str: LLM 응답 텍스트 청크
        """
        tracker = CostTracker()

        # ── Pre-retrieval ──────────────────────────────────────
        tracker.start_stage("pre")
        queries = self._generate_queries(user_message, tracker)
        tracker.end_stage("pre")
        self._last_queries = queries

        # ── Retrieval ──────────────────────────────────────────
        tracker.start_stage("retrieval")
        candidates = self._hybrid_search_all(queries, top_k, threshold, max_per_source, tracker)
        tracker.end_stage("retrieval")
        self._last_candidates = candidates

        # ── Post-retrieval ─────────────────────────────────────
        tracker.start_stage("post")
        final_hits = self._post_process(user_message, candidates, top_k, use_compression, tracker)
        tracker.end_stage("post")

        # ── Generation ─────────────────────────────────────────
        tracker.start_stage("generation")
        system_content = self._build_system_prompt(final_hits)
        yield from stream_response(system_content, self.conversation, user_message, tracker=tracker)
        tracker.end_stage("generation")

        self._last_usage = getattr(stream_response, "_last_usage", {"input": 0, "output": 0})
        self._last_cost_summary = tracker.get_summary()

    # ── 검색 단독 실행 ────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.2,
        max_per_source: int = 2,
    ) -> list[dict]:
        query_vector = embed_texts([query])[0]
        return self.store.search(query_vector, top_k, threshold, max_per_source)

    # ── 관리 ──────────────────────────────────────────────────

    def reset_conversation(self):
        self.conversation.clear()

    def get_indexed_sources(self) -> list[dict]:
        return self.store.get_sources()

    def delete_source(self, source: str) -> bool:
        before = self.store.total_chunks()
        self.store.remove_source(source)
        return self.store.total_chunks() < before

    def get_stats(self) -> dict:
        return {
            "total_chunks":    self.store.total_chunks(),
            "total_documents": len(self.store.get_sources()),
            "embedding_model": EMBEDDING_MODEL,
            "chat_model":      CHAT_MODEL,
        }
