# 6주차: Streamlit UI + 세션 관리 - minseon

## 기술 스택

| 항목 | 선택 | 대안 | 선택 이유 |
|------|------|------|----------|
| UI 프레임워크 | Streamlit | FastAPI + React | Python만으로 빠르게 웹 서비스 구현 가능 |
| 세션 저장 | JSON 파일 | SQLite, Redis | 별도 DB 설치 없이 간단하게 영속 저장 |
| RAG 파이프라인 | Advanced RAG (week05 재사용) | Naive RAG | 검색 품질 유지하면서 UI/세션 기능 집중 |
| 쿼리 전략 | 싱글홉 / 멀티홉 자동 분류 | 무조건 멀티쿼리 | 단순 질문 비용 절감, 복잡 질문 품질 향상 |

---

## 5주차 대비 추가된 기능

| 기능 | 설명 |
|------|------|
| 멀티 세션 | 여러 대화를 독립적으로 관리, 자유롭게 전환 |
| 세션 영속 | sessions.json 자동 저장 → 서버 재시작 후에도 대화 유지 |
| 세션 이름 변경 | ✏️ 버튼 클릭 → 인라인 편집 |
| 세션 삭제 | 🗑 버튼으로 삭제, 자동으로 다음 세션 전환 |
| 대화 내보내기 | 세션 내용을 .md 파일로 다운로드 |
| 비용 대시보드 | 전체 누적 비용 + 세션별 비용 비교 |
| 싱글홉/멀티홉 분류 | 질문 유형 자동 감지 → 검색 전략 최적화 |

---

## 폴더 구조

```
week06-webservice/minseon/
├── app.py                  # Streamlit UI (멀티세션)
├── rag_pipeline.py         # Advanced RAG 오케스트레이터
├── session_manager.py      # 세션 CRUD + JSON 영속
├── requirements.txt
├── data/                   # 청년 정책 문서
│   ├── 국가장학금.md
│   ├── 청년도약계좌.md
│   ├── 청년희망적금.md
│   ├── 청년_주거_지원.md
│   ├── 미래내일일경험프로그램.md
│   └── 청년성장프로젝트.md
└── services/
    ├── query_classifier.py   # [NEW] 싱글홉/멀티홉 분류
    ├── query_service.py      # Multi-query Generation
    ├── vector_store.py       # BM25 + Vector + RRF
    ├── reranker_service.py   # GPT Re-ranking
    ├── compression_service.py# Context Compression
    ├── embedding_service.py  # OpenAI 임베딩
    ├── llm_service.py        # GPT 스트리밍
    ├── cost_tracker.py       # 비용·시간 추적
    ├── chunking_service.py   # 마크다운 구조 기반 청킹
    └── document_service.py   # 문서 로딩 (md/txt/pdf)
```

---

## 코드 실행

```bash
# 1. 가상환경 활성화 (Python 3.12)
c:\Users\user\ai_study\.venv312\Scripts\activate

# 2. 패키지 설치 (처음 한 번만)
pip install -r c:\Users\user\ai_study\week06-webservice\minseon\requirements.txt

# 3. 실행
cd c:\Users\user\ai_study\week06-webservice\minseon
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 실행코드
streamlit run c:\Users\user\ai_study\week06-streamlit-ui\minseon\app.py

streamlit run c:\Users\user\ai_study\week06-streamlit-ui\minseon\app.py



---

## 핵심 구현 설명

### 1. 세션 관리 (SessionManager)

```
sessions.json 구조:
{
  "abc12345": {
    "name": "청년 주거 질문",
    "messages": [...],       ← 화면 표시용
    "conversation": [...],   ← RAG 문맥 (LLM에 전달)
    "total_cost_usd": 0.00128
  }
}
```

세션 전환 시:
1. 현재 세션 messages + rag.conversation → JSON 저장
2. 새 세션 messages + conversation → 화면/RAG 복원
→ 각 세션이 독립적인 대화 문맥 유지

### 2. 싱글홉 / 멀티홉 자동 분류

```
단순 질문 → single → 검색 1번 (비용 절약)
  예) "청년도약계좌 조건이 뭐야?"

비교/복합 질문 → multi → Multi-query 3개 (품질 확보)
  예) "청년도약계좌랑 희망적금 중 뭐가 유리해?"
```

전체 질문의 약 80%가 단순 질문 → 평균 API 비용 절감

---

## Advanced RAG 파이프라인

```
질문
  ↓
① classify_query() → single / multi 판단
  ↓
② Pre-retrieval: (multi면) 쿼리 3개 생성
  ↓
③ Retrieval: BM25 + Vector → RRF Fusion
  ↓
④ Post-retrieval: Re-ranking → Compression
  ↓
⑤ Generation: LLM 스트리밍 답변
```

---

## WHY (의사결정 기록)

1. **Q**: 왜 세션을 JSON 파일로 저장했나?
   **A**: SQLite나 Redis는 별도 설치·설정이 필요하지만, JSON은 Python 기본 내장 라이브러리로 즉시 사용 가능. 이 프로젝트 규모(단일 사용자)에서는 충분함.

2. **Q**: 왜 싱글홉/멀티홉을 LLM으로 분류하나?
   **A**: 키워드 기반 분류("비교", "중에" 등)도 가능하지만 한국어 문장은 맥락이 중요해서 LLM 분류가 더 정확. 분류 비용(max_tokens=10)이 작아서 손해보다 이득이 큼.

3. **Q**: week05 코드를 왜 복사했나?
   **A**: sys.path로 다른 폴더를 참조하면 폴더 구조가 바뀔 때 깨짐. 로컬 복사 방식이 더 독립적이고 공부할 때 한 폴더에서 모든 코드를 볼 수 있어 편함.

---

## 트러블슈팅 로그

| # | 문제 상황 | 원인 | 해결 방법 |
|---|----------|------|----------|
| 1 | chromadb ImportError | Python 3.14 + pydantic v1 호환 안 됨 | Python 3.12로 가상환경 새로 생성 |
| 2 | chromadb ConfigError | chromadb 최신 버전 pydantic v2 전용 | requirements.txt에 chromadb==0.4.24 고정 |

---

## 회고

- 이번 주 배운 점: 세션 관리의 핵심은 "전환 시 저장 + 복원" 패턴. RAG의 conversation 히스토리도 세션별로 독립 관리해야 대화 문맥이 섞이지 않음.
- 다음 주 준비할 것: 더 많은 문서, 실제 사용자 피드백 기반 품질 개선
