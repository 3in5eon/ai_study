"""
6주차: Streamlit UI + 세션 관리 + 웹 기반 RAG 서비스

[5주차 대비 추가된 기능]
  1. 멀티 세션: 대화 여러 개를 독립적으로 관리 (생성·전환·삭제)
  2. 세션 영속: sessions.json에 자동 저장 → 서버 재시작 후에도 대화 유지
  3. 세션 이름 변경: 인라인 편집
  4. 대화 내보내기: 세션 내용을 마크다운 파일로 다운로드
  5. 파일 업로드 개선: 드래그앤드롭 + 진행 상태 표시
  6. 누적 비용 대시보드: 세션별 비용 비교 탭

[세션 전환 원리]
  전환 전: 현재 messages + rag.conversation → SessionManager에 저장
  전환 후: 새 세션의 messages + conversation → 화면/RAG에 복원
  → 각 세션이 독립적인 대화 문맥을 유지
"""

import os
import streamlit as st

from rag_pipeline import AdvancedRagPipeline
from session_manager import SessionManager

# ── 경로 설정 ───────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
SESSIONS_PATH = os.path.join(_BASE, "data", "sessions.json")

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="청년도우미 RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ════════════════════════════════════════════════════════════════

def _render_source_cards(hits: list[dict]):
    items = []
    for hit in hits:
        source    = hit["metadata"]["source"]
        sim       = hit["similarity"]
        chunk_idx = hit["metadata"]["chunk_index"]
        badge     = " 🗜️" if hit.get("compressed") else ""
        items.append(f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:4px;padding:2px 7px;font-size:10px;color:#64748b;margin-right:4px;">{source}{badge} {sim:.0%} #{chunk_idx}</span>')
    tooltip_content = "".join(items)
    st.markdown(
        f'''<div style="margin-top:4px;position:relative;display:inline-block;">
  <span style="font-size:10px;color:#94a3b8;cursor:default;border-bottom:1px dotted #94a3b8;">📎 참고출처</span>
  <div style="display:none;position:absolute;bottom:120%;left:0;background:#1e293b;border:1px solid #334155;
    border-radius:6px;padding:8px 10px;white-space:nowrap;z-index:999;box-shadow:0 4px 12px rgba(0,0,0,0.3);">
    {tooltip_content}
  </div>
</div>
<style>
  div:has(> span):hover > div {{ display:block !important; }}
</style>''',
        unsafe_allow_html=True,
    )


def _render_cost_badge(cost: dict):
    if not cost:
        return
    usd = cost.get("total_cost_usd", 0)
    sec = cost.get("total_elapsed", 0)
    tok = cost.get("total_input_tokens", 0) + cost.get("total_output_tokens", 0)
    st.markdown(
        f"""<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">
  <span style="background:#f0fdf4;border:1px solid #86efac;border-radius:6px;
    padding:2px 9px;font-size:11px;color:#166534;">💰 ${usd:.5f} (₩{usd*1380:.2f})</span>
  <span style="background:#eff6ff;border:1px solid #93c5fd;border-radius:6px;
    padding:2px 9px;font-size:11px;color:#1e40af;">⏱ {sec:.1f}s</span>
  <span style="background:#faf5ff;border:1px solid #c4b5fd;border-radius:6px;
    padding:2px 9px;font-size:11px;color:#5b21b6;">🔤 {tok:,} 토큰</span>
</div>""",
        unsafe_allow_html=True,
    )


def _render_pipeline(rag: AdvancedRagPipeline):
    cost     = rag._last_cost_summary
    by_stage = cost.get("by_stage", {})
    with st.expander("🔬 파이프라인 실행 결과", expanded=True):
        for stage, (num, phase, label) in {
            "pre":         ("①", "Pre-retrieval",  "Multi-query Generation"),
            "embedding":   ("②", "Retrieval",      "Hybrid Search 임베딩"),
            "reranking":   ("③", "Post-retrieval", "GPT Re-ranking"),
            "compression": ("③", "Post-retrieval", "Context Compression"),
            "generation":  ("④", "Generation",     "LLM 스트리밍"),
        }.items():
            s = by_stage.get(stage, {})
            if not s:
                continue
            st.markdown(f"#### {num} {phase}: {label}")
            c1, c2, c3 = st.columns(3)
            c1.metric("시간",  f"{s.get('elapsed',0):.2f}s")
            c2.metric("토큰",  f"{s.get('input_tokens',0)+s.get('output_tokens',0):,}")
            c3.metric("비용",  f"${s.get('cost_usd',0):.5f}")
            st.divider()


# ── 커스텀 CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  .session-card {
    border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 6px;
    background: white; cursor: pointer;
  }
  .session-card.active {
    border-color: #2563eb; background: #eff6ff;
  }
  .session-name { font-weight: 700; font-size: 14px; color: #111827; }
  .session-meta { font-size: 11px; color: #6b7280; margin-top: 2px; }
  .cost-badge {
    display: inline-block;
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 6px; padding: 2px 8px;
    font-size: 11px; color: #166534; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 초기화 ───────────────────────────────────────────

def _init_state():
    if "sm" not in st.session_state:
        st.session_state.sm = SessionManager(SESSIONS_PATH)
    if "rag" not in st.session_state:
        st.session_state.rag = AdvancedRagPipeline()
    if "active_sid" not in st.session_state:
        st.session_state.active_sid = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "auto_indexed" not in st.session_state:
        st.session_state.auto_indexed = False
    if "rename_sid" not in st.session_state:
        st.session_state.rename_sid = None


_init_state()

sm: SessionManager       = st.session_state.sm
rag: AdvancedRagPipeline = st.session_state.rag


# ── 세션 전환 ──────────────────────────────────────────────────

def switch_session(new_sid: str):
    """현재 세션 저장 후 새 세션으로 전환"""
    cur = st.session_state.active_sid
    if cur:
        sm.save_messages(cur, st.session_state.messages, rag.conversation)

    session = sm.get(new_sid)
    st.session_state.messages = list(session["messages"])
    rag.conversation = list(session["conversation"])
    st.session_state.active_sid = new_sid


def new_session():
    """새 세션 생성 후 전환"""
    cur = st.session_state.active_sid
    if cur:
        sm.save_messages(cur, st.session_state.messages, rag.conversation)
    sid = sm.create()
    st.session_state.messages = []
    rag.conversation = []
    st.session_state.active_sid = sid


# ── 자동 인덱싱 ────────────────────────────────────────────────

if not st.session_state.auto_indexed:
    with st.spinner("data/ 폴더 자동 인덱싱 중..."):
        results = rag.auto_index_data_dir()
    if results:
        st.toast(f"✅ {len(results)}개 파일 자동 인덱싱 완료!")
    st.session_state.auto_indexed = True


# ── 첫 세션 자동 생성 ──────────────────────────────────────────

if st.session_state.active_sid is None:
    sessions = sm.list()
    if sessions:
        switch_session(sessions[0]["id"])
    else:
        new_session()


# ════════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 청년도우미")
    st.caption("Advanced RAG 기반 청년 정책 상담 챗봇")
    st.divider()

    # ── 세션 목록 ──────────────────────────────────────────────
    st.markdown("### 💬 대화 목록")

    if st.button("＋ 새 대화", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    st.markdown("")
    sessions = sm.list()

    for session in sessions:
        sid       = session["id"]
        is_active = sid == st.session_state.active_sid
        msg_count = len([m for m in session["messages"] if m["role"] == "user"])
        cost      = session["total_cost_usd"]

        col_main, col_act = st.columns([5, 1])

        with col_main:
            # 이름 수정 모드
            if st.session_state.rename_sid == sid:
                new_name = st.text_input(
                    "이름 변경", value=session["name"],
                    key=f"rename_input_{sid}", label_visibility="collapsed"
                )
                c1, c2 = st.columns(2)
                if c1.button("저장", key=f"save_{sid}", use_container_width=True):
                    sm.rename(sid, new_name)
                    st.session_state.rename_sid = None
                    st.rerun()
                if c2.button("취소", key=f"cancel_{sid}", use_container_width=True):
                    st.session_state.rename_sid = None
                    st.rerun()
            else:
                border_color = "#2563eb" if is_active else "#e5e7eb"
                bg_color     = "#eff6ff" if is_active else "white"
                name_color   = "#1d4ed8" if is_active else "#111827"
                st.markdown(
                    f"""<div style="border:1px solid {border_color};border-radius:10px;
                    padding:8px 12px;background:{bg_color};margin-bottom:2px;">
                    <div style="font-weight:700;font-size:13px;color:{name_color};">
                        {"▶ " if is_active else ""}{session['name']}
                    </div>
                    <div style="font-size:11px;color:#6b7280;margin-top:2px;">
                        질문 {msg_count}개 · ${cost:.5f}
                    </div></div>""",
                    unsafe_allow_html=True,
                )
                if not is_active:
                    if st.button("전환", key=f"sw_{sid}", use_container_width=True):
                        switch_session(sid)
                        st.rerun()

        with col_act:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            if st.button("✏️", key=f"ren_{sid}", help="이름 변경"):
                st.session_state.rename_sid = sid
                st.rerun()
            if st.button("🗑", key=f"del_{sid}", help="삭제"):
                if is_active:
                    remaining = [s for s in sessions if s["id"] != sid]
                    if remaining:
                        switch_session(remaining[0]["id"])
                    else:
                        new_session()
                sm.delete(sid)
                st.rerun()

    st.divider()

    # ── 관리자 모드 (기본 숨김) ────────────────────────────────
    with st.expander("🔧 관리자 설정", expanded=False):
        admin_pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw")
        is_admin = admin_pw == "admin1234"

        if is_admin:
            st.success("관리자 모드 활성화")

            # 문서 관리
            st.markdown("#### 📂 문서 관리")
            uploaded = st.file_uploader(
                "MD / TXT / PDF 업로드",
                type=["md", "txt", "pdf"],
                label_visibility="visible",
            )
            if uploaded:
                if st.button("인덱싱 시작", type="primary", use_container_width=True):
                    import tempfile
                    suffix = os.path.splitext(uploaded.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    with st.spinner(f"'{uploaded.name}' 처리 중..."):
                        result = rag.index_document(tmp_path, source_name=uploaded.name)
                    os.remove(tmp_path)
                    st.success(f"완료! {result['chunks']}개 청크 ({result['chars']:,}자)")
                    st.rerun()

            sources = rag.get_indexed_sources()
            if sources:
                for src in sources:
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{src['source']}**  \n`{src['chunks']}청크`")
                    if c2.button("삭제", key=f"delsrc_{src['source']}", use_container_width=True):
                        rag.delete_source(src["source"])
                        st.rerun()
            else:
                st.caption("인덱싱된 문서 없음")

            st.markdown("#### ⚙️ 검색 설정")
            top_k          = st.slider("top-k", 1, 8, 5)
            threshold      = st.slider("유사도 임계값", 0.0, 1.0, 0.2, step=0.05)
            max_per_source = st.slider("문서당 최대 청크", 1, 3, 2)
            use_compression = st.checkbox("Context Compression", value=True)
            show_sources   = st.checkbox("출처 카드 표시", value=True)
            show_pipeline  = st.checkbox("파이프라인 단계 표시", value=False)

            st.markdown("#### 📊 통계")
            stats = rag.get_stats()
            c1, c2 = st.columns(2)
            c1.metric("문서", f"{stats['total_documents']}개")
            c2.metric("청크", f"{stats['total_chunks']}개")
        else:
            # 관리자 아닐 때 기본값 고정
            top_k           = 5
            threshold       = 0.2
            max_per_source  = 2
            use_compression = True
            show_sources    = True
            show_pipeline   = False
            sources         = rag.get_indexed_sources()

    active_session = sm.get(st.session_state.active_sid)
    if active_session:
        usd = active_session["total_cost_usd"]
        st.markdown(
            f"<span style='font-size:13px;font-weight:700;color:#166534'>"
            f"세션 비용: ${usd:.5f} (₩{usd*1380:.1f})</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 대화 내보내기 ──────────────────────────────────────────
    if active_session and active_session["messages"]:
        md_text = sm.export_markdown(st.session_state.active_sid)
        st.download_button(
            "⬇️ 대화 내보내기 (MD)",
            data=md_text,
            file_name=f"{active_session['name']}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        rag.conversation = []
        sm.save_messages(st.session_state.active_sid, [], [])
        st.rerun()


# ════════════════════════════════════════════════════════════════
# 메인 화면: 탭
# ════════════════════════════════════════════════════════════════

active_session = sm.get(st.session_state.active_sid)
session_name   = active_session["name"] if active_session else "대화"

if st.session_state.get("admin_pw") == "admin1234":
    tab_chat, tab_dashboard, tab_pipeline = st.tabs([
        f"💬 {session_name}",
        "📊 비용 대시보드",
        "⚙️ 파이프라인 구조",
    ])
else:
    tab_chat = st.container()
    tab_dashboard = None
    tab_pipeline  = None


# ════════════════════════════════════════════════════════════════
# 탭 1: 챗봇
# ════════════════════════════════════════════════════════════════

with tab_chat:
    st.header(f"💬 {session_name}")


    if not sources:
        st.info("왼쪽 사이드바에서 문서를 업로드하거나, data/ 폴더에 파일을 넣고 재시작하세요.")

    # 이전 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("hits") and show_sources:
                _render_source_cards(msg["hits"]) if "hits" in msg else None

    # ── 사용자 입력 ─────────────────────────────────────────────
    user_input = st.chat_input(
        "질문을 입력하세요... (예: 청년도약계좌 자격이 뭐야?)" if sources
        else "먼저 사이드바에서 문서를 업로드하세요",
        disabled=not sources,
    )

    if user_input:
        # 첫 번째 질문이면 세션 이름 자동 지정
        if len(st.session_state.messages) == 0:
            words = user_input.replace("?", "").replace("？", "").strip()
            auto_name = words[:20] + ("…" if len(words) > 20 else "")
            sm.rename(st.session_state.active_sid, auto_name)

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # AI 응답 스트리밍
        with st.chat_message("assistant"):
            full_response = st.write_stream(
                rag.chat_stream(
                    user_input,
                    top_k=top_k,
                    threshold=threshold,
                    max_per_source=max_per_source,
                    use_compression=use_compression,
                )
            )

        hits_result  = rag._last_compressed
        cost_summary = rag._last_cost_summary
        usage_result = rag._last_usage

        # 출처 카드
        if hits_result and show_sources:
            with st.chat_message("assistant"):
                _render_source_cards(hits_result)
                _render_cost_badge(cost_summary)

        # 파이프라인 단계
        if show_pipeline:
            _render_pipeline(rag)

        # 메시지 저장
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_response,
            "hits":    hits_result,
        })

        # 세션 데이터 업데이트
        sm.add_cost(
            st.session_state.active_sid,
            cost_summary.get("total_cost_usd", 0),
            usage_result,
        )
        sm.save_messages(
            st.session_state.active_sid,
            st.session_state.messages,
            rag.conversation,
        )
        st.rerun()


# ════════════════════════════════════════════════════════════════
# 탭 2, 3: 관리자 전용
# ════════════════════════════════════════════════════════════════

if tab_dashboard is not None:
    with tab_dashboard:
        st.header("📊 세션별 비용 대시보드")
        all_sessions = sm.list()
        if not all_sessions:
            st.info("대화를 시작하면 비용이 여기에 표시됩니다.")
        else:
            total_usd = sum(s["total_cost_usd"] for s in all_sessions)
            total_in  = sum(s["total_tokens"]["input"] for s in all_sessions)
            total_out = sum(s["total_tokens"]["output"] for s in all_sessions)
            total_q   = sum(len([m for m in s["messages"] if m["role"] == "user"]) for s in all_sessions)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("전체 누적 비용", f"${total_usd:.5f}")
            c2.metric("원화 환산",      f"₩{total_usd*1380:.1f}")
            c3.metric("총 질문 수",     f"{total_q}개")
            c4.metric("총 토큰",        f"{total_in+total_out:,}")
            st.divider()
            st.subheader("세션별 상세")
            for s in all_sessions:
                q_count = len([m for m in s["messages"] if m["role"] == "user"])
                avg     = s["total_cost_usd"] / q_count if q_count else 0
                is_cur  = "👈 현재" if s["id"] == st.session_state.active_sid else ""
                with st.expander(f"**{s['name']}** {is_cur}  —  ${s['total_cost_usd']:.5f}"):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("질문 수",     f"{q_count}개")
                    cc2.metric("총 비용",     f"${s['total_cost_usd']:.5f}")
                    cc3.metric("질문당 평균", f"${avg:.5f}")
                    cc4.metric("원화",        f"₩{s['total_cost_usd']*1380:.1f}")
                    tok = s["total_tokens"]
                    st.caption(f"입력: {tok['input']:,} / 출력: {tok['output']:,} / 생성일: {s['created_at'][:10]}")

if tab_pipeline is not None:
    with tab_pipeline:
        st.header("⚙️ Advanced RAG 파이프라인 구조")
        st.markdown("""
```
질문 → classify → single/multi
  ↓
① Pre-retrieval: Multi-query Generation
  ↓
② Retrieval: BM25 + Vector → RRF Fusion
  ↓
③ Post-retrieval: Re-ranking → Compression
  ↓
④ Generation: LLM 스트리밍
```
""")
        st.divider()
        st.subheader("6주차 추가 기능")
        st.markdown("""
| 기능 | 설명 |
|---|---|
| **멀티 세션** | 여러 대화 독립 관리, 자유롭게 전환 |
| **세션 영속** | sessions.json 자동 저장 |
| **관리자 모드** | 문서 관리·검색 설정 숨김 |
| **비용 대시보드** | 세션별 비용·토큰 비교 |
| **싱글홉/멀티홉** | 질문 유형 자동 분류 |
""")


