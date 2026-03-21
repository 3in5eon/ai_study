"""
청킹 서비스 - 텍스트를 의미 단위로 분할

[관심사 분리 역할]
  이 파일: 구조 기반 청킹 (마크다운 헤더) + Recursive Splitting 폴백
  document_service.py: 파일 로딩
  embedding_service.py: 벡터 변환

[청킹 전략]
  마크다운 헤더(#/##/###)가 있으면 → 섹션 단위로 분할 (구조 기반)
    각 청크 앞에 "[대제목 > 소제목]" 컨텍스트를 붙여 GPT가 어느 섹션인지 파악 가능
    섹션이 chunk_size 초과하면 Recursive Splitting으로 추가 분할
  헤더가 없으면 → 기존 Recursive Character Splitting 폴백
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

_HEADERS_TO_SPLIT = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    텍스트를 의미 단위로 분할

    마크다운 헤더가 있으면 섹션 구조 기반 분할,
    없으면 Recursive Character Splitting으로 폴백
    """
    if _has_markdown_headers(text):
        return _split_by_structure(text, chunk_size, chunk_overlap)
    return _split_recursive(text, chunk_size, chunk_overlap)


# ── 내부 함수 ────────────────────────────────────────────────

def _has_markdown_headers(text: str) -> bool:
    """텍스트에 마크다운 헤더(#으로 시작하는 줄)가 있는지 확인"""
    return any(line.startswith("#") for line in text.splitlines())


def _split_by_structure(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    마크다운 헤더 기준 섹션 분할

    각 청크 앞에 "[h1 > h2 > h3]" 형태의 헤더 컨텍스트를 붙임
    → GPT가 청크를 읽을 때 어느 섹션의 내용인지 파악 가능
    섹션이 chunk_size를 초과하면 Recursive Splitting으로 추가 분할
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    chunks = []
    for section in sections:
        content = section.page_content.strip()
        if not content:
            continue

        header_parts = [
            section.metadata[k]
            for k in ("h1", "h2", "h3")
            if section.metadata.get(k)
        ]
        header_ctx = " > ".join(header_parts)
        prefix = f"[{header_ctx}]\n" if header_ctx else ""

        if len(prefix + content) <= chunk_size:
            chunks.append(prefix + content)
        else:
            for sub in recursive_splitter.split_text(content):
                chunks.append(prefix + sub)

    return chunks


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """헤더 없는 문서용 Recursive Character Splitting 폴백"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)
