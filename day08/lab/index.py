"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_BASE_URL")

OPENAI_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"


# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Có thể tinh chỉnh thêm theo dữ liệu thực tế của nhóm
# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access

    Đã implement:
    - Extract metadata từ dòng đầu file (Source, Department, Effective Date, Access)
    - Bỏ các dòng header metadata khỏi nội dung chính
    - Normalize khoảng trắng, xóa ký tự rác

    Gợi ý: dùng regex để parse dòng "Key: Value" ở đầu file.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    metadata_pattern = re.compile(r"^([A-Za-z ]+):\s*(.+)$")

    for line in lines:
        stripped = line.strip()
        if not header_done:
            metadata_match = metadata_pattern.match(stripped)
            if metadata_match:
                key = metadata_match.group(1).lower().strip().replace(" ", "_")
                value = metadata_match.group(2).strip()
                if key in metadata:
                    metadata[key] = value
                continue
            if stripped.startswith("===") and stripped.endswith("==="):
                # Gặp section heading đầu tiên → kết thúc header
                header_done = True
                content_lines.append(stripped)
            elif stripped == "" or stripped.isupper():
                # Dòng tên tài liệu (toàn chữ hoa) hoặc dòng trống
                continue
            else:
                # Không còn ở phần metadata header
                header_done = True
                content_lines.append(stripped)
        else:
            content_lines.append(line.rstrip())

    cleaned_text = "\n".join(content_lines)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n[ \t]+", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    Đã implement:
    1. Split theo heading "=== Section ... ===" hoặc "=== Phần ... ===" trước
    2. Nếu section quá dài (> CHUNK_SIZE * 4 ký tự), split tiếp theo paragraph
    3. Thêm overlap: lấy đoạn cuối của chunk trước vào đầu chunk tiếp theo
    4. Mỗi chunk giữ metadata đầy đủ từ tài liệu gốc

    Gợi ý: Ưu tiên cắt tại ranh giới tự nhiên (section, paragraph)
    thay vì cắt theo token count cứng.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Bước 1: Split theo heading pattern "=== ... ===" (theo từng dòng)
    sections = re.split(r"(?m)^(===\s*.*?\s*===)\s*$", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"^===\s*.*?\s*===$", part.strip()):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.

    Đã implement split theo paragraph và overlap, có fallback khi paragraph quá dài.
    """
    if overlap_chars >= chunk_chars:
        overlap_chars = max(chunk_chars // 4, 0)

    if len(text) <= chunk_chars:
        # Toàn bộ section vừa một chunk
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    def split_long_paragraph(paragraph: str) -> List[str]:
        slices: List[str] = []
        start = 0
        para_len = len(paragraph)

        while start < para_len:
            end = min(start + chunk_chars, para_len)
            if end < para_len:
                window = paragraph[start:end]
                # Ưu tiên cắt ở xuống dòng / cuối câu
                cut_candidates = [
                    window.rfind("\n"),
                    window.rfind(". "),
                    window.rfind("! "),
                    window.rfind("? "),
                    window.rfind("; "),
                ]
                best_cut = max(cut_candidates)
                if best_cut > int(chunk_chars * 0.5):
                    # +1 để giữ dấu câu/chữ cuối hợp lý
                    end = start + best_cut + 1

            piece = paragraph[start:end].strip()
            if piece:
                slices.append(piece)

            if end >= para_len:
                break

            next_start = end - overlap_chars
            if next_start <= start:
                next_start = end
            start = next_start

        return slices

    expanded_paragraphs: List[str] = []
    for para in paragraphs:
        if len(para) > chunk_chars:
            expanded_paragraphs.extend(split_long_paragraph(para))
        else:
            expanded_paragraphs.append(para)

    def make_chunk(chunk_text: str) -> Dict[str, Any]:
        return {
            "text": chunk_text,
            "metadata": {**base_metadata, "section": section},
        }

    def overlap_tail(chunk_text: str) -> str:
        if overlap_chars <= 0:
            return ""
        tail = chunk_text[-overlap_chars:]
        split_pos = tail.find("\n")
        if 0 <= split_pos < len(tail) - 1:
            tail = tail[split_pos + 1 :]
        return tail.strip()

    chunks = []
    current_chunk = ""

    for para in expanded_paragraphs:
        candidate = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        if len(candidate) <= chunk_chars:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(make_chunk(current_chunk))
            tail = overlap_tail(current_chunk)
            current_chunk = f"{tail}\n\n{para}".strip() if tail else para
            if len(current_chunk) > chunk_chars:
                long_parts = split_long_paragraph(current_chunk)
                for long_part in long_parts[:-1]:
                    chunks.append(make_chunk(long_part))
                current_chunk = long_parts[-1] if long_parts else ""
        else:
            long_parts = split_long_paragraph(para)
            for long_part in long_parts[:-1]:
                chunks.append(make_chunk(long_part))
            current_chunk = long_parts[-1] if long_parts else ""

    if current_chunk:
        chunks.append(make_chunk(current_chunk))

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str, input_type: str = "query") -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.
    """

    if not api_key:
        raise ValueError("Thiếu NVIDIA_API_KEY trong .env")
    if not base_url:
        raise ValueError("Thiếu NVIDIA_BASE_URL trong .env")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(
            input=[text],
            model=OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"}
    )

    return [float(value) for value in response.data[0].embedding]

def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.

     Đã implement đầy đủ pipeline:
     1. Khởi tạo ChromaDB client và collection
     2. Đọc từng file trong docs_dir
     3. preprocess_document() → chunk_document() → embedding → upsert
     4. In tổng số chunk đã index

    Gợi ý khởi tạo ChromaDB:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"}
        )
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)
        print(f"    → {len(chunks)} chunks")

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"].strip()
            if not chunk_text:
                continue

            ids.append(f"{filepath.stem}_{i}")
            documents.append(chunk_text)
            metadatas.append(chunk["metadata"])
            embeddings.append(get_embedding(chunk_text, input_type="passage"))

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_chunks += len(ids)

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")
    print(f"Collection 'rag_lab' hiện có {collection.count()} chunks.")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.

    Hàm dùng để kiểm tra nhanh chất lượng index sau khi build.
    Kiểm tra:
    - Chunk có giữ đủ metadata không? (source, section, effective_date)
    - Chunk có bị cắt giữa điều khoản không?
    - Metadata effective_date có đúng không?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.

    Checklist Sprint 1:
    - Mọi chunk đều có source?
    - Có bao nhiêu chunk từ mỗi department?
    - Chunk nào thiếu effective_date?

    Đã implement kiểm tra coverage cho source/section/effective_date.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        departments = {}
        access_levels = {}
        missing_source = 0
        missing_section = 0
        missing_date = 0
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            access = meta.get("access", "unknown")
            access_levels[access] = access_levels.get(access, 0) + 1

            if meta.get("source") in ("unknown", "", None):
                missing_source += 1
            if meta.get("section") in ("unknown", "", None):
                missing_section += 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")
        print("Phân bố theo access:")
        for access, count in access_levels.items():
            print(f"  {access}: {count} chunks")
        print(f"Chunks thiếu source: {missing_source}")
        print(f"Chunks thiếu section: {missing_section}")
        print(f"Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index (cần API key để embed)
    print("\n--- Build Full Index ---")
    if api_key and base_url:
        build_index()

        # Bước 4: Kiểm tra index
        list_chunks()
        inspect_metadata_coverage()
    else:
        print("Thiếu NVIDIA_API_KEY hoặc NVIDIA_BASE_URL trong .env, bỏ qua bước build index.")

    print("\nSprint 1 setup hoàn thành!")
    print("Việc tiếp theo nên kiểm tra:")
    print("  1. Chạy lại python index.py để rebuild sau khi chỉnh chunk size")
    print("  2. Dùng list_chunks() kiểm tra section/metadata/citation quality")
    print("  3. Nếu cần tune retrieval, chuyển sang rag_answer.py (Sprint 2)")
