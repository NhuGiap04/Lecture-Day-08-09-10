"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_base_url = os.getenv("NVIDIA_BASE_URL")


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score

    Đã implement:
    1. Embed query bằng cùng model đã dùng khi index (xem index.py)
    2. Query ChromaDB với embedding đó
    3. Trả về kết quả kèm score

    Gợi ý:
        import chromadb
        from index import get_embedding, CHROMA_DB_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")

        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Lưu ý: distances trong ChromaDB cosine = 1 - similarity
        # Score = 1 - distance
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query, input_type="query")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output: List[Dict[str, Any]] = []
    for i, doc in enumerate(documents):
        distance = distances[i] if i < len(distances) else 1.0
        score = 1 - float(distance)
        output.append(
            {
                "id": ids[i] if i < len(ids) else f"dense_{i}",
                "text": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": score,
            }
        )

    return output


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa

    Đã implement BM25 retrieval:
    1. Load tất cả chunks từ ChromaDB
    2. Tokenize và tạo BM25Index
    3. Query và trả về top_k kết quả
    4. Có fallback lexical score nếu thiếu rank_bm25

    Gợi ý:
        from rank_bm25 import BM25Okapi
        corpus = [chunk["text"] for chunk in all_chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    """
    import chromadb
    from index import CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    # Load full corpus từ index để chạy BM25 trên documents đã chunk
    all_rows = collection.get(include=["documents", "metadatas"])
    ids = all_rows.get("ids", [])
    docs = all_rows.get("documents", [])
    metas = all_rows.get("metadatas", [])

    if not docs:
        return []

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    tokenized_corpus = [tokenize(doc) for doc in docs]
    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    try:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
    except Exception:
        # Fallback nhẹ khi thiếu rank_bm25: dùng overlap lexical score
        bm25_scores = []
        q_terms = set(tokenized_query)
        for toks in tokenized_corpus:
            if not toks:
                bm25_scores.append(0.0)
                continue
            overlap = sum(1 for t in toks if t in q_terms)
            bm25_scores.append(float(overlap) / max(len(q_terms), 1))

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:top_k]

    output: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices):
        output.append(
            {
                "id": ids[idx] if idx < len(ids) else f"sparse_{rank}",
                "text": docs[idx],
                "metadata": metas[idx] if idx < len(metas) else {},
                "score": float(bm25_scores[idx]),
            }
        )

    return output


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản

    Args:
        dense_weight: Trọng số cho dense score (0-1)
        sparse_weight: Trọng số cho sparse score (0-1)

     Đã implement:
     1. Chạy retrieve_dense() → dense_results
     2. Chạy retrieve_sparse() → sparse_results
     3. Merge bằng RRF:
       RRF_score(doc) = dense_weight * (1 / (60 + dense_rank)) +
                        sparse_weight * (1 / (60 + sparse_rank))
       60 là hằng số RRF tiêu chuẩn
     4. Sort theo RRF score giảm dần, trả về top_k

    Khi nào dùng hybrid (từ slide):
    - Corpus có cả câu tự nhiên VÀ tên riêng, mã lỗi, điều khoản
    - Query như "Approval Matrix" khi doc đổi tên thành "Access Control SOP"
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    k_rrf = 60
    merged: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        doc_id = item.get("id") or f"dense_{rank}"
        merged.setdefault(doc_id, {**item, "score": 0.0})
        merged[doc_id]["score"] += dense_weight * (1.0 / (k_rrf + rank))

    for rank, item in enumerate(sparse_results, start=1):
        doc_id = item.get("id") or f"sparse_{rank}"
        if doc_id not in merged:
            merged[doc_id] = {**item, "score": 0.0}
        merged[doc_id]["score"] += sparse_weight * (1.0 / (k_rrf + rank))

    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked[:top_k]


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Cross-encoder: chấm lại "chunk nào thực sự trả lời câu hỏi này?"
    MMR (Maximal Marginal Relevance): giữ relevance nhưng giảm trùng lặp

    Funnel logic (từ slide):
      Search rộng (top-20) → Rerank (top-6) → Select (top-3)

    Đã implement rerank nhẹ:
    Option A — Cross-encoder:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]

    Option B — Rerank bằng LLM (đơn giản hơn nhưng tốn token):
        Gửi list chunks cho LLM, yêu cầu chọn top_k relevant nhất

    Khi nào dùng rerank:
    - Dense/hybrid trả về nhiều chunk nhưng có noise
    - Muốn chắc chắn chỉ 3-5 chunk tốt nhất vào prompt
    """
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    q_tokens = set(tokenize(query))
    if not q_tokens:
        return candidates[:top_k]

    rescored: List[Tuple[Dict[str, Any], float]] = []
    for c in candidates:
        text = c.get("text", "")
        c_tokens = tokenize(text)
        if not c_tokens:
            lexical = 0.0
        else:
            overlap = len([tok for tok in c_tokens if tok in q_tokens])
            lexical = overlap / len(q_tokens)

        dense_score = float(c.get("score", 0.0))
        final_score = 0.7 * dense_score + 0.3 * lexical
        rescored.append((c, final_score))

    rescored.sort(key=lambda x: x[1], reverse=True)

    output: List[Dict[str, Any]] = []
    for c, s in rescored[:top_k]:
        updated = dict(c)
        updated["score"] = float(s)
        output.append(updated)

    return output


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    Đã implement query transformation theo heuristic cho từng strategy.

    Ví dụ expansion prompt:
        "Given the query: '{query}'
         Generate 2-3 alternative phrasings or related terms in Vietnamese.
         Output as JSON array of strings."

    Ví dụ decomposition:
        "Break down this complex query into 2-3 simpler sub-queries: '{query}'
         Output as JSON array."

    Khi nào dùng:
    - Expansion: query dùng alias/tên cũ (ví dụ: "Approval Matrix" → "Access Control SOP")
    - Decomposition: query hỏi nhiều thứ một lúc
    - HyDE: query mơ hồ, search theo nghĩa không hiệu quả
    """
    base = query.strip()
    if not base:
        return []

    strategy = strategy.lower().strip()
    transformed: List[str] = [base]

    alias_map = {
        "approval matrix": "access control sop",
        "p1": "critical incident",
        "hoàn tiền": "refund",
        "cấp quyền": "access request",
        "ticket": "jira incident ticket",
    }

    if strategy == "expansion":
        q_lower = base.lower()
        for alias, canonical in alias_map.items():
            if alias in q_lower:
                transformed.append(base.lower().replace(alias, canonical))
                transformed.append(f"{base} {canonical}")
        transformed.append(f"{base} policy quy trình")

    elif strategy == "decomposition":
        parts = re.split(r"\?|\bvà\b|\band\b|\,", base, flags=re.IGNORECASE)
        for part in parts:
            sub = part.strip(" .")
            if len(sub) >= 6:
                transformed.append(sub)

    elif strategy == "hyde":
        transformed.append(
            f"Tài liệu nội bộ liên quan đến: {base}. Cung cấp định nghĩa, SLA, điều kiện và quy trình chi tiết."
        )

    # unique giữ thứ tự
    uniq: List[str] = []
    seen = set()
    for q in transformed:
        key = q.lower().strip()
        if key and key not in seen:
            uniq.append(q)
            seen.add(key)

    return uniq


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # Có thể mở rộng thêm effective_date/department nếu muốn debug sâu hơn.
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    Đã implement đầy đủ provider routing:

    Option A — OpenAI (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,     # temperature=0 để output ổn định, dễ đánh giá
            max_tokens=512,
        )
        return response.choices[0].message.content

    Option B — Google Gemini (cần GOOGLE_API_KEY):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    Lưu ý: Dùng temperature=0 hoặc thấp để output ổn định cho evaluation.
    """
    provider = LLM_PROVIDER

    # Ưu tiên OpenAI chuẩn nếu có key; fallback qua OpenAI-compatible NVIDIA endpoint.
    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        nvidia_base_url = os.getenv("NVIDIA_BASE_URL")

        if openai_key:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return (response.choices[0].message.content or "").strip()

        if nvidia_key and nvidia_base_url:
            from openai import OpenAI

            client = OpenAI(api_key=nvidia_key, base_url=nvidia_base_url)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return (response.choices[0].message.content or "").strip()

        raise ValueError("Thiếu OPENAI_API_KEY hoặc bộ NVIDIA_API_KEY/NVIDIA_BASE_URL trong .env")

    if provider == "gemini":
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise ValueError("Thiếu GOOGLE_API_KEY trong .env")

        import google.generativeai as genai

        genai.configure(api_key=gemini_key)
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        return (response.text or "").strip()

    raise ValueError(f"LLM_PROVIDER không hợp lệ: {provider}")


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    query_transform_strategy: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    Đã implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    Variant đã hỗ trợ:
    - Variant A: retrieval_mode="hybrid"
    - Variant B: use_rerank=True
    - Variant C: query_transform_strategy trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "query_transform_strategy": query_transform_strategy,
    }

    # --- Bước 1: Retrieve ---
    queries = (
        transform_query(query, strategy=query_transform_strategy)
        if query_transform_strategy
        else [query]
    )

    merged: Dict[str, Dict[str, Any]] = {}

    def _retrieve_one(q: str) -> List[Dict[str, Any]]:
        if retrieval_mode == "dense":
            return retrieve_dense(q, top_k=top_k_search)
        if retrieval_mode == "sparse":
            return retrieve_sparse(q, top_k=top_k_search)
        if retrieval_mode == "hybrid":
            return retrieve_hybrid(q, top_k=top_k_search)
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    for q in queries:
        partial = _retrieve_one(q)
        for rank, cand in enumerate(partial, start=1):
            cid = cand.get("id") or f"cand_{rank}_{len(merged)}"
            rank_boost = 1.0 / (rank + 1)
            merged_score = float(cand.get("score", 0.0)) + rank_boost

            if cid not in merged or merged_score > float(merged[cid].get("score", 0.0)):
                merged[cid] = {**cand, "score": merged_score, "id": cid}

    candidates = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        if len(queries) > 1:
            print(f"[RAG] Expanded queries: {queries}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Guardrail: nếu retrieval yếu, abstain sớm để giảm hallucination ---
    if not candidates:
        return {
            "query": query,
            "answer": "Không đủ dữ liệu trong tài liệu hiện có để trả lời câu hỏi này.",
            "sources": [],
            "chunks_used": [],
            "config": config,
        }

    top_score = float(candidates[0].get("score", 0.0))
    if retrieval_mode == "dense" and top_score < 0.15:
        return {
            "query": query,
            "answer": "Không đủ dữ liệu trong tài liệu hiện có để trả lời câu hỏi này.",
            "sources": [],
            "chunks_used": candidates,
            "config": config,
        }

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "sparse", "hybrid"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")

    print("\nHoan thanh rag_answer.py: da co dense/sparse/hybrid, rerank, query transform va grounded generation.")
