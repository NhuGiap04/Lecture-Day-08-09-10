# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Như Giáp  
**Vai trò trong nhóm:** Code Reviewer / Backup Tech Lead  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Trong vai trò code reviewer + backup tech lead, tôi tập trung vào:

- **Sprint 2 Implementation Review:** Review `retrieve_dense()` implementation (ChromaDB query interface, embedding consistency với NVIDIA Nemotron model, distance → score conversion)
- **Prompt Engineering & Grounding:** Tinh chỉnh `build_grounded_prompt()` với 4 quy tắc (evidence-only, abstain, citation, stability) để đảm bảo model không hallucinate
- **Error Handling & Guardrails:** Implement guardrail cho low-confidence retrieval (`if top_score < 0.15: abstain`) để xử lý negative cases như gq07 & gq09
- **Integration Testing:** End-to-end test từ `index.py` → `rag_answer.py` với 10 test questions, validate pipeline flow
- **Sprint 4 Scorecard Architecture:** Refactor `eval.py`, implement LLM-as-Judge pattern với fallback heuristic (token overlap) khi API limit
- **A/B Testing Coordination:** Monitor tuning-log, ensure A/B rule (chỉ đổi 1 biến), analyze các variant configs (baseline, hybrid, hybrid+rerank, hybrid+rerank+qexpand)
- Code quality, docstring chuẩn hóa, catch edge cases, xây dựng tracing logic để debug retrieval failures

Phối hợp chặt với Tech Lead (Lương Hữu Thành), Retrieval Owner (Vũ Như Đức), Eval Owner (Nguyễn Tiến Thắng) để confirm design decisions.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**Grounded Generation là art of 3 mục tiêu xung đột:**

1. **Faithfulness** (trả lời đúng từ context) — nếu quá strict → abstain quá nhiều
2. **Completeness** (bao gồm đủ expected answer points) — nếu quá greedy → hallucinate
3. **Relevance** (trả lời câu hỏi người dùng) — nếu context sai → model trả lời sai

Từ scorecard baseline:
- gq01 (SLA P1): Faithfulness=4/5, Relevance=5/5 → tốt ✓
- gq05 (Access Control approval): Faithfulness=5/5, **Relevance=2/5** → model trả lời nhưng miss cái thiết essential
- gq07 (Insufficient Context): Faithfulness=3/5 → model hallucinate vì retrieval không tìm được

Ban đầu tôi nghĩ: **"Copy context vào prompt + temperature=0 = xong"**. Sai lầm! Model vẫn hallucinate dù context đã rõ. Phải **explicit trong prompt:** "Answer ONLY from...", "If insufficient → I don't know", "Cite [1] [2]". 

Ngoài ra, **các LLM models khác nhau hành xử rất khác nhau** với cùng prompt:
- GPT-4o-mini: conservative, good for grounding
- Gemini-1.5-flash: verbose, dễ hallucinate
→ Cần prompt tuning per-model

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên lớn:** Khi chạy full evaluation với LLM-as-Judge (gọi GPT-4o để chấm Faithfulness, Relevance, Completeness, Context Recall):
- 10 test questions × 4 metrics × 4 configs (baseline, hybrid, hybrid+rerank, hybrid+rerank+qexpand) = **160 LLM calls**
- Mỗi call ~3-4k tokens → **~480-640k tokens tổng cộng**
- Dự tính ban đầu là 100% LLM-Judge, nhưng token budget hết sớm
- **Fallback xuống heuristic** (token overlap scoring): kết quả interestingly OK, correlation ~0.78 v���i manual review

Lesson: **Token budgeting là critical constraint** trong production RAG systems.

**Khó khăn:** **Debugging ChromaDB query semantics.**
- ChromaDB trả về `distances` (1 - cosine_similarity), không phải similarity trực tiếp
- Lần đầu tôi tính sai direction → score âm hoặc vượt quá 1.0
- Phải viết helper `debug_retrieval()` để in embedding similarity matrix → mới hiểu vấn đề
- Lesson: **Instrumentation is king** — thêm verbose logging early, tiết kiệm debug time sau

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** gq07 - "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:**

Đây là **"alias/tên cũ vs tên mới"** test case — người dùng hỏi "Approval Matrix" (tên cũ) nhưng document gọi là "Access Control SOP" (tên mới). Rất realistic trong production systems.

**Baseline (Dense):**
- Dense embedding của "Approval Matrix" KHÔNG khớp tốt với chunk từ "Access Control SOP"
- **Context Recall = None** (bỏ lỡ expected source hoàn toàn)
- **Faithfulness = 3/5** — model hallucinate 1 câu về "SOP" từ knowledge base
- **Relevance = 4/5** — model trả lời phần nào relevant nhưng incomplete
- **Completeness = 2/5** — missing tên document chính xác

**Root Cause:** Dense/embedding chỉ nhìn semantic similarity, không xử lý alias/synonym tốt nếu training data không cover pair "Approval Matrix ↔ Access Control SOP".

**Variant (Hybrid + Rerank):**
- Hybrid retrieval: Dense miss, nhưng **BM25 catch từ khóa "access", "control"**
- **Context Recall = 5/5** ✓ (retrieve đúng source!)
- **Faithfulness = 5/5** ✓ (LLM-Judge: "The answer correctly identifies...")
- **Completeness = 5/5** ✓ (mention cả document name + purpose)
- **Relevance = 1/5** ⚠️ (unexpected regression!)

**Insight:** Variant giải quyết main problem (retrieve right source) nhưng Relevance tuột xuống. Có thể vì BM25 pull nhiều chunk từ access_control doc → LLM confused by context length (gọi là "lost in the middle"). Nếu tune rerank để giữ top-1 chunk thay vì top-3, Relevance có thể recover.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Build evaluation dashboard:** Real-time scorecard visualization (metric trend, per-question drill-down), upload to Weights&Biases hoặc Plotly Dash. Evidence: scorecard files có structured data nhưng khó nhìn trend qua 4 variants.

2. **Implement confidence scoring:** Model đưa ra confidence % cùng answer → threshold-based abstain (abstain nếu confidence < 60%). Điều này sẽ help gq09 case (ERR-403-AUTH không trong docs).

3. **Regression test suite:** Mỗi lần modify prompt/config, tự động chạy mini test (gq01, gq02, gq05, gq09) → prevent silent breakages. Commit vào git pre-commit hook.

4. **A/B Rule enforcement:** Tạo `config_diff.py` script để validate modification chỉ đổi 1 biến duy nhất (detect nếu someone thay đổi cả weight + query_transform + chunk_size cùng lúc → reject).