# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Vũ Như Đức  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Tôi làm phần **Sprint 1 (STEP 3: Embed + Store) và Sprint 3 (retrieval tuning)**. Cụ thể:

- **STEP 3 - Embed & Store:** Embed các chunk bằng embedding model và lưu vào ChromaDB với metadata (source, section, department, effective_date, access)
- **Chunking Strategy:** Tinh chỉnh chunk size từ 300 → 400 tokens, overlap từ 50 → 80 tokens qua 3-4 iteration test
- **Verify Storage Quality:** Kiểm tra các chunk được lưu đúng vào ChromaDB, metadata đầy đủ, không có chunk nào bị cắt giữa điều khoản quan trọng
- **Sprint 3 Hybrid Retrieval:** Implement retrieve_dense() + retrieve_sparse() (BM25) + RRF fusion (60 là hằng số)
- **Reranking Module:** Thêm cross-encoder lightweight rerank kết hợp dense score (70%) + lexical overlap (30%)
- Kiểm thử từng retrieval strategy trên Q1-Q10, ghi lại Context Recall metric

Phối hợp chặt với Tech Lead cho config tuning, với Eval Owner cho test case design.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**Hybrid Retrieval** không phải là "thêm BM25 là xong". Khi merge dense + sparse scores, cách bạn **weight & normalize** là then chốt:
- RRF (Reciprocal Rank Fusion) tỏ ra hiệu quả hơn simple score addition
- K_rrf = 60 là default, nhưng có thể tune theo corpus size và query type
- Dense (embedding) good at paraphrase, sparse (BM25) good at exact term — **complementary strengths**

Ví dụ real: Q7 "Approval Matrix" — dense miss vì embedding khác, sparse catch vì từ khóa "access control" overlap → hybrid save the day. Trước lab này tôi chỉ biết dense = SOTA, bây giờ thấy rằng hybrid là **practical choice** cho corpus lẫn lộn text tự nhiên + tên riêng.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên:** Sparse retrieval (BM25) dễ implement hơn tưởng (chỉ cần `rank-bm25` library), nhưng **tuning tokenization** rất quan trọng:
- Simple split() vs regex `\w+` vs stemming → kết quả khác biệt
- Cho Vietnamese text, BM25 mặc định khá OK, nhưng sẽ tốt hơn nếu có Vietnamese tokenizer (pyvi)

**Khó khăn:** **Dense + Sparse fusion không dễ balance**. Ban đầu tôi dùng 50-50 weight, kết quả tệ (LLM confusion từ 2 strategy khác nhau). Sau đó thử 60-40 (dense prioritized), hoạt động tốt hơn cho Q1-Q6, nhưng Q7 vẫn tệ cho tới khi thêm query expansion.

Lesson learned: **A/B Rule** từ slide rất đúng — tôi cố thay đổi đồng thời weight + query_transform → khó debug. Lần sau phải test từng biến một cách nghiêm túc.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** Q3 - "Ai phải phê duyệt để cấp quyền Level 3?"

**Phân tích:**

- **Expected answer:** "Level 3 cần phê duyệt từ Line Manager, IT Admin, và IT Security"
- **Document source:** `access_control_sop.txt`, section "=== Level 3 Approval ===" 
- **Baseline (Dense):** 
  - Dense embedding của "cấp quyền Level 3" khớp tốt với section này
  - Context Recall = 5/5 (retrieve đúng source)
  - Faithfulness = 4/5 (model trả lời đúng nhưng missing chi tiết về "concurrent approval" requirement)
  
- **Variant (Hybrid + Rerank):**
  - BM25 bắt từ khóa "phê duyệt", "Level 3", "approver"
  - Rerank re-score các passage dựa vào relevance riêng để "Level 3" → giữ top chunk
  - Faithfulness = 5/5, Completeness = 5/5 (toàn bộ approval chain được mention)

- **Insight:** Hybrid + Rerank không phải lúc nào cũng cải thiện, nhưng cho **multi-entity queries** (tên organization + role + condition) thì rất hữu ích.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Implement ColBERT** (late interaction COLBERT model) thay dense embedding — tính toán relevance ở token level có thể better capture "long-form context" như access control policy.

2. **Add semantic caching layer:** Cache query → (dense_emb, bm25_tokens) → retrieval result, tái sử dụng khi query similar → giảm latency.

3. **Query-specific weighting:** Detect query type (factual / procedural / multi-faceted) → auto-adjust dense_weight vs sparse_weight, ví dụ procedural queries dùng higher sparse weight.