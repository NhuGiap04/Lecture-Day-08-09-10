# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode = "dense"
# chunk_size = 500 tokens (assumed default)
# overlap = 50 tokens (assumed default)
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "openai/gpt-oss-20b"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.20 /5 |
| Answer Relevance | 4.60 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 3.00 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
> - **gq05 (Access Control):** Answer Relevance = 2/5, Completeness = 1/5. Dense retrieval dường như không lấy đúng hoặc đủ context cho phép model trả lời đúng yêu cầu cụ thể (thời gian xử lý, training bắt buộc, role cụ thể là CISO thay vì IT Manager).
> - **gq07 (Insufficient Context):** Faithfulness = 3/5, Completeness = 2/5. Model có xu hướng hallucinate câu trả lời vì bị thiếu context, dẫn tới độ tin cậy và hoàn thiện thấp.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias cụ thể trong các context phức tạp.
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [x] Generation: Context quá dài / chứa thông tin gây nhiễu → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13
**Biến thay đổi:** Đổi sang Hybrid + Reranking + Query Expansion
**Lý do chọn biến này:**
> Baseline thất bại trong việc trả lời đầy đủ và trực tiếp ở các gq05 (câu hỏi phức tạp nhiều chi tiết) do dense search gặp khó khăn với cross-match hoặc technical keyword (e.g. CISO, Level 4). Hybrid sẽ kết hợp được cả ngữ nghĩa (dense) và keyword exact match (sparse/lexical), Rerank sẽ đẩy các passage quan trọng nhất lên đầu và Query Expansion sẽ giúp model tìm được document tốt hơn.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   
use_rerank = True
query_transform_strategy = "expansion"
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.20/5 | 4.20/5 | +0.00 |
| Answer Relevance | 4.60/5 | 4.60/5 | +0.00 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.00/5 | 3.10/5 | +0.10 |

**Nhận xét:**
> Variant 1 cải thiện đáng kể ở **gq05** (Relevance tăng từ 2 lên 5, Completeness tăng từ 1 lên 3) cho thấy hybrid và reranker đã đưa đúng chunk vào context. Ở **gq07**, Faithfulness tăng lên 5 (model biết Abstain tốt hơn) nhưng Relevance lại giảm đánh kể (có thể do judge không đánh giá cao câu trả lời abstain/từ chối). Nhìn chung, Completeness có sự cải thiện nhẹ (+0.10).

**Kết luận:**
> Variant 1 (Hybrid + Rerank + Expansion) cho thấy có khắc phục được phần nào các câu missing/wrong facts, làm cho pipeline bao quát hơn (Completeness tăng). Mặc dù điểm tổng thể chỉ nhích nhẹ, nhưng việc giải quyết triệt để sự thiếu sót của các câu hỏi dài (gq05) là một tín hiệu rất tích cực chứng minh tính hiệu quả của hybrid + reranker.

---

## Tóm tắt học được

> Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Hallucination (suy diễn ngoài văn bản) ở những câu không đủ context (gq07) hoặc thiếu ý ở các câu hỏi cross-document (gq02) do đoạn trả về không đủ bao quát tất cả file cần thiết.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Hybrid search kết hợp với Rerank. Nó khắc phục các case Dense Retrieval bị bối rối bởi các thuật ngữ chuyên ngành hay câu hỏi đòi hỏi matching nhiều góc độ thông tin.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > - Chỉnh sửa kịch bản chunking: Tăng chunk_size/overlap để tránh mất ngữ cảnh ở các chính sách dài hoặc thêm metadata filtering để xử lý chính xác Temporal/Version scoping.
   > - Thay đổi logic Reranker hoặc dùng Rerank mạnh hơn chuyên cho tiếng Việt để đẩy điểm Relevance/Completeness lên mức tối đa.
