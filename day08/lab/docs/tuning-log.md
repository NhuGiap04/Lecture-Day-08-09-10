# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode = "dense"
# chunk_size = 400 tokens
# overlap = 80 tokens
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

## Ablation Study (Sprint 3)

**Ngày:** 2026-04-13
**Nội dung:** Lần lượt đánh giá ảnh hưởng của các thành phần (Hybrid, Rerank, Query Expansion) so với Baseline (Dense).

**1. Các Config thực nghiệm:**
- **Baseline (Dense):** Chỉ dùng dense search.
- **Variant 1 (Hybrid):** Kết hợp dense search và sparse search (BM25).
- **Variant 2 (Hybrid + Rerank):** Sử dụng hybrid search, lấy top_k lớn hơn rồi dùng Cross-Encoder (Rerank) để chọn ra những passage sát nhất.
- **Variant 3 (Hybrid + Rerank + QExpand):** Tương tự Variant 2 nhưng có phân tích và mở rộng từ khóa câu hỏi (Query Expansion).

**2. Bảng kết quả (Scorecard Grading):**

| Metric | Baseline (Dense) | Hybrid | Hybrid + Rerank | Hybrid + Rerank + QExpand | Best |
|--------|------------------|--------|-----------------|---------------------------|------|
| Faithfulness | 4.20/5 | 4.20/5 | **4.40/5** | 3.90/5 | Hybrid + Rerank |
| Answer Relevance | **4.60/5** | **4.60/5** | 4.20/5 | 4.30/5 | Dense / Hybrid |
| Context Recall | **5.00/5** | **5.00/5** | **5.00/5** | **5.00/5** | All |
| Completeness | 3.00/5 | 3.10/5 | **3.30/5** | 3.20/5 | Hybrid + Rerank |

**Cải thiện & Ảnh hưởng từng biến:**
> - **Hybrid (vs Dense):** Duy trì sự ổn định của Dense và tăng nhẹ Completeness (+0.10). Các câu hỏi thiên về tìm keyword chính xác (như mã lỗi, từ khóa kỹ thuật) được trợ giúp nhẹ nhưng hiệu ứng chưa thực sự đột phá, điểm số tổng gần tương đương.
> - **Rerank (+0.20 Completeness, +0.20 Faithfulness so với Hybrid):** Đóng vai trò cực kì quan trọng. Reranker đã thành công lôi các đoạn text "nặng kí" nhất về mặt ngữ cảnh lên trên đầu, giúp LLM trả lời đầy đủ ý hơn (Completeness cao nhất 3.30) và hạn chế bịa đặt (Faithfulness cao nhất 4.40). Hiện tượng giảm Answer Relevance có thể đến từ sự khắt khe của LLM Judge đối với câu trả lời Abstain (do Reranker nhận diện đúng các đoạn context không đủ dữ kiện).
> - **QExpand (Query Expansion):** Khi thêm chức năng này, kết quả lại sụt giảm (Faithfulness xuống 3.90, Completeness xuống 3.20). Có vẻ như Query Expansion đã sinh ra các từ khóa nhiễu (hallucinated sub-queries), làm trôi mất trọng tâm của câu hỏi ban đầu, kéo theo các chunk kém chất lượng lọt vào top K trước khi rerank.

**Kết luận:**
> Config tốt nhất hiện tại là **Hybrid + Rerank**. Sự kết hợp này mang lại độ trung thực (Faithfulness) và tính hoàn thiện (Completeness) tốt nhất mà không làm thay đổi thông điệp cốt lõi. Việc lạm dụng Query Expansion chưa mang lại lợi ích trong corpus này, cần phải điều chỉnh prompt cho LLM QExpand hoặc cân nhắc bỏ hẳn.

---

## Tóm tắt học được

> Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Hiện tượng Answer incompleteness (thiếu sót nhỏ) với dense retrieval, và sự sinh ra từ khóa rác khi áp dụng Query Expansion dễ dẫn đến Hallucination (Faithfulness giảm mạnh).

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > **Reranking** kết hợp trên nền Hybrid Search. Nó thực sự đóng vai trò là "chốt chặn" phân loại lại relevance cực kỳ đáng tin cậy.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > - Tắt Query Expansion và tập trung tuning tham số của Reranker (top_k_search lớn hơn ví dụ lấy 20 documents rồi rerank lấy 3).
   > - Chỉnh sửa kịch bản chunking: Tăng độ dài chunk_size để đảm bảo những tài liệu dạng bảng/chế độ không bị vỡ nát giữa các chunk.
