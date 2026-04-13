# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Lương Hữu Thành  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Trong tư cách Tech Lead, tôi chủ yếu tập trung vào **kết nối end-to-end giữa các Sprint** và điều phối để team có thể chạy được baseline/variant ổn định. Cụ thể, tôi đã:

- Lãnh đạo nhóm trong giai đoạn khởi động, phân công vai trò (Retrieval Owner, Eval Owner, Documentation Owner)
- Xây dựng và rà soát luồng chạy end-to-end: retrieval/generation trong `rag_answer.py` → đánh giá trong `eval.py` (scoring)
- Debug retrieval khi Q1, Q3, Q5 có độ chính xác thấp, chuyển từ dense sang hybrid + rerank
- Điều phối và review code từ các thành viên khác, đảm bảo cấu hình/format output thống nhất để dễ so sánh baseline vs variant

Công việc của tôi kết nối với các phần khác như: variant retrieval mà Retrieval Owner triển khai, test questions/scorecard mà Eval Owner thiết kế, và phần tổng hợp kết quả mà Documentation Owner báo cáo.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Tôi hiểu rõ hơn về **trade-off trong retrieval**. Dense retrieval (embedding) mạnh ở semantic relevance nhưng dễ miss exact keyword/tên riêng hoặc alias. Ví dụ Q7 hỏi "Approval Matrix" nhưng tài liệu dùng tên "Access Control SOP"; dense baseline hụt, trong khi hybrid (dense + BM25) và query expansion kéo đúng context. Điều này cho tôi thấy **không có “silver bullet”**: cần đặt giả thuyết rõ ràng, A/B test đúng “chỉ đổi một biến”, rồi dùng scorecard để quyết định variant nào thực sự cải thiện.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên:** Khi chạy eval.py lần đầu, LLM-as-Judge (gọi GPT-4o để chấm điểm Faithfulness, Relevance, ...) tốn **rất nhiều token** (5-6k token mỗi câu × 10 câu hỏi × 2 config = ~100k token). Dự tính ban đầu là dùng LLM judge 100% cho cả 10 câu, nhưng phải fallback xuống heuristic token-overlap scoring để tiết kiệm.

**Khó khăn:** Q9 (ERR-403-AUTH) là câu **"golden question"** để test khả năng abstain. Lần đầu, dense retrieval vẫn trả lại 3 chunks "liên quan" (từ security docs) rồi LLM model hallucinate thêm từ knowledge base. Phải thêm guardrail trong `rag_answer()`: 
```python
if top_score < 0.15:
    return "Không đủ dữ liệu..."
```
để buộc model "biết khi nào không biết".

---

## 4. Phân tích một câu hỏi trong scorecard

Câu hỏi: Q7 - "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

Phân tích:

Đây là câu hỏi **alias/tên cũ vs tên mới**: người dùng hỏi "Approval Matrix" nhưng tài liệu thực tế đã đổi tên thành "Access Control SOP".

- **Baseline (Dense):** Context Recall = 1/5 (chỉ tìm được khoảng 50% expected source). Dense embedding của "Approval Matrix" không khớp tốt với chunk nào từ "Access Control SOP" vì đây là hai tên gọi hoàn toàn khác nhau.

- **Root cause:** Dense/embedding chủ yếu dựa vào semantic similarity, nên xử lý alias/synonym chưa tốt nếu training data không cover cặp thuật ngữ này.

- **Variant (Hybrid + Rerank):** Context Recall = 5/5. Hybrid retrieval + query expansion (`transform_query("Approval Matrix") -> ["Approval Matrix", "access control sop", ...]`) giúp BM25 bắt từ khóa "access control" và kéo đúng chunk; sau đó rerank đánh giá lại để giữ đúng ngữ cảnh ở top.

- **Cải thiện:** Điểm (faithfulness + relevance) tăng từ 2/5 lên 4/5. Variant hoạt động tốt hơn với truy vấn có synonym hoặc deprecated terminology.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Thêm query decomposition cho Q10** (câu "nếu VIP thì khác không?") vì đây là multi-faceted question cần tách thành các sub-queries.

2. **Implement query-specific reranker** thay vì dùng chung một rerank config. Ví dụ: Q3 (Access Control) có thể dùng cross-encoder từ domain security, còn Q2 (Refund) dùng cross-encoder từ e-commerce.

3. **Active learning loop:** Thu thập các query mà model thường abstain hoặc có score thấp, sau đó bổ sung tài liệu hoặc tinh chỉnh retrieval strategy theo pattern lỗi.
