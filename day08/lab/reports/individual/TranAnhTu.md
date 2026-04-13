# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Anh Tú  
**Vai trò trong nhóm:** Documentation & Indexing  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Trong project này, tôi tập trung vào việc xây dựng nền tảng Indexing và tài liệu kiến trúc:

- **Indexing Phase (Sprint 1):** 
    - Trực tiếp implement hàm `preprocess_document` trong `index.py` để bóc tách metadata quan trọng (Source, Department, Effective Date, Access) từ các file văn bản thô.
    - Xây dựng công cụ kiểm tra chất lượng index bao gồm `list_chunks()` để xem trước nội dung sau khi cắt và `inspect_metadata_coverage()` để thống kê độ bao phủ của metadata trên toàn bộ collection.
- **Architecture Documentation (docs/architecture.md):** 
    - Chịu trách nhiệm chính trong việc xây dựng file `architecture.md`.
    - Thiết kế sơ đồ luồng dữ liệu (RAG Pipeline flow) bằng Mermaid diagram, giải thích chi tiết các bước từ Indexing đến Retrieval và Generation.
    - Ghi lại các quyết định kỹ thuật về Chunking strategy (Head-based + paragraph fallback) và các tham số `chunk_size=400`, `overlap=80`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**"Architecture is the map, Indexing is the ground."**

Khi viết `architecture.md`, tôi hiểu sâu hơn về luồng đi của dữ liệu. Tuy nhiên, chỉ khi trực tiếp code phần `preprocess_document`, tôi mới thấy được sự phức tạp của việc làm sạch dữ liệu thực tế. 

Tôi hiểu rõ hơn về **Engineering Tradeoffs** trong chunking:
- Nếu chunk quá nhỏ: Mất ngữ cảnh (Context Loss).
- Nếu chunk quá lớn: Gây nhiễu cho LLM (Lost in the middle).
- Con số **400 tokens (~1600 ký tự)** là điểm cân bằng giúp giữ đủ thông tin cho một điều khoản policy mà vẫn đảm bảo hiệu năng retrieval.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên:** Tôi nhận ra rằng việc bóc tách metadata thủ công bằng Regex đòi hỏi sự chuẩn xác rất cao để bao quát được các biến số định dạng (ví dụ: xử lý key có dấu cách như 'Effective Date' hay các khoảng trắng dư thừa). Dù dữ liệu lab khá sạch, nhưng nếu bộ Filter không được thiết kế đủ chặt chẽ và linh hoạt, metadata sẽ rất dễ bị parse sai hoặc bị sót khi gặp các thay đổi nhỏ về trình bày.

**Khó khăn:** Việc đồng bộ hóa giữa code thực tế và tài liệu kiến trúc. Khi team thay đổi logic retrieval (ví dụ thêm Rerank), tôi phải cập nhật lại sơ đồ Mermaid và các bảng giải thích trong `architecture.md` để đảm bảo tài liệu luôn phản ánh đúng trạng thái của hệ thống.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** Q6 - "Escalation trong sự cố P1 diễn ra như thế nào?"

**Phân tích:**

- **Document:** sla_p1_2026.txt, section "=== P1 Escalation Protocol ==="
- **Expected:** "Auto escalate lên Senior Engineer nếu không phản hồi trong 10 phút"

- **Baseline Dense:**
  - Dense embedding tốt match → context recall = 5/5
  - Answer length moderate, structure clear
  - Faithfulness = 5/5, Relevance = 5/5, Completeness = 5/5
  - **"Model performance tuân thủ prompt"**

- **Variant (Hybrid + Rerank):** Kết quả tương tự (không đổi)

- **Documentation Insight:** Câu hỏi này là **positive baseline** — khi cả 2 variant perform equally well, nó cho signal rằng:
  - Retrieval đã tốt rồi (không cần hybrid/rerank)
  - Prompt đã clear (model không hallucinate)
  - Metadata đủ tốt (section tag giúp context)

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Refine Preprocessing logic:** Viết thêm các bộ parser thông minh hơn (có thể dùng LLM hoặc template matching) để xử lý các loại header phức tạp hơn trong hệ thống tài liệu nội bộ.
2. **Expand Architecture Docs:** Thêm các phần về "Error Handling" và "Retry Strategy" vào sơ đồ kiến trúc để project sẵn sàng hơn cho việc deployment thực tế.
3. **Automated Index QA:** Phát triển thêm script tự động verify tính đúng đắn của vector store sau mỗi lần rebuild index.