# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Tiến Thắng  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Tôi chịu trách nhiệm thiết kế và thực thi **Baseline Retrieval (Sprint 2)**:

- **Dense Retrieval Implementation:** Viết hàm `retrieve_dense()` trong `rag_answer.py` để tìm kiếm thông tin bằng vector similarity. Tôi đã gọi hàm `get_embedding(query, input_type="query")` để embed câu hỏi người dùng, sau đó query vào ChromaDB.
- **Xử lý kết quả trả về của ChromaDB:** Trích xuất `documents`, `metadatas`, `distances` từ ChromaDB, chuyển đổi distance (cosine distance) thành score (`1 - distance`) để có cái nhìn trực quan về độ tương đồng.
- **Lựa chọn các tham số tìm kiếm:** Đặt `top_k=10` (`TOP_K_SEARCH`) làm mức cơ sở để lấy các chunks tiềm năng trước khi truyền xuống cho prompt hoặc Reranker.
- Phối hợp với team Generation để thống nhất cấu trúc đầu ra của hàm retrieval (List các dict chứa `text`, `metadata`, `score`) để lắp ghép vào prompt.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**Vector Search (Dense Retrieval) làm việc trên "ngữ nghĩa" chứ không phải trên "từ khóa"**. 
Trước đây tôi nghĩ tìm kiếm chỉ là so khớp chữ, nhưng khi code `retrieve_dense`, tôi hiểu hệ thống nhúng (embed) câu hỏi "Làm sao để lấy lại tiền?" và chunk tài liệu "Quy trình hoàn tiền" thành các vector gần nhau trong không gian nhiều chiều, từ đó tính khoảng cách cosine (`collection.query(...)`) để lấy ra nội dung liên quan nhất.

**Top-K Trade-off:** Việc chọn số lượng tài liệu (`K`) lấy lên tạo ra sự đánh đổi lớn. Nếu lấy K nhỏ (ví dụ K=2), tính toán nhanh nhưng ta có thể sót thông tin. Lấy K quá lớn (ví dụ K=10) thì context window bị loãng, LLM bị nhiễu (lost in the middle) và tốn token. Điểm ngọt (sweet spot) cần được tinh chỉnh và cắt giảm phù hợp sau bước search rộng (như `TOP_K_SELECT = 3`).

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên:** Khoảng cách (distance) chuẩn của ChromaDB. Lúc ban đầu tôi dùng thẳng `distances` từ kết quả của ChromaDB để lập luận, nhưng thực ra với cosine similarity, distance càng nhỏ thì độ tương đồng lại càng cao. Tôi phải đảo ngược lại giá trị bằng công thức `score = 1 - distance` để báo cáo điểm tương đồng của chunk một cách logic và trực quan hơn.

**Khó khăn:** Dense retrieval thể hiện điểm yếu khi đối mặt với các câu hỏi tìm chính xác từ khóa (keyword/lexical search), như các mã lệnh hoặc mã ticket (ví dụ: "SLA ticket P1"). Do model embedding bị mất đi độ "sắc nét" của term cụ thể, đôi khi nó kéo về các SLA khác thay vì P1 SLA. Điều này là tiền đề bắt buộc phải nâng cấp lên Hybrid Retrieval.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** Q5 - "Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai?"

**Phân tích:**

Đây là một câu hỏi trực diện được quy định trong tài liệu kiểm soát truy cập (`access_control_sop.txt`).

- **Baseline Dense (do tôi build):**
  - Hàm `retrieve_dense()` đã thành công lấy được đoạn chunk chứa thông tin "khóa sau 5 lần đăng nhập sai liên tiếp".
  - Với câu hỏi này, câu chữ tự nhiên và gần gũi với nội dung văn bản gốc, do đó cosine similarity score rất cao và đoạn văn bản nằm ngay top 1 kết quả trả về, kéo ngữ cảnh đi vào prompt rất tốt.
  - Model sinh ra đáp án có độ chính xác cao và nguồn (source) rõ ràng.

- **Root Cause & Lỗi phát sinh:** Đối với retrieval baseline thì đây là best case scenario. Tuy nhiên, qua phân tích scorecard, tôi nhận ra lỗi sinh ra từ phía Generation khi model thỉnh thoảng bỏ sót chi tiết "liên tiếp" làm giảm điểm Faithfulness nhẹ. Lỗi này không đến từ việc lấy thiếu thông tin (Recall cao), mà thông tin bị LLM "lược bớt" khi tổng hợp chữ quá cô đọng.

- **Variant (Hybrid + Rerank):** Không có quá nhiều sự khác biệt đối với query ngắn này. Cả Dense và cụm tổ hợp Hybrid đều đủ sức kéo đúng chunk lên ưu tiên đầu.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Implement Hybrid Search (Dense + Sparse/BM25):** Tôi muốn tích hợp thêm kết quả tìm kiếm BM25 vào thẳng hàm `retrieve_dense` (để tạo thành Hybrid) nhằm khắc phục nhược điểm mất trí nhớ "từ khóa" đối với mã P1 SLA hay các con số chính xác.
2. **Metadata Filtering tự động:** Xây dựng một logic phân tích intent đơn giản phân loại câu hỏi (Ví dụ: intent = refund) để tự động gán metadata filter `{"source": "policy_refund"}` thẳng vào hàm `collection.query()` của ChromaDB trước khi truy xuất, giúp tăng độ chính xác lên tối đa.