# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Vũ Phúc Thành  
**Vai trò trong nhóm:** Retrieval Owner (BM25/Hybrid Retrieval)  
**Ngày nộp:** 13/04/2026  

---

## 1. Tôi đã làm gì trong lab này?

Em phụ trách chủ yếu **Sprint 3 (tuning retrieval)**, tập trung vào **sparse retrieval bằng BM25** trong `day08/lab/rag_answer.py`. Cụ thể, em implement `retrieve_sparse()` để chạy keyword search: load các chunks từ ChromaDB, tokenize bằng regex để chuẩn hoá, sau đó dùng `rank_bm25.BM25Okapi` để chấm điểm và lấy top-k chunk liên quan. Em cũng thêm phương án fallback bằng lexical overlap để pipeline vẫn chạy được khi thiếu `rank_bm25`.

Ngoài BM25, em hỗ trợ **hybrid retrieval** bằng Reciprocal Rank Fusion trong `retrieve_hybrid()` (kết hợp dense + sparse) và tích hợp vào `rag_answer()` qua `retrieval_mode` (`dense`/`sparse`/`hybrid`) để nhóm có thể tạo variant và so sánh với baseline. Phần này nối trực tiếp với khâu **indexing/dense retrieval** (nguồn dữ liệu từ ChromaDB) và phục vụ **evaluation/scorecard** khi chạy A/B.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Em hiểu rõ hơn về **quản lý cấu hình và secrets** thay vì hard-code. Khi hard-code `NVIDIA_API_KEY` vào code, chương trình không chạy được trên máy khác, nên em chuyển sang `.env` + `dotenv` và thêm kiểm tra thiếu key để fail-fast. Em cũng rút ra rằng cần tách cấu hình theo môi trường (dev/test/prod), và không bao giờ commit secrets lên repo. Ngoài ra, việc tách provider embedding/LLM thành biến môi trường giúp pipeline linh hoạt hơn khi đổi nhà cung cấp hoặc khi API gặp sự cố.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**ChromaDB persistent client có thể gặp lỗi lock khi chạy đồng thời.**

Lần đầu có 2 thành viên chạy `index.py` cùng lúc dẫn tới database bị lock và treo. Em phải tìm cách xử lý lock và thêm cơ chế retry/timeout để giảm khả năng treo khi thao tác đồng thời.

**Khó khăn:** xung đột phiên bản thư viện (đặc biệt quanh `pydantic`). Em giải quyết bằng cách pin version rõ ràng trong `requirements.txt` để mọi người cài cùng môi trường.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** Q8 - "Nhân viên được làm remote tối đa mấy ngày mỗi tuần?"

**Phân tích:** Câu này có đáp án rõ trong `hr_leave_policy.txt` (“2 ngày/tuần, cần Team Lead phê duyệt”). Với baseline/variant của nhóm, retrieval lấy đúng chunk liên quan và LLM trả lời đúng nên điểm cao. Nếu câu này sai, em sẽ kiểm tra theo thứ tự: tài liệu HR có được index chưa, retrieval có trả đúng nguồn không, và cuối cùng là lỗi gọi LLM/embedding.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Containerization:** tạo `Dockerfile` + `docker-compose.yml` để setup bằng một lệnh, giảm lỗi “works on my machine”.

2. **Monitoring & Logging:** thêm structured logging xuyên suốt pipeline để dễ debug và theo dõi.

3. **Scalability:** viết roadmap khi scale lớn (ví dụ chuyển từ ChromaDB local sang vector DB dạng cluster như Milvus/Weaviate).